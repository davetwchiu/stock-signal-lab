"""Streamlit dashboard for Stock Signal Lab."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.backtest.engine import run_backtest
from src.backtest.signals import positions_from_regimes
from src.data.fetch import load_daily_data
from src.decision.config import DEFAULT_ADVANCED_OVERRIDE, load_decision_config, profile_settings
from src.decision.explain import ticker_explanation
from src.decision.report import generate_markdown_report, main_warning, portfolio_summary_text
from src.decision.table import action_counts, build_decision_table
from src.decision.user_benchmark import resolve_active_benchmark, save_user_benchmark
from src.decision.user_portfolio import (
    parse_portfolio_tickers,
    resolve_active_portfolio_tickers,
    save_user_portfolio,
    select_active_portfolio_frames,
)
from src.features.fourier import rolling_fourier_features
from src.features.regime import classify_regime
from src.features.technical import build_technical_features
from src.features.wavelet import rolling_wavelet_features
from src.ml.datasets import build_supervised_panel, feature_group_columns
from src.ml.diagnostics import build_ml_diagnostics
from src.ml.metrics import calibration_table, confusion_matrix_frame, score_quintile_analysis
from src.ml.models import MODEL_OPTIONS
from src.ml.scoring import current_ml_score_table
from src.ml.validation import compare_feature_groups, walk_forward_validate_classifier
from src.portfolio.allocation import AllocationConfig
from src.portfolio.risk import RiskControlConfig
from src.portfolio.simulator import simulate_portfolio
from src.robustness.ablation import run_feature_ablation
from src.robustness.runner import run_robustness_tests
from src.ui.charts import drawdown_chart, equity_curve_chart, feature_chart, price_chart
from src.ui.tables import current_regime_table, overview_table, relative_strength_ranking
from src.utils.config import FeatureConfig


st.set_page_config(page_title="Stock Signal Lab", layout="wide")

DECISION_HELP = {
    "ml_score": "A 0-100 read on relative strength and risk conditions. Higher means the setup looks more constructive; it is not a price target or guaranteed forecast.",
    "drawdown": "Estimated chance of a meaningful pullback over the forward review window.",
    "action": "A plain-English action label based on trend, opportunity, pullback risk, relative strength, and risk controls.",
}
DECISION_TABLE_COLUMN_LABELS = {
    "Rule-based regime": "Trend backdrop",
    "ML score": "Opportunity score",
    "Drawdown-risk probability": "Pullback risk",
    "Target exposure bucket": "Suggested position size",
    "One-line reason": "Reason",
}
BENCHMARK_OPTIONS = ["SPY", "QQQ", "SMH", "SOXX"]
MIN_ML_HEALTH_BUCKET_COUNT = 5


@st.cache_data(show_spinner=False)
def cached_load(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Streamlit cache wrapper around the local data cache/provider."""

    return load_daily_data(ticker, start=start, end=end, use_cache=True)


def parse_tickers(raw: str) -> list[str]:
    """Parse ticker input."""

    return parse_portfolio_tickers(raw)


def dataframe_to_csv(data: pd.DataFrame) -> bytes:
    """Encode a DataFrame for Streamlit download buttons."""

    return data.to_csv(index=False).encode("utf-8")


def parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats."""

    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def build_features_for_universe(
    frames: dict[str, pd.DataFrame],
    include_fourier: bool,
    include_wavelet: bool,
    feature_config: FeatureConfig,
) -> dict[str, pd.DataFrame]:
    """Build feature frames for every ticker."""

    benchmark_frames = {
        benchmark: frames[benchmark]
        for benchmark in ("SPY", "QQQ")
        if benchmark in frames
    }
    output: dict[str, pd.DataFrame] = {}
    for ticker, frame in frames.items():
        features = build_technical_features(frame, benchmark_frames=benchmark_frames, config=feature_config)
        if include_fourier:
            features = features.join(
                rolling_fourier_features(
                    frame["Adj Close"],
                    window=feature_config.fourier_window,
                    n_components=feature_config.fourier_components,
                    input_mode=feature_config.fourier_input,
                )
            )
        if include_wavelet:
            features = features.join(
                rolling_wavelet_features(
                    frame["Adj Close"],
                    window=feature_config.wavelet_window,
                    wavelet=feature_config.wavelet,
                    level=feature_config.wavelet_level,
                )
            )
        output[ticker] = classify_regime(features, use_signal_features=include_fourier or include_wavelet)
    return output


def latest_feature_table(feature_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return latest feature rows with ticker column."""

    rows: list[pd.DataFrame] = []
    for ticker, frame in feature_frames.items():
        if not frame.empty:
            row = frame.iloc[[-1]].copy()
            row.insert(0, "Ticker", ticker)
            rows.append(row.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def portfolio_curve_chart(curve: pd.DataFrame) -> go.Figure:
    """Plot portfolio and benchmark equity curves."""

    fig = go.Figure()
    if not curve.empty:
        fig.add_trace(go.Scatter(x=curve.index, y=curve["portfolio_equity"], name="Portfolio"))
        if "benchmark_equity" in curve:
            fig.add_trace(go.Scatter(x=curve.index, y=curve["benchmark_equity"], name="Benchmark"))
    fig.update_layout(title="Portfolio equity curve", height=330, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def score_panel_from_validation(out_predictions: pd.DataFrame, risk_predictions: pd.DataFrame) -> pd.DataFrame:
    """Merge outperformance and drawdown-risk predictions into score rows."""

    from src.ml.scoring import ml_score

    merged = out_predictions.merge(risk_predictions, on=["Date", "Ticker"], suffixes=("_out", "_risk"))
    if merged.empty:
        return pd.DataFrame()
    out_prob = merged["probability_out"]
    risk_prob = merged["probability_risk"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(merged["Date"]),
            "Ticker": merged["Ticker"],
            "ML Outperformance Probability": out_prob,
            "ML Drawdown-Risk Probability": risk_prob,
            "ML Score": ml_score(out_prob, risk_prob),
        }
    )


def display_decision_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with investor-facing column labels."""

    return table.copy().rename(columns=DECISION_TABLE_COLUMN_LABELS)


def _decision_display_column(table: pd.DataFrame, internal_column: str) -> str:
    """Resolve the visible column label used for styling."""

    display_column = DECISION_TABLE_COLUMN_LABELS.get(internal_column, internal_column)
    return display_column if display_column in table.columns else internal_column


def styled_decision_table(table: pd.DataFrame):
    """Return a lightly styled DataFrame for the cockpit."""

    if table.empty:
        return table
    score_column = _decision_display_column(table, "ML score")
    risk_column = _decision_display_column(table, "Drawdown-risk probability")
    return table.style.background_gradient(
        subset=[score_column],
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
    ).background_gradient(
        subset=[risk_column],
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )


def _numeric_or_none(value: object) -> float | None:
    """Return a numeric value when Streamlit diagnostics have one."""

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else None


def _format_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _format_signed_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.1%}"


def _diagnostic_bucket_row(score_buckets: pd.DataFrame, bucket: str) -> pd.Series | None:
    if score_buckets.empty or "score_bucket" not in score_buckets:
        return None
    bucket_rows = score_buckets[score_buckets["score_bucket"].astype(str) == bucket]
    return None if bucket_rows.empty else bucket_rows.iloc[0]


def _drawdown_risk_calibration_health(calibration: pd.DataFrame) -> str:
    required = {"average_probability", "observed_drawdown_risk_rate", "count"}
    if calibration.empty or not required.issubset(calibration.columns):
        return "Insufficient data"

    data = calibration.dropna(subset=["average_probability", "observed_drawdown_risk_rate"]).copy()
    data = data[data["count"] > 0].sort_values("average_probability")
    if len(data) < 2:
        return "Insufficient data"

    low_risk_rate = _numeric_or_none(data.iloc[0]["observed_drawdown_risk_rate"])
    high_risk_rate = _numeric_or_none(data.iloc[-1]["observed_drawdown_risk_rate"])
    if low_risk_rate is None or high_risk_rate is None:
        return "Insufficient data"

    risk_spread = high_risk_rate - low_risk_rate
    if risk_spread > 0.02:
        return "Rises with predicted risk"
    if risk_spread < -0.02:
        return "Looks inverted"
    return "Unclear"


def ml_signal_health_interpretation(diagnostics) -> tuple[str, str, pd.DataFrame]:
    """Summarize existing ML diagnostics without changing production logic."""

    score_buckets = diagnostics.score_buckets
    low_bucket = _diagnostic_bucket_row(score_buckets, "Low")
    high_bucket = _diagnostic_bucket_row(score_buckets, "High")
    calibration_health = _drawdown_risk_calibration_health(diagnostics.drawdown_risk_calibration)

    empty_metrics = pd.DataFrame(
        [
            {"Metric": "Top score bucket hit rate", "Value": "N/A"},
            {"Metric": "Bottom score bucket hit rate", "Value": "N/A"},
            {"Metric": "Top-minus-bottom hit-rate spread", "Value": "N/A"},
            {"Metric": "Top score bucket average forward return", "Value": "N/A"},
            {"Metric": "Bottom score bucket average forward return", "Value": "N/A"},
            {"Metric": "Return spread", "Value": "N/A"},
            {"Metric": "Drawdown-risk calibration", "Value": calibration_health},
        ]
    )
    if low_bucket is None or high_bucket is None:
        return (
            "Insufficient data",
            "The diagnostics-only score buckets are missing low or high bucket observations in this walk-forward sample.",
            empty_metrics,
        )

    low_count = _numeric_or_none(low_bucket.get("count"))
    high_count = _numeric_or_none(high_bucket.get("count"))
    top_hit_rate = _numeric_or_none(high_bucket.get("outperformance_hit_rate"))
    bottom_hit_rate = _numeric_or_none(low_bucket.get("outperformance_hit_rate"))
    top_return = _numeric_or_none(high_bucket.get("average_forward_return"))
    bottom_return = _numeric_or_none(low_bucket.get("average_forward_return"))

    hit_spread = (
        top_hit_rate - bottom_hit_rate
        if top_hit_rate is not None and bottom_hit_rate is not None
        else None
    )
    return_spread = (
        top_return - bottom_return
        if top_return is not None and bottom_return is not None
        else None
    )

    metrics = pd.DataFrame(
        [
            {"Metric": "Top score bucket hit rate", "Value": _format_percent(top_hit_rate)},
            {"Metric": "Bottom score bucket hit rate", "Value": _format_percent(bottom_hit_rate)},
            {"Metric": "Top-minus-bottom hit-rate spread", "Value": _format_signed_percent(hit_spread)},
            {"Metric": "Top score bucket average forward return", "Value": _format_percent(top_return)},
            {"Metric": "Bottom score bucket average forward return", "Value": _format_percent(bottom_return)},
            {"Metric": "Return spread", "Value": _format_signed_percent(return_spread)},
            {"Metric": "Drawdown-risk calibration", "Value": calibration_health},
        ]
    )

    if (
        low_count is None
        or high_count is None
        or low_count < MIN_ML_HEALTH_BUCKET_COUNT
        or high_count < MIN_ML_HEALTH_BUCKET_COUNT
    ):
        return (
            "Insufficient data",
            (
                "The low and high ML score buckets are too small for a useful diagnostics-only read "
                f"in this walk-forward sample (minimum {MIN_ML_HEALTH_BUCKET_COUNT} observations each)."
            ),
            metrics,
        )

    hit_positive = hit_spread is not None and hit_spread > 0
    return_positive = return_spread is not None and return_spread > 0
    if hit_positive and return_positive:
        verdict = "Healthy"
        reason = (
            "The high ML score bucket appears better than the low bucket on both hit rate and average "
            "forward return in this walk-forward sample."
        )
    elif hit_positive or return_positive:
        verdict = "Mixed"
        reason = (
            "The high ML score bucket appears better on one diagnostics-only outcome, but the other "
            "outcome is weak, noisy, or unavailable in this walk-forward sample."
        )
    else:
        verdict = "Weak"
        reason = (
            "The high ML score bucket does not appear to outperform the low bucket on hit rate or "
            "average forward return in this walk-forward sample."
        )

    if calibration_health == "Rises with predicted risk":
        reason += " Drawdown-risk calibration rises with predicted risk."
    elif calibration_health == "Looks inverted":
        reason += " Drawdown-risk calibration looks inverted, so treat the risk read as a caveat."
    elif calibration_health == "Unclear":
        reason += " Drawdown-risk calibration is unclear, so treat the risk read as a caveat."
    else:
        reason += " Drawdown-risk calibration has insufficient data."

    return verdict, reason, metrics


config = load_decision_config()

st.title("Stock Signal Lab")
st.caption("Decision support for regime, risk, and portfolio exposure. Research only, not financial advice.")

with st.sidebar:
    st.header("Decision Cockpit")
    profile_name = st.selectbox(
        "Profile",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
        help="Choose how conservative the suggested cash reserve and position sizes should be.",
    )
    advanced_override = st.toggle(
        "Show advanced settings",
        value=DEFAULT_ADVANCED_OVERRIDE,
        help="Show benchmark and date-range settings for the decision view.",
    )

    default_end = date.today()
    default_start = default_end - timedelta(days=365 * 3)
    active_tickers, saved_portfolio = resolve_active_portfolio_tickers(config.default_ticker_universe)
    tickers = list(active_tickers)
    benchmark = resolve_active_benchmark(config.default_benchmark, allowed_benchmarks=BENCHMARK_OPTIONS)
    start_date = default_start
    end_date = default_end

    if saved_portfolio is not None:
        st.caption(f"Using saved portfolio stock list: {len(active_tickers)} tickers")
    else:
        st.caption("No saved portfolio stock list found; using system default list")
    if st.session_state.get("portfolio_saved_message"):
        st.success(st.session_state.pop("portfolio_saved_message"))
    portfolio_text = st.text_area(
        "Portfolio stock list",
        value=", ".join(active_tickers),
        help="Paste comma-separated or newline-separated tickers. Click Save portfolio stock list to persist them locally.",
    )
    if st.button("Save portfolio stock list"):
        saved_tickers = parse_portfolio_tickers(portfolio_text)
        if saved_tickers:
            save_user_portfolio(saved_tickers)
            st.session_state["portfolio_saved_message"] = (
                f"Using saved portfolio stock list: {len(saved_tickers)} tickers"
            )
            cached_load.clear()
            st.rerun()
        else:
            st.error("Enter at least one ticker before saving.")

    if advanced_override:
        st.divider()
        benchmark = st.selectbox(
            "Benchmark",
            options=BENCHMARK_OPTIONS,
            index=BENCHMARK_OPTIONS.index(benchmark) if benchmark in BENCHMARK_OPTIONS else 0,
            help="Market benchmark used to compare relative strength.",
        )
        if st.session_state.get("benchmark_saved_message"):
            st.success(st.session_state.pop("benchmark_saved_message"))
        if st.button("Save benchmark"):
            save_user_benchmark(benchmark)
            st.session_state["benchmark_saved_message"] = f"Using saved benchmark: {benchmark}"
            st.rerun()
        start_date = st.date_input("Start date", value=default_start, help="Historical data start date.")
        end_date = st.date_input("End date", value=default_end, help="Historical data end date.")

data_tickers = list(tickers)
for required in {benchmark, "SPY", "QQQ"}:
    if required not in data_tickers:
        data_tickers.append(required)

profile = profile_settings(config, profile_name)
feature_config = FeatureConfig()

today_tab, explain_tab, research_tab = st.tabs(["Today / Decision Cockpit", "Explain / Why", "Research Lab"])

with st.spinner("Loading data and preparing today's decision view..."):
    frames: dict[str, pd.DataFrame] = {}
    load_errors: dict[str, str] = {}
    for ticker in data_tickers:
        try:
            frames[ticker] = cached_load(ticker, str(start_date), str(end_date))
        except Exception as exc:
            load_errors[ticker] = str(exc)

    feature_frames = build_features_for_universe(
        frames,
        include_fourier=True,
        include_wavelet=True,
        feature_config=feature_config,
    )

benchmark_frame = frames.get(benchmark)
decision_feature_frames = select_active_portfolio_frames(feature_frames, tickers)
if benchmark_frame is not None and not benchmark_frame.empty:
    supervised = build_supervised_panel(
        decision_feature_frames,
        benchmark_price=benchmark_frame["Adj Close"],
        horizon=config.default_label_horizon,
    )
    feature_columns = feature_group_columns(supervised, config.default_feature_group)
    current_scores = current_ml_score_table(
        supervised,
        decision_feature_frames,
        feature_columns=feature_columns,
        model_name=config.default_model,
        horizon=config.default_label_horizon,
    )
else:
    supervised = pd.DataFrame()
    feature_columns = []
    current_scores = pd.DataFrame()

latest_features = latest_feature_table(decision_feature_frames)
decision_table = build_decision_table(current_scores, latest_features, config, profile)
market_regime = ""
if benchmark in feature_frames and not feature_frames[benchmark].empty:
    market_regime = str(feature_frames[benchmark].iloc[-1].get("regime", ""))
suggested_cash = profile.cash_floor
suggested_gross = 1.0 - suggested_cash
summary_text = portfolio_summary_text(market_regime, decision_table, suggested_cash)
warning_text = main_warning(decision_table, market_regime)
report_markdown = generate_markdown_report(
    decision_table,
    profile=profile_name,
    benchmark=benchmark,
    market_regime=market_regime,
    suggested_gross_exposure=suggested_gross,
    suggested_cash_level=suggested_cash,
    summary_text=summary_text,
)

with today_tab:
    if load_errors:
        with st.expander("Data load warnings", expanded=False):
            for ticker, message in load_errors.items():
                st.warning(f"{ticker}: {message}")

    st.write(f"**Portfolio stock list:** {len(tickers)} tickers")
    st.write("**Tickers being reviewed:** " + ", ".join(tickers))

    counts = action_counts(decision_table)
    card_1, card_2, card_3, card_4 = st.columns(4)
    card_1.metric("Market regime", market_regime or "Unavailable", help="Rule-based regime for the selected benchmark.")
    risk_state = "Defensive" if "Downtrend" in market_regime or "elevated" in summary_text else "Normal"
    card_2.metric("Portfolio risk state", risk_state, help="Plain-English portfolio risk posture.")
    card_3.metric("Suggested gross exposure", f"{suggested_gross:.0%}", help="Maximum suggested invested exposure.")
    card_4.metric("Suggested cash level", f"{suggested_cash:.0%}", help="Suggested minimum cash reserve.")

    st.write(
        f"Actions: Add {counts['Add']} | Hold {counts['Hold']} | Trim {counts['Trim']} | "
        f"Exit {counts['Exit']} | Watch {counts['Watch']}"
    )
    st.info(summary_text)
    if warning_text != "No dominant warning.":
        st.warning(warning_text)

    st.subheader("Today's Decision Table")
    st.caption(
        "Opportunity score: "
        + DECISION_HELP["ml_score"]
        + " Pullback risk: "
        + DECISION_HELP["drawdown"]
    )
    displayed_decision_table = display_decision_table(decision_table)
    st.dataframe(styled_decision_table(displayed_decision_table), width="stretch")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Export today's decision table to CSV",
        dataframe_to_csv(decision_table),
        file_name="today_decision_table.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Export Decision Report to Markdown",
        report_markdown.encode("utf-8"),
        file_name="decision_report.md",
        mime="text/markdown",
    )

    with st.expander("Optional selected ticker price chart", expanded=False):
        ticker_options = decision_table["Ticker"].tolist() if not decision_table.empty else tickers
        selected_chart_ticker = st.selectbox("Ticker", ticker_options, key="decision_chart_ticker")
        if selected_chart_ticker in feature_frames:
            st.plotly_chart(
                price_chart(feature_frames[selected_chart_ticker], f"{selected_chart_ticker} price and moving averages"),
                width="stretch",
                key=f"decision_price_chart_{selected_chart_ticker}",
            )

with explain_tab:
    st.subheader("Explain / Why")
    if decision_table.empty:
        st.warning("No decision rows are available.")
    else:
        selected_ticker = st.selectbox(
            "Ticker",
            decision_table["Ticker"].tolist(),
            help="Choose a ticker to see the concise explanation behind its action.",
        )
        decision_row = decision_table[decision_table["Ticker"] == selected_ticker].iloc[0]
        feature_row = latest_features[latest_features["Ticker"] == selected_ticker].iloc[0]
        explanation = ticker_explanation(decision_row, feature_row)

        st.metric("Current suggested action", explanation["action"], help=DECISION_HELP["action"])
        st.metric("Target exposure", explanation["target_exposure"], help="Exposure bucket relative to the max allowed position size.")
        st.write("**Reason**")
        st.write(explanation["reason"])

        left, right = st.columns(2)
        with left:
            st.write("**Top bullish evidence**")
            for item in explanation["bullish_evidence"]:
                st.write(f"- {item}")
        with right:
            st.write("**Risk flags**")
            for item in explanation["bearish_evidence"]:
                st.write(f"- {item}")

        st.write("**What would change this**")
        for item in explanation["what_would_change"]:
            st.write(f"- {item}")

with research_tab:
    st.subheader("Research Lab")
    st.caption("Advanced diagnostics and parameter-heavy testing live here.")

    with st.expander("Market overview and rule-based baseline", expanded=False):
        st.dataframe(overview_table(feature_frames), width="stretch")
        regime_table = current_regime_table(feature_frames)
        st.dataframe(regime_table, width="stretch")
        ranking = relative_strength_ranking(regime_table, benchmark=benchmark)
        st.dataframe(ranking, width="stretch")

    research_ticker = st.selectbox(
        "Research ticker",
        list(feature_frames.keys()),
        index=list(feature_frames.keys()).index(tickers[0]) if tickers and tickers[0] in feature_frames else 0,
        help="Ticker used for research charts and single-name backtests.",
    )
    selected_features = feature_frames[research_ticker]
    target_positions = positions_from_regimes(selected_features["regime"])
    benchmark_price = frames.get(benchmark, pd.DataFrame()).get("Adj Close")
    baseline_result = run_backtest(
        selected_features["Adj Close"],
        target_positions,
        benchmark_price=benchmark_price,
        transaction_cost_bps=config.default_transaction_cost_bps,
        slippage_bps=config.default_slippage_bps,
    )

    with st.expander("Charts and baseline backtest", expanded=False):
        st.plotly_chart(
            price_chart(selected_features, f"{research_ticker} price and moving averages"),
            width="stretch",
            key=f"research_price_chart_{research_ticker}",
        )
        st.plotly_chart(
            drawdown_chart(selected_features["Adj Close"], f"{research_ticker} drawdown"),
            width="stretch",
            key=f"research_drawdown_chart_{research_ticker}",
        )
        st.plotly_chart(
            feature_chart(selected_features),
            width="stretch",
            key=f"research_feature_chart_{research_ticker}",
        )
        st.dataframe(baseline_result.summary, width="stretch")
        st.plotly_chart(
            equity_curve_chart(baseline_result.curve),
            width="stretch",
            key=f"research_equity_curve_chart_{research_ticker}",
        )

    with st.expander("Walk-forward validation and diagnostics", expanded=False):
        model_name = st.selectbox(
            "Model",
            options=list(MODEL_OPTIONS),
            format_func=lambda value: value.replace("_", " ").title(),
            help="Stage 2 model family. Decision Mode uses the locked default model.",
        )
        feature_group = st.selectbox(
            "Feature group",
            options=["technical", "technical_fourier", "technical_wavelet", "all"],
            format_func=lambda value: value.replace("_", " + ").title(),
            index=3,
            help="Feature group used for walk-forward validation.",
        )
        train_window = st.number_input("Train window", min_value=60, max_value=1260, value=504, step=21)
        test_window = st.number_input("Test window", min_value=20, max_value=252, value=63, step=21)
        step = st.number_input("Step size", min_value=20, max_value=252, value=63, step=21)
        embargo = st.number_input("Embargo gap", min_value=0, max_value=60, value=20, step=5)
        probability_threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05)
        show_ml_diagnostics = st.checkbox(
            "Show ML signal diagnostics",
            value=False,
            help="Run extra research-only diagnostics for existing walk-forward ML signal outputs.",
        )
        run_validation = st.button("Run walk-forward validation")

        if run_validation and benchmark_frame is not None and not supervised.empty:
            columns = feature_group_columns(supervised, feature_group)
            result = walk_forward_validate_classifier(
                supervised,
                columns,
                label_column=f"label_outperform_{config.default_label_horizon}d",
                model_name=model_name,
                train_window=int(train_window),
                test_window=int(test_window),
                step=int(step),
                embargo=int(embargo),
                probability_threshold=float(probability_threshold),
            )
            if result.predictions.empty:
                st.warning("No walk-forward predictions. Try a longer date range or smaller windows.")
            else:
                st.dataframe(result.overall_metrics, width="stretch")
                st.dataframe(result.fold_metrics, width="stretch")
                quintiles = score_quintile_analysis(result.predictions)
                st.dataframe(quintiles, width="stretch")
                st.dataframe(confusion_matrix_frame(result.predictions["actual"], result.predictions["probability"]), width="stretch")
                st.dataframe(calibration_table(result.predictions), width="stretch")

                group_options = {
                    name: feature_group_columns(supervised, name)
                    for name in ("technical", "technical_fourier", "technical_wavelet", "all")
                }
                comparison = compare_feature_groups(
                    supervised,
                    group_options,
                    label_column=f"label_outperform_{config.default_label_horizon}d",
                    model_name=model_name,
                    train_window=int(train_window),
                    test_window=int(test_window),
                    step=int(step),
                    embargo=int(embargo),
                    probability_threshold=float(probability_threshold),
                )
                st.dataframe(comparison, width="stretch")

                if show_ml_diagnostics:
                    try:
                        risk_result = walk_forward_validate_classifier(
                            supervised,
                            columns,
                            label_column=f"label_drawdown_risk_{config.default_label_horizon}d",
                            model_name=model_name,
                            train_window=int(train_window),
                            test_window=int(test_window),
                            step=int(step),
                            embargo=int(embargo),
                            probability_threshold=float(probability_threshold),
                        )
                        if risk_result.predictions.empty:
                            st.warning("No drawdown-risk predictions were available for ML diagnostics.")
                        else:
                            diagnostics = build_ml_diagnostics(
                                result.predictions,
                                risk_result.predictions,
                                result.overall_metrics,
                                risk_result.overall_metrics,
                            )
                            verdict, reason, health_metrics = ml_signal_health_interpretation(diagnostics)
                            st.write("**ML signal health**")
                            st.caption(
                                "Research-only interpretation of existing walk-forward diagnostics. "
                                "This does not affect Decision Mode, scores, actions, or position sizing."
                            )
                            verdict_message = f"**{verdict}** - {reason}"
                            if verdict == "Healthy":
                                st.success(verdict_message)
                            elif verdict == "Weak":
                                st.warning(verdict_message)
                            else:
                                st.info(verdict_message)
                            st.dataframe(health_metrics, width="stretch", hide_index=True)
                            st.write("**ML diagnostics summary**")
                            st.caption(
                                "Research-only diagnostics for existing walk-forward signal outputs. "
                                "These tables do not change Decision Mode logic."
                            )
                            st.dataframe(diagnostics.summary, width="stretch")
                            st.write("**ML score buckets**")
                            st.dataframe(diagnostics.score_buckets, width="stretch")
                            st.write("**Drawdown-risk calibration**")
                            st.dataframe(diagnostics.drawdown_risk_calibration, width="stretch")
                    except Exception as exc:
                        st.warning(f"ML diagnostics could not be built: {exc}")

    with st.expander("Feature ablation, portfolio simulation, and robustness", expanded=False):
        run_ablation = st.button("Run feature ablation")
        if run_ablation and not supervised.empty:
            ablation = run_feature_ablation(
                supervised,
                f"label_outperform_{config.default_label_horizon}d",
                model_name=config.default_model,
                train_window=252,
                test_window=63,
                step=63,
                embargo=config.default_label_horizon,
            )
            st.dataframe(ablation, width="stretch")

        run_portfolio = st.button("Run portfolio simulation")
        if run_portfolio and not supervised.empty:
            columns = feature_group_columns(supervised, config.default_feature_group)
            out_result = walk_forward_validate_classifier(
                supervised,
                columns,
                f"label_outperform_{config.default_label_horizon}d",
                model_name=config.default_model,
                train_window=252,
                test_window=63,
                step=63,
                embargo=config.default_label_horizon,
            )
            risk_result = walk_forward_validate_classifier(
                supervised,
                columns,
                f"label_drawdown_risk_{config.default_label_horizon}d",
                model_name=config.default_model,
                train_window=252,
                test_window=63,
                step=63,
                embargo=config.default_label_horizon,
            )
            score_panel = score_panel_from_validation(out_result.predictions, risk_result.predictions)
            simulation = simulate_portfolio(
                frames,
                score_panel,
                allocation_config=AllocationConfig(
                    max_position_size=profile.max_single_position_exposure,
                    cash_floor=profile.cash_floor,
                    max_gross_exposure=1.0 - profile.cash_floor,
                    drawdown_risk_threshold=profile.high_drawdown_risk_threshold,
                ),
                risk_config=RiskControlConfig(cash_floor=profile.cash_floor, max_gross_exposure=1.0 - profile.cash_floor),
                benchmark_features=feature_frames.get(benchmark),
                benchmark_price=benchmark_price,
                rebalance_frequency=config.default_rebalance_frequency,
                transaction_cost_bps=config.default_transaction_cost_bps,
                slippage_bps=config.default_slippage_bps,
            )
            st.dataframe(simulation.summary, width="stretch")
            st.plotly_chart(
                portfolio_curve_chart(simulation.curve),
                width="stretch",
                key="portfolio_simulation_equity_curve_chart",
            )
            st.dataframe(simulation.contribution, width="stretch")

        robustness_feature_groups = st.multiselect(
            "Robustness feature groups",
            options=["technical", "technical_fourier", "technical_wavelet", "all"],
            default=["technical", "all"],
        )
        robustness_horizons = st.multiselect("Label horizons", options=[10, 20, 60], default=[20])
        threshold_grid = st.text_input("ML threshold grid", value="0.55, 0.65")
        risk_grid = st.text_input("Risk threshold grid", value="0.40, 0.60")
        run_robustness = st.button("Run robustness test")
        if run_robustness:
            robustness_results, robustness_summary = run_robustness_tests(
                feature_frames,
                frames,
                {symbol: frames[symbol] for symbol in ("SPY", "QQQ", "SMH", "SOXX") if symbol in frames},
                feature_groups=robustness_feature_groups,
                horizons=[int(horizon) for horizon in robustness_horizons],
                ml_thresholds=parse_float_list(threshold_grid),
                drawdown_risk_thresholds=parse_float_list(risk_grid),
                transaction_costs_bps=[config.default_transaction_cost_bps],
                slippage_bps_values=[config.default_slippage_bps],
                model_name=config.default_model,
                train_windows=[252],
                test_window=63,
                step=63,
                embargo=config.default_label_horizon,
            )
            st.dataframe(robustness_summary, width="stretch")
            st.dataframe(robustness_results, width="stretch")
