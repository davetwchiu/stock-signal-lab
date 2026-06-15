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
from src.decision.shortlist import SHORTLIST_VIEW_OPTIONS, filter_decision_shortlist
from src.decision.table import action_counts, build_decision_table
from src.decision.user_benchmark import resolve_active_benchmark, save_user_benchmark
from src.decision.user_portfolio import (
    create_user_portfolio_list,
    delete_user_portfolio_list,
    parse_portfolio_tickers,
    resolve_user_portfolio_lists,
    save_user_portfolio_list,
    save_user_portfolio_lists,
    set_active_user_portfolio_list,
    select_active_portfolio_frames,
)
from src.features.fourier import rolling_fourier_features
from src.features.regime import classify_regime
from src.features.technical import build_technical_features
from src.features.wavelet import rolling_wavelet_features
from src.ml.datasets import build_supervised_panel, feature_group_columns
from src.ml.diagnostics import (
    MLFeatureAudit,
    MLFeatureSignalDiagnostics,
    MLLabelAudit,
    build_ml_diagnostics,
    build_ml_feature_audit,
    build_ml_feature_signal_diagnostics,
    build_ml_label_audit,
)
from src.ml.interpretations import (
    DiagnosticInterpretation,
    ResearchLabRunInterpretation,
    interpret_drawdown_calibration,
    interpret_ml_diagnostics_summary,
    interpret_research_lab_run,
    interpret_ml_score_buckets,
    ml_signal_health_interpretation,
)
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
    "ml_score": "Higher means the relative setup looks more constructive under current trend and risk conditions.",
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


def diagnostic_export_frame(data: object) -> pd.DataFrame:
    """Normalize diagnostics-only outputs for CSV export."""

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame(
            [{"metric": key, "value": value} for key, value in data.items()]
        )
    if data is None:
        return pd.DataFrame()
    return pd.DataFrame([{"metric": "value", "value": data}])


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


def _format_percent(value: float) -> str:
    """Format a probability as a compact percentage for display."""

    if pd.isna(value):
        return ""
    percent = float(value) * 100
    if abs(percent - round(percent)) < 0.05:
        return f"{percent:.0f}%"
    return f"{percent:.1f}%"


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
    rank_column = _decision_display_column(table, "Relative strength rank")
    number_formats = {
        score_column: "{:.0f}",
        rank_column: "{:.0f}",
        risk_column: _format_percent,
    }
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
    ).format(
        {column: formatter for column, formatter in number_formats.items() if column in table.columns}
    )


def show_diagnostic_interpretation(interpretation: DiagnosticInterpretation) -> None:
    """Render display-only diagnostics interpretation text."""

    message = f"**{interpretation.label}** - {interpretation.message}"
    if interpretation.level == "success":
        st.success(message)
    elif interpretation.level == "warning":
        st.warning(message)
    else:
        st.info(message)


def show_research_lab_run_interpretation(interpretation: ResearchLabRunInterpretation) -> None:
    """Render a compact whole-run Research Lab interpretation."""

    message = "\n\n".join(
        [
            "**How to read this run**",
            f"**Overall:** {interpretation.overall}",
            f"**Walk-forward validation:** {interpretation.walk_forward_validation}",
            f"**ML score buckets:** {interpretation.ml_score_buckets}",
            f"**Drawdown-risk calibration:** {interpretation.drawdown_risk_calibration}",
            f"**Use:** {interpretation.use}",
        ]
    )
    if interpretation.level == "success":
        st.success(message)
    elif interpretation.level == "warning":
        st.warning(message)
    else:
        st.info(message)


def _show_label_audit_table(title: str, data: pd.DataFrame, empty_message: str) -> None:
    st.write(f"**{title}**")
    if data.empty:
        st.info(empty_message)
    else:
        st.dataframe(data, width="stretch", hide_index=True)


def show_ml_label_audit(audit: MLLabelAudit) -> None:
    """Render compact read-only supervised label diagnostics."""

    with st.expander("ML label audit", expanded=False):
        st.caption(
            "Diagnostics-only review of current supervised labels and existing forward outcome columns."
        )
        _show_label_audit_table(
            "Label prevalence summary",
            audit.prevalence_summary,
            "No active supervised label columns were available for this sample.",
        )
        _show_label_audit_table(
            "Return-label threshold sensitivity",
            audit.return_threshold_sensitivity,
            "Forward excess-return outcomes were unavailable for threshold sensitivity.",
        )
        _show_label_audit_table(
            "Drawdown-label threshold sensitivity",
            audit.drawdown_threshold_sensitivity,
            "Forward drawdown outcomes were unavailable for threshold sensitivity.",
        )
        _show_label_audit_table(
            "Label distribution by ticker",
            audit.ticker_distribution,
            "Ticker-level label distribution was unavailable for this sample.",
        )
        _show_label_audit_table(
            "Label distribution by regime",
            audit.regime_distribution,
            "Regime-level label distribution was unavailable for this sample.",
        )
        _show_label_audit_table(
            "Return-vs-drawdown label overlap",
            audit.label_overlap,
            "Return and drawdown-risk labels were not both available for overlap diagnostics.",
        )


def show_ml_feature_audit(audit: MLFeatureAudit) -> None:
    """Render compact read-only ML feature diagnostics."""

    with st.expander("ML feature audit and importance", expanded=False):
        st.caption(
            "Diagnostics-only review of selected model feature columns, sample shape, missingness, "
            "feature family mix, redundancy, and available fitted-model importance."
        )
        _show_label_audit_table(
            "Selected model feature inventory summary",
            audit.inventory_summary,
            "No feature inventory was available for this sample.",
        )
        _show_label_audit_table(
            "Feature family summary",
            audit.family_summary,
            "Feature names did not support a family summary.",
        )
        _show_label_audit_table(
            "Missingness and stability warnings",
            audit.warnings,
            "No feature warnings were available for this sample.",
        )
        _show_label_audit_table(
            "Redundancy summary",
            audit.redundancy_summary,
            "No redundancy summary was available for this sample.",
        )
        _show_label_audit_table(
            "Top highly correlated feature pairs",
            audit.high_correlation_pairs,
            "No high-correlation feature pairs were detected at the configured threshold.",
        )
        st.write("**Feature importance**")
        if audit.feature_importance.empty:
            st.info(
                "Feature importance is unavailable because the current validation output does not expose "
                "a fitted estimator with simple importance or coefficient attributes."
            )
        else:
            st.dataframe(audit.feature_importance, width="stretch", hide_index=True)


def show_ml_feature_signal_diagnostics(diagnostics: MLFeatureSignalDiagnostics) -> None:
    """Render compact read-only univariate feature signal diagnostics."""

    with st.expander("ML feature signal diagnostics", expanded=False):
        st.caption(
            "Diagnostics-only review of simple historical selected-feature relations to current supervised targets."
        )
        _show_label_audit_table(
            "Feature signal summary",
            diagnostics.signal_table,
            "No feature signal summary was available for this sample.",
        )
        _show_label_audit_table(
            "Feature family signal summary",
            diagnostics.family_summary,
            "No feature family signal summary was available for this sample.",
        )
        _show_label_audit_table(
            "Top quantile target spreads",
            diagnostics.quantile_summary,
            "Quantile target-spread diagnostics were unavailable for this sample.",
        )
        _show_label_audit_table(
            "Signal caution warnings",
            diagnostics.warnings,
            "No feature signal warnings were available for this sample.",
        )


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
    portfolio_lists = resolve_user_portfolio_lists(config.default_ticker_universe)
    portfolio_names = list(portfolio_lists.names)
    st.write("**Portfolio lists**")
    st.caption("Select a saved list, edit its tickers, then save changes to this device.")
    selected_portfolio_name = st.selectbox(
        "Active portfolio list",
        options=portfolio_names,
        index=portfolio_names.index(portfolio_lists.active.name),
        help="Switch between named portfolio lists. The active list is saved locally.",
    )
    if selected_portfolio_name != portfolio_lists.active.name:
        portfolio_lists = set_active_user_portfolio_list(portfolio_lists, selected_portfolio_name)
        save_user_portfolio_lists(portfolio_lists)
        st.rerun()

    active_portfolio = portfolio_lists.active
    active_tickers = list(active_portfolio.tickers)
    tickers = list(active_tickers)
    benchmark = resolve_active_benchmark(config.default_benchmark, allowed_benchmarks=BENCHMARK_OPTIONS)
    start_date = default_start
    end_date = default_end

    if portfolio_lists.source == "default":
        st.caption("No saved portfolio lists yet; using the system default list until you save one.")
    else:
        st.caption(f"Using saved list: {active_portfolio.name} ({len(active_tickers)} tickers)")
    if st.session_state.get("portfolio_saved_message"):
        st.success(st.session_state.pop("portfolio_saved_message"))
    st.write("**Edit active list**")
    portfolio_text = st.text_area(
        "Tickers",
        value=", ".join(active_tickers),
        key=f"portfolio_text_{active_portfolio.name}",
        help="Enter comma-separated or newline-separated tickers for the active list.",
    )
    if st.button("Save active list"):
        saved_tickers = parse_portfolio_tickers(portfolio_text)
        if saved_tickers:
            portfolio_lists = save_user_portfolio_list(
                portfolio_lists,
                active_portfolio.name,
                saved_tickers,
            )
            save_user_portfolio_lists(portfolio_lists)
            st.session_state["portfolio_saved_message"] = (
                f"Saved {active_portfolio.name}: {len(saved_tickers)} tickers"
            )
            cached_load.clear()
            st.rerun()
        else:
            st.error("Enter at least one ticker in the active list before saving.")
    st.write("**Create or delete lists**")
    new_portfolio_name = st.text_input(
        "New list name",
        value="",
        placeholder="Core",
        help="Create a named list from the tickers currently shown above.",
    )
    create_col, delete_col = st.columns(2)
    with create_col:
        if st.button("Create list", help="Create a new named list using the tickers above."):
            try:
                new_tickers = parse_portfolio_tickers(portfolio_text)
                portfolio_lists = create_user_portfolio_list(
                    portfolio_lists,
                    new_portfolio_name,
                    new_tickers,
                )
                save_user_portfolio_lists(portfolio_lists)
                st.session_state["portfolio_saved_message"] = f"Created {portfolio_lists.active.name}"
                st.rerun()
            except ValueError as error:
                st.error(f"Could not create list: {error}")
    with delete_col:
        if st.button(
            "Delete active list",
            disabled=len(portfolio_names) <= 1,
            help="Remove the selected list. At least one list must remain.",
        ):
            try:
                deleted_name = active_portfolio.name
                portfolio_lists = delete_user_portfolio_list(portfolio_lists, deleted_name)
                save_user_portfolio_lists(portfolio_lists)
                st.session_state["portfolio_saved_message"] = f"Deleted {deleted_name}"
                cached_load.clear()
                st.rerun()
            except ValueError as error:
                st.error(f"Could not delete list: {error}")

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
        "Opportunity score is a 0-100 ranking aid, not a price target, expected return, or buy probability. "
        + DECISION_HELP["ml_score"]
        + " Pullback risk: "
        + DECISION_HELP["drawdown"]
    )
    cockpit_view = st.selectbox(
        "Cockpit view",
        SHORTLIST_VIEW_OPTIONS,
        help="Filter the displayed rows without changing the calculated Decision Cockpit table.",
    )
    filtered_decision_table = filter_decision_shortlist(decision_table, cockpit_view)
    st.caption(
        f"Showing {len(filtered_decision_table)} of {len(decision_table)} rows. "
        "This display filter does not change scores, ranking, actions, or position sizing."
    )
    displayed_decision_table = display_decision_table(filtered_decision_table)
    if displayed_decision_table.empty:
        st.info("No rows match this cockpit view.")
    else:
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
                    st.info(
                        "ML signal health is research-only. It checks whether past out-of-sample "
                        "ML predictions separated stronger and weaker signals. The labels are historical "
                        "diagnostics only. This panel does not affect Decision Mode, scores, sizing, "
                        "ranking, allocation, saved portfolio, benchmark, or cache."
                    )
                    try:
                        show_ml_label_audit(
                            build_ml_label_audit(
                                supervised,
                                horizon=config.default_label_horizon,
                            )
                        )
                        feature_audit = build_ml_feature_audit(
                            supervised,
                            columns,
                        )
                        show_ml_feature_audit(feature_audit)
                        show_ml_feature_signal_diagnostics(
                            build_ml_feature_signal_diagnostics(
                                supervised,
                                columns,
                                horizon=config.default_label_horizon,
                                high_correlation_pairs=feature_audit.high_correlation_pairs,
                            )
                        )
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
                        risk_adjusted_result = walk_forward_validate_classifier(
                            supervised,
                            columns,
                            label_column=f"label_risk_adjusted_outperform_{config.default_label_horizon}d",
                            model_name=model_name,
                            train_window=int(train_window),
                            test_window=int(test_window),
                            step=int(step),
                            embargo=int(embargo),
                            probability_threshold=float(probability_threshold),
                        )
                        tail_risk_result = walk_forward_validate_classifier(
                            supervised,
                            columns,
                            label_column=f"label_tail_risk_adjusted_outperform_{config.default_label_horizon}d",
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
                                baseline_panel=supervised,
                                risk_adjusted_predictions=risk_adjusted_result.predictions,
                                tail_risk_predictions=tail_risk_result.predictions,
                            )
                            show_research_lab_run_interpretation(
                                interpret_research_lab_run(
                                    diagnostics.summary,
                                    diagnostics.score_buckets,
                                    diagnostics.drawdown_risk_calibration,
                                )
                            )
                            verdict, reason, health_metrics = ml_signal_health_interpretation(diagnostics)
                            st.write("**ML signal health**")
                            st.caption(
                                "Research-only interpretation of existing walk-forward diagnostics. "
                                "This does not affect Decision Mode, scores, sizing, ranking, or allocation."
                            )
                            verdict_message = f"**{verdict}** - {reason}"
                            if verdict == "Healthy":
                                st.success(verdict_message)
                            elif verdict == "Weak":
                                st.warning(verdict_message)
                            else:
                                st.info(verdict_message)
                            st.dataframe(health_metrics, width="stretch", hide_index=True)
                            health_export = diagnostic_export_frame(health_metrics)
                            if not health_export.empty:
                                st.download_button(
                                    "Download health CSV",
                                    dataframe_to_csv(health_export),
                                    file_name="ml_signal_health.csv",
                                    mime="text/csv",
                                    key="download_ml_signal_health_csv",
                                )
                            st.write("**ML diagnostics summary**")
                            st.caption(
                                "Research-only diagnostics for existing walk-forward signal outputs. "
                                "These tables do not change Decision Mode logic."
                            )
                            st.dataframe(diagnostics.summary, width="stretch")
                            summary_export = diagnostic_export_frame(diagnostics.summary)
                            if not summary_export.empty:
                                st.download_button(
                                    "Download summary CSV",
                                    dataframe_to_csv(summary_export),
                                    file_name="ml_diagnostics_summary.csv",
                                    mime="text/csv",
                                    key="download_ml_diagnostics_summary_csv",
                                )
                            show_diagnostic_interpretation(interpret_ml_diagnostics_summary(diagnostics.summary))
                            st.write("**ML score buckets**")
                            st.dataframe(diagnostics.score_buckets, width="stretch")
                            score_buckets_export = diagnostic_export_frame(diagnostics.score_buckets)
                            if not score_buckets_export.empty:
                                st.download_button(
                                    "Download buckets CSV",
                                    dataframe_to_csv(score_buckets_export),
                                    file_name="ml_score_buckets.csv",
                                    mime="text/csv",
                                    key="download_ml_score_buckets_csv",
                                )
                            show_diagnostic_interpretation(interpret_ml_score_buckets(diagnostics.score_buckets))
                            st.write("**ML baseline comparison**")
                            st.caption(
                                "Diagnostics-only comparison of existing ML score bucket separation "
                                "against no-skill and simple price-derived baselines."
                            )
                            if diagnostics.baseline_comparison.empty:
                                st.info("No baseline comparison was available for this walk-forward sample.")
                            else:
                                st.dataframe(diagnostics.baseline_comparison, width="stretch")
                            st.write("**ML target comparison: v1 vs v2 vs tail-risk v3**")
                            st.caption(
                                "Diagnostics-only comparison of the current target, the risk-adjusted "
                                "relative target, and the tail-risk target. This does not replace production ML scoring."
                            )
                            if diagnostics.target_comparison.empty:
                                st.info("No target comparison was available for this walk-forward sample.")
                            else:
                                st.dataframe(
                                    diagnostics.target_comparison,
                                    width="stretch",
                                    hide_index=True,
                                )
                            st.write("**ML score direction diagnostics**")
                            st.caption(
                                "Diagnostics-only checks for score direction, label alignment, bucket "
                                "monotonicity, inverted-score separation, and regime-specific direction."
                            )
                            if diagnostics.score_direction_summary.empty:
                                st.info("No score-direction summary was available for this walk-forward sample.")
                            else:
                                st.dataframe(
                                    diagnostics.score_direction_summary,
                                    width="stretch",
                                    hide_index=True,
                                )
                            if diagnostics.probability_label_alignment.empty:
                                st.info("No probability or label alignment table was available.")
                            else:
                                st.dataframe(diagnostics.probability_label_alignment, width="stretch")
                            if diagnostics.score_bucket_monotonicity.empty:
                                st.info("No score-bucket monotonicity table was available.")
                            else:
                                st.dataframe(diagnostics.score_bucket_monotonicity, width="stretch")
                            if diagnostics.score_inversion.empty:
                                st.info("No score inversion comparison was available.")
                            else:
                                st.dataframe(diagnostics.score_inversion, width="stretch")
                            if diagnostics.regime_score_direction.empty:
                                st.info("No regime score-direction summary was available.")
                            else:
                                st.dataframe(diagnostics.regime_score_direction, width="stretch")
                            st.write("**Regime-segmented ML diagnostics**")
                            st.caption(
                                "Diagnostics-only view of existing ML score bucket separation grouped "
                                "by available regime labels."
                            )
                            if diagnostics.regime_segmented.empty:
                                st.info(
                                    "No regime-segmented diagnostics were available for this walk-forward sample."
                                )
                            else:
                                st.dataframe(diagnostics.regime_segmented, width="stretch")
                            st.write("**Drawdown-risk calibration**")
                            st.dataframe(diagnostics.drawdown_risk_calibration, width="stretch")
                            drawdown_calibration_export = diagnostic_export_frame(
                                diagnostics.drawdown_risk_calibration
                            )
                            if not drawdown_calibration_export.empty:
                                st.download_button(
                                    "Download calibration CSV",
                                    dataframe_to_csv(drawdown_calibration_export),
                                    file_name="drawdown_risk_calibration.csv",
                                    mime="text/csv",
                                    key="download_drawdown_risk_calibration_csv",
                                )
                            show_diagnostic_interpretation(
                                interpret_drawdown_calibration(diagnostics.drawdown_risk_calibration)
                            )
                            st.write("**Drawdown-risk calibration quality**")
                            if diagnostics.drawdown_risk_calibration_quality.empty:
                                st.info(
                                    "Calibration quality metrics were unavailable for this walk-forward sample."
                                )
                            else:
                                st.dataframe(
                                    diagnostics.drawdown_risk_calibration_quality,
                                    width="stretch",
                                    hide_index=True,
                                )
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
