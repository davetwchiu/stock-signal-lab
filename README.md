# Stock Signal Lab

Stock Signal Lab is a local-first Python and Streamlit MVP for researching stock time-series features, interpretable regime labels, risk flags, and no-lookahead backtests.

This project is for research and testing only. It is not a trading system, it does not predict stock prices, and it should not be used as financial advice.

## What The MVP Does

- Loads daily OHLCV data from `yfinance` or local CSV files.
- Caches downloaded data locally in `data/cache/`.
- Computes technical, Fourier, and wavelet features using rolling windows.
- Labels each ticker with an interpretable regime.
- Runs a simple close-to-close backtest with signals shifted by one trading day.
- Compares strategy results against buy-and-hold and an optional benchmark.
- Adds a Stage 2 ML research layer for out-of-sample feature-group comparison.
- Provides a basic Streamlit dashboard with CSV exports.

## Install

```bash
cd stock-signal-lab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If `python3.11` is not installed, use any Python version `>=3.11`.

## Run

```bash
streamlit run app.py
```

The default ticker universe is:

`NVDA, TSM, AMD, AVGO, MU, LITE, COHR, CRDO, XLK, SMH, SOXX, QQQ, SPY`

## Local CSV Format

CSV loading expects these columns:

`Date, Open, High, Low, Close, Adj Close, Volume`

The loader normalizes dates, sorts rows, removes duplicate dates, and keeps a daily `DatetimeIndex`.

## Plain-English Feature Notes

Fourier features decompose a rolling window of returns or log prices into cyclic components. In this MVP, the app records the strongest frequencies, their amplitudes, and how concentrated the spectral energy is. These features can help describe whether recent movement is dominated by a few repeatable cycles or is more diffuse.

Wavelet features decompose a rolling price window across multiple time scales. This MVP records energy by scale, a denoised trend estimate, and short-scale noise intensity. These features can help separate slow trend behavior from short-term turbulence.

Both feature families are computed with rolling windows ending at date `t`. They do not use data after date `t`.

## Backtest Assumptions

- Signals are generated from information available through date `t`.
- Positions are shifted forward by one trading day.
- Returns are daily close-to-close returns.
- Transaction cost and slippage are modeled as basis-point costs on turnover.
- The default regime strategy is long during uptrend regimes and flat during distribution/downtrend regimes.

## Stage 2 ML Research Layer

Stage 2 adds an advisory machine-learning layer for research. It does not replace the rule-based regime classifier. The rule-based classifier remains the baseline for comparison.

The ML layer compares feature groups:

- Technical features only.
- Technical + Fourier features.
- Technical + wavelet features.
- Technical + Fourier + wavelet features.

The main research question is whether signal-processing features add useful information after strict walk-forward validation.

Stage 2 uses simple scikit-learn classifiers:

- Logistic regression with regularization.
- Random forest classifier.
- Histogram gradient boosting classifier.

The models are wrapped in sklearn `Pipeline` objects with imputation and scaling where appropriate. This keeps preprocessing inside each fold and avoids leakage-prone manual preprocessing.

### Labels

Labels are forward-looking targets and must never be model inputs.

- `label_outperform_20d`: `1` when the ticker's future 20-trading-day return beats the benchmark's future 20-trading-day return by more than the configured threshold. The default threshold is `+2%`.
- `label_drawdown_risk_20d`: `1` when the ticker's forward 20-trading-day drawdown is worse than the configured threshold. The default threshold is `-10%`.
- `label_regime_deterioration_20d`: `1` when the rule-based regime deteriorates into Distribution or Downtrend / high risk during the next 20 trading days.

Features at date `t` use only data available through date `t`. Labels use future data only to define the supervised target, and label/forward-outcome columns are explicitly excluded from the feature matrix.

### Walk-Forward Validation

Financial time series should not use random train/test splits because random splits mix past and future observations and can make a model look stronger than it is. Stage 2 uses date-ordered walk-forward validation:

- Train on an earlier rolling window.
- Optionally skip an embargo gap to reduce overlap from forward labels.
- Test only on later unseen dates.
- Save fold predictions, probabilities, labels, dates, and tickers.

Classification metrics include accuracy, precision, recall, F1, ROC AUC where valid, PR AUC where valid, confusion matrix, and calibration buckets. Investment-usefulness diagnostics include forward return, forward excess return, forward drawdown, and hit rate by model-score quintile.

### ML Score

The current ML score is an advisory 0-100 score combining:

- Probability of 20-day benchmark outperformance.
- Probability of 20-day drawdown risk.

Higher scores mean the simple research model sees a better balance of opportunity and risk under the current selected feature group. This is not a price prediction and not a trading recommendation.

## Stage 3 Portfolio Lab

Stage 3 adds robustness testing, portfolio simulation, decision-support reports, and local experiment logs. It is still a research tool. It does not connect to brokerage accounts, place trades, optimize a portfolio, or automatically pick stocks.

The portfolio simulator uses transparent allocation rules:

- Higher ML scores allow a larger fraction of the configured max position size.
- High drawdown-risk probability reduces or exits exposure.
- Weak rule-based regimes reduce exposure.
- Cash floor, max gross exposure, max single-position size, benchmark de-risking, and portfolio drawdown de-risking are enforced as guardrails.
- Target weights are lagged by one trading day before returns are earned.

Suggested actions mean:

- `Add`: target weight is meaningfully above current or zero weight.
- `Hold`: current and target weights are close.
- `Trim`: target weight is meaningfully below current weight.
- `Exit`: target is zero while current weight is non-trivial.
- `Watch`: target is zero and there is no current position.

Robustness testing runs the same research logic across assumptions such as feature groups, horizons, thresholds, costs, benchmarks, and train windows. The goal is to detect whether apparent value is stable or dependent on narrow settings. Feature-ablation results are summarized as `adds value`, `mixed`, `no clear value`, or `likely overfit`.

Decision reports can be exported as CSV, Markdown, or HTML from the dashboard. Experiment logs are written locally under `data/experiments/`; no database is required.

Stage 3 remains vulnerable to overfitting through repeated experiments. Treat good-looking simulations as hypotheses for further testing, not evidence that a trading strategy will work.

## Stage 3.5 Decision Mode

Stage 3.5 simplifies the app into three top-level tabs:

- `Today / Decision Cockpit`: the default landing page with one clear action table and portfolio summary.
- `Explain / Why`: concise per-ticker explanation of the current action.
- `Research Lab`: advanced diagnostics, walk-forward validation, ablation, robustness, and backtest tools.

Decision Mode uses locked defaults from `config/default_decision_mode.yaml`. The sidebar exposes only simple preset profiles by default: Conservative, Balanced, and Aggressive. Parameter overrides are hidden unless `Advanced override` is enabled; deeper tuning remains in Research Lab.

See `docs/decision_mode.md` for action labels, exposure buckets, confidence, ML score interpretation, and limitations.

## Important Limitations

In-sample backtests are not evidence of predictive power. They are useful for debugging assumptions, measuring behavior, and generating research hypotheses, but they can easily overfit. Any future predictive model should be evaluated out of sample, with walk-forward validation, realistic costs, and strict no-lookahead controls.

Stage 2 model comparisons can still overfit through repeated experimentation, threshold tuning, and feature selection. Do not treat the best-looking run as evidence of durable predictive power without additional out-of-sample validation.

Data quality depends on the provider. `yfinance` is convenient for MVP testing, but it is not a professional market data feed. The provider interface is intentionally small so Polygon, Tiingo, Alpaca, IBKR, or another data source can be added later without rewriting the feature or backtest code.

No API keys are hard-coded. Future provider keys should be read from environment variables.

## Run Tests

```bash
pytest
```

## Roadmap

- Add richer provider adapters for paid data sources.
- Add more validation reports and parameter sweeps.
- Add model-ready notebooks and richer explainability reports.
- Add portfolio-level research tools.
- Add notebook examples for feature interpretation.
- Add stricter data-quality diagnostics.
