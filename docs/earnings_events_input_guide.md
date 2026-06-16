# Earnings Events Input Guide

Stock Signal Lab can use a local earnings event file for research-only PEAD diagnostics.
PEAD means post-earnings announcement drift. It checks whether stocks tend to keep moving after earnings, and whether ML score behaves differently near earnings windows.

These diagnostics do not change production scoring, Decision Cockpit output, suggested actions, position size, ranking, allocation, persistence, or cache behavior.

## Where to Put the File

Use the example file as a starting point:

```text
data/research_inputs/earnings_events.example.csv
```

For real research runs, create your own local file here:

```text
data/research_inputs/earnings_events.csv
```

The real `earnings_events.csv` file is ignored by git so user-supplied research data does not get committed by accident.

PEAD diagnostics will stay `unavailable` until a real `data/research_inputs/earnings_events.csv` file is supplied.

## Columns

Only two columns are required:

- `ticker`: The symbol used by the app, such as `AAPL`, `MSFT`, or `TSM`.
- `earnings_date`: The earnings announcement date in `YYYY-MM-DD` format.

Optional columns:

- `report_timing`: When the company reported earnings.
- `eps_surprise_pct`: EPS surprise as a percent, if you have it.
- `revenue_surprise_pct`: Revenue surprise as a percent, if you have it.

Accepted `report_timing` values:

- `before_open`
- `after_close`
- `during_market`
- blank

Leave optional fields blank when you do not have reliable data.

## Data Quality Notes

Use verified earnings dates from a source you trust. Do not use guessed earnings dates for evidence, because guessed dates can make the PEAD result look more confident than it really is.

Tickers should match the app's ticker format as far as possible. The loader trims spaces and uppercases symbols, but it cannot know whether two different ticker formats mean the same security.

The example CSV is only a template. Its sample dates are illustrative and have not been validated as historical earnings dates.
