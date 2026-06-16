# Codex ML Iteration Protocol

This protocol is for evidence-driven Research Lab work only. It is not permission to change production ML scoring, targets, labels, Decision Cockpit behavior, allocation, persistence, cache behavior, or dependencies.

## Required Loop

1. Run headless Research Lab baseline export.
2. Read `codex_handoff.md` and key CSV files.
3. State evidence summary before changing code.
4. Propose one coherent enhancement.
5. Make bounded code changes.
6. Run focused tests and full pytest.
7. Run headless Research Lab candidate export.
8. Compare baseline vs candidate export.
9. Commit only if tests pass and diagnostics are not worse overall.
10. If evidence worsens or is mixed, stop and report instead of forcing a win.

## Stop Rules

- Max iterations per Codex task: 2.
- No production target switch unless explicitly instructed.
- No `ml_score()` change unless explicitly instructed.
- No Decision Cockpit gating unless explicitly instructed.
- No new dependencies.
- No broad refactor.
- No generated research run outputs committed.
- If one metric improves but calibration or regime stability worsens, call it mixed, not success.
- If full pytest fails, do not commit.
- If headless Research Lab runner fails, do not continue algorithm iteration.

## Objective Templates

`ml_target_research`: production target should remain baseline unless explicitly instructed; alternative target is improved only if bucket spread improves or remains positive, calibration gap does not materially worsen, regime inversion count does not increase, Brier score does not materially worsen, and evidence is stable across feature groups or clearly marked feature-group dependent.

`feature_engineering_research`: outperformance AUC / PR-AUC are not materially worse, bucket spread is not materially worse, unstable / missing / redundant features do not increase, and drawdown-risk calibration does not deteriorate.

`calibration_research`: absolute calibration gap and Brier score improve or are not materially worse, bucket spread does not invert, and regime stability does not worsen.

`regime_reliability_research`: regime inversion and regime-sensitive counts do not increase, overall AUC / PR-AUC / bucket spread are not materially worse, and feature-group consistency does not deteriorate.

## Heuristic Tolerances

- AUC deterioration tolerance: 0.005.
- Brier deterioration tolerance: 0.02.
- Absolute calibration gap deterioration tolerance: 0.03.
- Bucket spread deterioration tolerance: 0.005.

These are conservative heuristics, not statistical proof.
