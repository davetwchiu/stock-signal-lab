# Stock Signal Lab — Codex Working Instructions

## Project status

This repo is currently at a working Stage 3.5 baseline.

Stable baseline:
- Git tag: stage3_5_stable
- Baseline commit: de0ab28
- Repo root: /Users/davidchiu/Documents/New project 2/stock-signal-lab

The app is a local Streamlit stock research and decision-support tool.
It is not a live trading system.

## Critical lesson from prior failed work

Previous Codex work became unstable because prompts mixed too many tasks at once:
- UI changes
- config persistence
- portfolio/watchlist persistence
- ML audit gates
- drawdown-risk model logic
- cache invalidation
- reporting changes

Do not repeat this.

Every task must be small, single-purpose, and reviewable.

## Working rule

One Codex task = one narrow change.

Do not combine:
- UI wording changes
- watchlist persistence
- benchmark persistence
- ML logic
- model labels
- probability mapping
- cache changes
- Research Lab changes
- report/export changes

If a user request implies multiple changes, stop and propose a step-by-step plan before editing code.

## Default safety rule

Do not touch ML internals unless the task explicitly asks for it.

Do not modify:
- ML training
- labels
- probability mapping
- ML score formula
- drawdown-risk probability logic
- backtesting
- portfolio simulation
- allocation mechanics
- cache logic

unless the prompt specifically names one of these as the task.

## Required workflow before coding

Before modifying files:
1. Inspect the relevant files.
2. State the exact files you intend to modify.
3. State what is out of scope.
4. Keep the change minimal.

## Required workflow after coding

After modifying files:
1. Report exact files changed.
2. Report tests run.
3. Report whether any tests failed.
4. Report how to manually verify in Streamlit.
5. Do not claim success unless tests pass or failures are clearly explained.

## Shared loop-engineering baseline

A Codex turn ending is not completion. Completion requires evidence from the applicable gates below.

Use this execution loop for every task:
1. Inspect the repo instructions and relevant files.
2. Restate the acceptance criteria, exact edit surface, and out-of-scope areas.
3. Implement the smallest coherent change.
4. Run targeted checks for the changed behavior.
5. Run the full applicable test suite and relevant formatter, lint, type-check, schema, build, or generated-file checks.
6. Review the complete diff against every acceptance criterion and project constraint.
7. Repair failures and re-test, with at most three repair cycles.
8. If the same failure repeats twice without meaningful new evidence, stop and report the exact blocker, commands run, relevant failing output, and likely root cause. Do not claim completion.

Completion gates:
- All acceptance criteria are implemented, including relevant error and edge paths.
- Targeted tests pass.
- The full applicable suite and relevant validation checks pass; explain anything unavailable or intentionally skipped.
- `git diff --check` passes.
- Documentation, schemas, fixtures, and generated files are updated when affected.
- The final diff contains no unintended changes and the working tree is in the expected state.
- Commit or push only when explicitly requested and only after the gates pass.

Keep the active context concise. Save long logs to `/tmp` or a repo-local report, quote only relevant failing lines, and after each repair cycle summarize the current state, remaining failure, and next hypothesis. Use the fewest clear tools needed and keep repeatable actions idempotent where practical.

The final report must include changed files, checks and results, branch, commit SHA when committed, push status when requested, working-tree state, manual verification when relevant, and residual risks.

## Git discipline

Before risky changes, confirm current baseline:

git log --oneline --decorate -5
git status --short

The stable restore point is:

git reset --hard stage3_5_stable

Do not delete tags, branches, or backup files unless explicitly asked.

## Current priorities

Near-term improvements should be done in this order, one at a time:

1. Persistent portfolio stock list.
2. Better one-line reason in Today Decision Table.
3. Display-only UI label renames.
4. Persistent market benchmark.
5. Ticker compatibility / failed ticker reporting.
6. Decision report export polish.

Do not skip ahead to ML audit or model logic unless explicitly instructed.

## Product intent

Decision Mode should be a simple investment decision cockpit.

It should help the user answer:
- Which names are healthy?
- Which names are weakening?
- Which names are Add / Hold / Trim / Exit / Watch?
- Why?
- What position size is suggested?

Research Lab may remain complex.
Decision Mode should not become a developer/debug dashboard.
