# Stock Signal Lab — Codex Working Rules

## Role

You are the engineering agent.

Implement bounded changes, run tests, generate required evidence, and commit clean work when allowed.

Do not make research-direction decisions. Report evidence; do not sell conclusions.

## Token efficiency

Use targeted search. Do not do a full repo tour.

Read only files needed for the task.

Do not paste large code blocks, CSV contents, or unrelated summaries.

## Default forbidden changes

Unless explicitly instructed, do not change:

- `ml_score()`
- production target or labels
- Decision Cockpit scoring
- suggested action
- suggested position size
- ranking or ordering
- allocation
- persistence
- cache behaviour
- dependencies

Generated research outputs must not be committed unless explicitly requested.

Do not push unless explicitly instructed.

## Task lanes

Identify the task lane before working:

- Lane A: Display / UX only
- Lane B: Research-only ML
- Lane C: Production decision output

Do not mix lanes in one commit unless explicitly instructed.

Lane C must be narrow and separately committed.

## Lane B workflow

For Research-only ML work:

1. Read `data/research_runs/latest/codex_handoff.md`.
2. Read `data/research_runs/latest/diagnostics_manifest.json`.
3. Read only relevant CSV files.
4. State the current evidence briefly.
5. Make bounded code changes.
6. Run focused tests.
7. Run full pytest when ready.
8. Run headless Research Lab export if research logic changed.
9. Compare baseline vs candidate evidence.
10. Stop if evidence is mixed or worse.

Maximum bounded implementation iterations: 2.

Do not keep trying blindly.

## Required evidence for Lane B

Report the relevant raw numbers. Use only fields that exist or can be produced within scope:

- run path
- manifest row counts
- sample count
- ticker count
- fold count
- AUC / PR-AUC / Brier
- bucket spread
- calibration gap
- regime inversion count
- worst ticker / regime / fold, if available
- stop-rule result
- one-line failure diagnosis

If a field is unavailable, say so.

No raw evidence, no research conclusion.

## Stop rules

Stop and report if:

- focused tests fail and cannot be fixed within scope
- full pytest fails
- headless Research Lab runner fails
- baseline or candidate export cannot be produced
- evidence is mixed or worse after 2 iterations
- task requires production changes without approval
- task requires a new dependency
- task requires broad refactor

## Commit policy

Commit after successful task unless instructed otherwise.

Before commit:

```bash
git diff
git status --short
````

After commit:

```bash
git status -sb
git log --oneline --decorate -3
```

Working tree should be clean after commit, except ignored local data files.

## Final report format

Use this format:

```text
Commit:
Files:
Tests:
Runtime / evidence:
Key evidence:
Key change:
Not done:
Git state:
```

Keep it short.

