# Changelog — agent branch refactor

This documents a refactor of the `agent` branch aimed at making the
autonomous orchestrator (`agent.py`) actually work end-to-end, and
bringing the codebase it depends on up to a standard suitable for
production use. It's organized by commit; `git log` has the full detail
for each (rationale, verification steps, exact before/after behavior).

See `AGENTIC_ASSESSMENT.md` for the original audit that first
identified several of these issues at a higher level — this changelog
is the accurate, current record of what's since been fixed.

## Fixed

**Crash on the very first step of every run.** `data_analyzer.py`
mutated a correlation matrix in place via `np.fill_diagonal`, which
raises `ValueError: underlying array is read-only` under pandas'
Copy-on-Write behavior. This crashed `INSPECT_DATASET` — the agent's
first action — on essentially any real dataset. Fixed by computing the
off-diagonal statistics via masking instead of mutation.

**The core "self-correction" loop never actually retried anything.**
This is the headline fix. All the scaffolding for iterative
improvement existed — a task graph with DIAGNOSE_FAILURE/
REVISE_STRATEGY nodes, a `--max-trials` guardrail, per-action retry
budgets — but `_tool_revise_strategy` computed a revised learning rate
and then unconditionally ended the run, reporting `completed` without
the revision ever being tried. Verified live: with `--max-trials 3`
and an unreachable threshold, the agent ran exactly one trial. Fixed
so revise_strategy resets the RUN_TRIAL task node and hands control
back to the planner, which now genuinely re-selects RUN_TRIAL with the
updated params — bounded by the pre-existing `--max-trials` guardrail,
so this can't run away. A run that never clears the threshold now
correctly ends in a `failed` status with a clear reason, instead of a
false `completed`.

Two smaller bugs in the same control loop, found while fixing the
above:
- `run()` called `planner()` twice per iteration in the common case
  (once to peek at status, once for real), silently double-logging the
  run's own policy/decision trace.
- A genuine exception during RUN_TRIAL (as opposed to a below-threshold
  result) hard-aborted the whole run without ever attempting recovery,
  even though the retry-policy scaffolding suggested that was the
  intent. Unified into the same recovery path, while a deliberate
  guardrail stop is still distinguished and terminates cleanly.

**Silent wrong-target-column training.** When a caller doesn't specify
`--target`, `DatasetAnalyzer` auto-detects it via name heuristics
(`diagnosis`, `target`, `label`, ...), falling back to "last
non-`Unnamed:` column" only as a last resort. None of the three
training entry points (`agent.py`, `bilevel.py`, `pipeline.py`) ever
learned what the analyzer had resolved to, so they fell through to
`DynamicTrainer`'s own, much dumber fallback: the literal last column
of a fresh, uncleaned read of the CSV. Verified live against the
Wisconsin breast-cancer dataset: the analyzer correctly resolves
`diagnosis`, but the literal last column is `Unnamed: 32` — a 100%-null
artifact of a trailing comma in the header row. Without this fix,
auto-detect mode would silently train the model to predict an entirely
empty column. `bilevel.py` had it worse: `BilevelOptimizer` never
passed `target_col` to its inner trainer *at all*, so bilevel mode was
hitting this on every single dataset, every single trial, regardless
of whether `--target` was given.

Fixed by propagating the analyzer's resolved target column back to
each caller. Related: `DynamicTrainer` also always re-read and
re-cleaned the raw CSV independently of whatever `DatasetAnalyzer` had
already done — most importantly, it never dropped null-target rows the
way the analyzer's cleaning does, risking a spurious "missing" class
reaching `LabelEncoder`. `DynamicTrainer` now accepts an optional
pre-loaded (and ideally pre-cleaned) `df`, which `agent.py` populates
from the analyzer's `cleaned_data` and passes through every training
call; a defensive null-target-row drop was also added directly in
`prepare_data()` as a safety net for callers that don't pass one.

**A latent BatchNorm crash, found by the new regression tests.**
`ValueError: Expected more than 1 value per channel when training` —
PyTorch's `BatchNorm1d` can't handle a training batch of size exactly
1, which happens whenever `training_rows % batch_size == 1`. Not an
exotic case: it depends only on whatever batch size the meta-learner
predicts for a given dataset size. Fixed with `drop_last=True` on the
training loader, conditional on there still being at least one full
batch afterward so an oversized batch_size can't drop every sample.

## Changed

**Repo hygiene.** ~24MB of generated/debug output was tracked in git —
a 17MB AI-assistant reference dump, run reports, temp datasets, and a
`knowledge_base.csv` that mutated on every test run (`.gitignore`
existed but its patterns didn't match any of the actual filenames
produced). All untracked, and `.gitignore` rewritten to match reality.
Removed `cleanup.py`: it "solved" the same problem by deleting every
`.csv` in the repo root except a short whitelist that didn't include
the real sample datasets — running it would have silently deleted
`wisconsin.csv` and `global_car_dataset.csv`. A correct `.gitignore`
makes it unnecessary without that risk. Moved the root-level
`test_audit.py` into `tests/` so there's one canonical test location.

**Output files no longer land in the repo root.** Episodic memory, run
reports, the trained model/preprocessing pipeline, and the
meta-learner's knowledge base all now default to a run-scoped
`--output-dir` (default `.metatune_runs/`) instead of wherever the
process happened to be invoked from. This was also a real test-
isolation bug: every test that trained the meta-learner shared and
mutated one global `knowledge_base.csv`, so test behavior could
silently depend on what ran earlier in the same session — which is how
the BatchNorm bug above was actually surfaced.

**Logging.** `agent.py` and its dependency chain used bare `print()`
throughout — no levels, no filtering, no way to redirect. Added
`metatune_logging.py` and converted the core modules to it; the CLI
gained `--log-level`. One deliberate exception:
`DatasetAnalyzer.print_summary()` stayed on `print()` since it's an
explicit "render a report" display method, not operational narration.

**Error classification.** `_classify_failure` matched on message
keywords only, which misclassified the read-only-array crash above as
an `io_error` purely because "read" matched an IO keyword. Now
dispatches on exception type first, with the keyword match kept as a
fallback for generic exception types.

**Packaging.** Added `pyproject.toml` (setuptools, dependencies mirrored
from `requirements.txt`, a `metatune-agent` console entry point).
Verified with a real `pip install -e .`.

**Tests.** Added 5 new tests targeting exactly the bugs above — none of
which were covered before, which is how they went unnoticed despite an
already-substantial test file. Full suite: 46 passed (was 38 at the
branch tip this refactor started from).

## Explicitly not done in this pass

These are real, understood, and — in a couple of cases — genuinely
compelling next steps, but each is a separate, sizable piece of work
with its own risk profile, and rushing them alongside the fixes above
would have made everything harder to review and trust:

- **`recommend_algorithms()`'s output is never actually acted on.** The
  algorithm recommender computes a ranked list (Random Forest, XGBoost,
  logistic regression, gradient boosting, ...) with per-algorithm
  search spaces, but `RUN_TRIAL` unconditionally trains the PyTorch MLP
  regardless of what was recommended. Wiring the agent up to actually
  dispatch to `sklearn_engine.py`'s implementations for non-MLP
  recommendations would make algorithm selection a real capability
  instead of a display-only one — probably the single highest-value
  next step, and a genuine feature addition rather than a bug fix.
- **Three parallel trainer implementations**: `engine.py` (batch, used
  by the agent), `engine_stream.py` (callback-based, used by the
  Streamlit UI), `sklearn_engine.py` (sklearn algorithms, used for
  production export). The split between `engine.py`/`engine_stream.py`
  is intentional and documented; whether it's worth consolidating is a
  separate architectural question involving `app.py`, which this pass
  deliberately didn't touch.
- **Bilevel optimization still re-fits preprocessing from scratch on
  every trial** (up to 15 per run). This pass fixed it to stop
  re-reading the CSV from disk each time, but each trial still refits
  the same preprocessing pipeline identically, since hyperparameters
  don't affect preprocessing. Splitting `DynamicTrainer`'s data-prep
  from its model-fit step would let bilevel prep once and reuse it —
  a real performance win, but a more invasive change to a file several
  other callers depend on.
- **Phase 5 (multi-agent) from `AGENTIC_ASSESSMENT.md`** — specialized
  sub-agents for data forensics, search strategy, etc. — was already
  scoped there as optional/exploratory, and stays out of scope here.
- `pipeline.py` (the legacy single-shot CLI) got the same target-
  column/data-consistency fix as the agent since it was directly
  adjacent and cheap, but wasn't otherwise modernized (logging,
  guardrails) — it's superseded by `agent.py`, which is now the
  recommended entry point.
