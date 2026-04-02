# MetaTune Agentic AI Build Log

> **Purpose:** Living implementation log for the full agentic-AI build.
> Keep this file updated at the end of every phase/PR so future work can resume quickly.

---

## 1) Project Context

- **Primary strategy document:** `AGENTIC_ASSESSMENT.md`
- **Operational recovery doc:** `BRANCH_RECOVERY_AND_NEXT_STEPS.md`
- **Runtime entrypoints:** `agent.py`, `pipeline.py`, `app.py`
- **Core intelligence/data modules:** `data_analyzer.py`, `brain.py`, `bilevel.py`, `engine.py`, `engine_stream.py`, `sklearn_engine.py`

---

## 2) Current Build Status (at a glance)

| Phase | Goal | Status | Notes |
|---|---|---|---|
| Phase 1 | Agent runtime hardening | ✅ Done | Added confidence calibration + action-level time/cost accounting |
| Phase 2 | Tooling architecture | ✅ Done | Registry dispatch + retry policies + guardrails are live |
| Phase 3 | Memory architecture | ⬜ Not started | Working/episodic/semantic split pending |
| Phase 4 | Explainability & governance | ⬜ Not started | Decision traces and safety checks pending |
| Phase 5 | Multi-agent evolution | ⬜ Not started | Specialist agents + coordinator pending |

Legend: ✅ done · 🟡 in progress · ⬜ not started

---

## 3) Implemented So Far

### 3.1 Data readiness hardening
- Added target detection logic with priority order:
  1) user-provided target,
  2) exact name heuristic,
  3) partial name heuristic,
  4) fallback to last column.
- Added null cleaning routine:
  - drops rows with null target,
  - numeric feature imputation (median),
  - categorical feature imputation (mode, fallback `unknown`).
- Added metadata emitted into dataset DNA:
  - `target_detection_method`
  - `cleaning_report`

### 3.2 Agent runtime scaffolding
- Added task-graph primitives:
  - `TaskNode`
  - `TaskStatus`
  - dependency/deadline/retry fields
- Introduced policy-based planner with:
  - budget awareness,
  - uncertainty awareness,
  - rationale trace logging.
- Executor now returns structured output payload:
  - `success`, `confidence`, `cost`, `artifacts`, `error_class`, `rationale`.
- Added failure taxonomy with `FailureType`.
- Added approval policy modes:
  - `manual`
  - `semi-auto`
  - `full-auto`
- Added non-interactive approval fallback behavior.
- Added `action_budget` CLI setting.

### 3.3 Critic improvements
- Upgraded from threshold-only to multi-objective breakdown:
  - quality
  - time
  - stability
  - budget
  - drift risk
- Persists `critic_breakdown` in runtime state.

### 3.5 Confidence and cost calibration
- Added confidence calibration in executor outputs using:
  - data quality signals (`missing_ratio`, `sparsity`),
  - runtime duration,
  - prior failure history,
  - error-class penalties.
- Added persistent action-level accounting:
  - `action_costs[]` entries with per-action duration + synthetic cost units,
  - aggregate `cost_summary` (`total_actions`, `total_runtime_sec`, `total_cost_units`).

### 3.6 Tooling architecture bootstrap (Phase 2 started)
- Added `ToolSpec` and action-to-handler `ToolRegistry` in the agent runtime.
- Routed executor dispatch through registry handlers (instead of direct inline branching).
- Added retry behavior by failure class using per-tool retry policies.
- Added initial guardrails:
  - `max_runtime_sec` wall-clock limit,
  - `max_trials` limit for trial actions,
  - rollback recommendation signal when metric regresses between trials.

### 3.4 Pipeline resilience
- Pipeline now aborts safely when:
  - dataset load fails,
  - analyzer returns no DNA.
- Avoids downstream `NoneType` crashes from missing analysis output.

---

## 4) Test Coverage Added for Agentic Build

- `tests/test_agent_orchestration.py`
  - memory reset behavior
  - approval mode behavior
  - structured action logging
  - planner contract
  - critic breakdown presence

- `tests/test_data_analyzer.py`
  - auto-detect target not-last-column
  - null-cleaning verification

- `tests/test_pipeline.py`
  - graceful handling when analyzer returns missing DNA

---

## 5) Phase-by-Phase Plan (Execution Checklist)

## Phase 1 — Agent runtime hardening
- [x] TaskGraph object with dependencies/retries/deadlines
- [x] Policy-based planner (state + uncertainty + budget -> action + rationale)
- [x] Structured executor return payload
- [x] Multi-objective critic score
- [x] Add richer confidence calibration (model uncertainty + data quality signal)
- [x] Add action-level time/cost accounting in persistent state

## Phase 2 — Tooling architecture
- [x] Implement `ToolRegistry` with typed schemas
- [x] Route planner actions through registry (no direct hardcoded execution path)
- [x] Add retry strategy by error class (`transient`, `data`, `logic`, `resource`)
- [ ] Add guardrails:
  - [x] max trials
  - [x] max runtime
  - [x] rollback on regression

## Phase 3 — Memory architecture
- [ ] Split memory into:
  - [ ] working memory
  - [ ] episodic memory
  - [ ] semantic memory
- [ ] Add retrieval scoring for prior episodes
- [ ] Add postmortem + root-cause tags

## Phase 4 — Explainability & governance
- [ ] Machine-readable decision traces
- [ ] Preflight checks:
  - [ ] leakage
  - [ ] imbalance risk
  - [ ] metric/task mismatch
- [ ] Uncertainty/abstention policy

## Phase 5 — Multi-agent evolution
- [ ] Data Forensics Agent
- [ ] Search Strategy Agent
- [ ] Training Execution Agent
- [ ] Audit & Safety Agent
- [ ] Coordinator Agent arbitration logic

---

## 6) Known Gaps / Risks

- Planner is policy-based but still lightweight (not yet full utility optimization).
- Tooling layer is still not registry-driven end-to-end.
- Memory is still mostly single-store JSON state (semantic retrieval pending).
- Governance/safety checks are not yet integrated into default run gate.

---

## 7) Definition of Done for “Agentic AI v1”

A version can be tagged `agentic-v1` when all of the following are true:
1. Planner uses task graph + registry tools + retries + budgets.
2. Executor outputs are fully structured and persisted with error taxonomy.
3. Critic uses multi-objective scoring and feeds replanning.
4. Memory has working+episodic+semantic split with retrieval.
5. Safety preflight checks run by default before expensive actions.
6. Full regression and phase-specific tests pass in CI.

---

## 8) Update Protocol (IMPORTANT)

After each merged PR touching agentic build:
1. Update **Section 2 status table**.
2. Update **Section 3 implemented items**.
3. Check/uncheck **Section 5 checklist**.
4. Add new risks in **Section 6**.
5. If milestone reached, update **Section 7** and create git tag.

This file is the canonical “where are we now?” source for implementation continuity.
