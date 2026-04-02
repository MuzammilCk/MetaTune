# MetaTune: After Merging/Deleting Agentic Branches — What To Do Next

## Short answer
You **do not need a new environment** just because you merged and deleted branches.

Continue in the same repo/environment **if**:
- `git status` is clean,
- tests pass,
- required data/dependency files still exist.

Create a fresh environment only if the current one is unstable (broken Python env, corrupted dependencies, missing artifacts you cannot easily restore).

---

## Recommended workflow now (safe + fast)

1. Confirm baseline health:
   - `git status`
   - `PYTHONPATH=. pytest -q`

2. Create a new feature branch from your current main working branch:
   - `git checkout -b feat/agentic-phase-2-tool-registry`

3. Continue phase-by-phase (next is Phase 2):
   - ToolRegistry with typed action schemas,
   - Retry policies by error class,
   - Guardrails (max trials, time budget, rollback policy).

4. Keep changes small and mergeable:
   - one phase slice per PR,
   - tests in same PR,
   - avoid giant all-in-one branch.

5. Tag milestones after each merged phase:
   - `v0.2-phase1-runtime`
   - `v0.3-phase2-tools`
   - `v0.4-phase3-memory`

---

## Decision checklist: Continue here vs new environment

### Continue in this environment (default)
- You can run tests successfully.
- No unresolved merge conflicts.
- Dependencies import correctly.

### Recreate environment only if needed
- Repeated package conflicts.
- Python version mismatch you cannot resolve quickly.
- Local state/artifacts are inconsistent and blocking work.

---

## Immediate next implementation target
Focus now on **Phase 2**:
1. `ToolRegistry` abstraction.
2. `execute_with_retry()` with transient/data/logic/resource handling.
3. Guardrails manager (`max_actions`, `max_runtime_sec`, rollback trigger).

Then run:
- focused unit tests for registry/retries/guardrails,
- full regression suite.
