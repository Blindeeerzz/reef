## Motivation

Generated tasks currently depend on their generation worker and play results for discovery and splitting.
This implements PR 1 of [#831](https://github.com/Human-Agent-Society/reef/issues/831), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): save validated tasks with their split before rollout.

## Changes

- Add a file-backed task store with generation checkpoints, atomic publication, a writer lock and a cumulative manifest. `TaskStore` and `FileTaskStore` live in `reef/record2task/store.py`, so any record-to-task stage that generates tasks can reuse them.
- Keep related source sessions in the same split across generations and reject conflicting train/eval bridges.
- Integrate the store with the existing worker and leave eval tasks unplayed, without Designer rewards or band measurements.
- Move shared task and arm readers out of the processor, expose store counts and retain scenario archive behavior.

### Task storage

`TaskStore` defines the storage interface, and `FileTaskStore` keeps the files under the record-to-task stage's configured `state_dir`:

```text
state_dir/<scenario>/store/
  store.json                version and split settings
  generations/gen-*.json    candidates and validation outcomes
  tasks/<task-name>/        Harbor task files
  split-manifest.json       cumulative train/eval split
  plays/round-*.json        current worker's round summaries
```

The worker saves candidates before validation.
Once a candidate passes validation, it publishes the task and assigns its split before any rollout:

```python
validation = await self.task_generator.validate(item.task)
if validation.is_valid:
    self.task_store.publish_task(generation=job.round_number, index=index)
```

Task files are written and synced before the store records publication.
Task metadata stays in generation documents, outside the Harbor directories.
Eval tasks remain saved without training or measurement plays.

Saving tasks before rollout lets us separate the two in later PRs.
Generation can finish while tasks wait to run.
Rollout can pick them up when capacity is available, retry failed runs without generating new tasks, and track which tasks were used for training.

## Documentation impact

The configuration reference, pipeline guide, recipe docstring and SPADE/AutoEnvScaling READMEs explain the new store, split settings and upgrade behavior.
The AutoEnvScaling task-pool example now reads the cumulative manifest from the canonical store.
Tests check example configuration, manifest readability and worker behavior; no live model or sandbox comparison was run.

## Compatibility and operational impact

`TaskPlayResult.split` defaults to `train` and rejects eval rewards.
Move generator-level `eval_fraction` and `seed` to `pipeline.record2task.config.eval_fraction` and `split_seed`.
The defaults are 0.0 and 0; existing stores require their original settings.

New stores do not import legacy rounds or tasks; existing records and training history remain intact.
Each store permits one writer, so composite components require distinct state directories.
Pydantic 2 is now a direct dependency for typed checkpoints, and the package check lists it as a core dependency.
Scheduling, delayed feedback, rollout buffers and training acknowledgement remain in subsequent PRs.

## Verification

- Focused regression suite: 289 tests passed, including the unchanged batch goldens.
- Store suite: passes with the additional interrupted-staging cleanup case.
- `pre-commit run --all-files`: passed.
- Targeted mypy across all nine changed source files: passed.
- Full pytest: collection blocked by missing Ray, Torch, Slime and `reef_client.record_import` dependencies (19 errors, 15 skips).
- Full mypy: 13 errors in unchanged `reef/observability/open_telemetry.py` with this environment's dependencies.

## AI assistance

Codex helped implement the store, integration, tests and documentation.
The contributor remains responsible for reviewing and understanding the changes.

## Checklist

- [x] The change is focused and contains no unrelated cleanup.
- [x] Changed code was reviewed against `AGENTS.md`.
- [x] Tests cover behavior changes and affected public contracts.
- [x] Affected documentation is updated.
- [x] Root README content is unchanged.
- [ ] Complete public RFC/split coordination before opening the PR.
- [ ] Run the full test suite and mypy in the supported dependency environment.
- [x] Relevant pre-commit and focused checks pass.
- [x] Non-trivial AI assistance is disclosed above.
