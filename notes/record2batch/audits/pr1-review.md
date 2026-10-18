# Record2batch PR 1 implementation review

PR 1 implements task storage and splitting before play for [#831](https://github.com/Human-Agent-Society/reef/issues/831).
The existing worker uses the store now; scheduling and rollout remain on the existing path until their planned PRs.

Worktree: `/mnt/data3/reef-r2d-eval/reef-record2batch`.
Branch: `feat/record2batch-task-store`.
Base: upstream main `b9d3ffb481e6f3a6be98da07f682d459d48e4727`.
Changes remain local and uncommitted.

## Implementation

- `reef/recipe/generated_tasks/store.py` defines the `TaskStore` ABC and `FileTaskStore` implementation.
  Candidates, validation outcomes, source references, measurement arms and generation completion persist independently of play results.
  Task files precede publication metadata; JSON writes use temporary files, replacement and filesystem syncs.
  Opening checks the version, split configuration and published task files, clears interrupted staging and rebuilds the manifest.
  A process lock prevents two writers; closing prevents a late worker from publishing after ownership passes to another writer.
- `reef/core/tasks/split.py` extends existing source grouping with fixed incremental placements.
  New tasks inherit a related component's split.
  A task connecting existing train and eval components is rejected without moving either component.
  Independent components use a seeded hash draw rather than forcing an eval task in every small generation.
- `GeneratedTaskTasks.run_round` saves candidates before validation, publishes valid tasks before play and plays train tasks only.
  Eval outcomes carry `split="eval"`, without an error or rewards.
  SPADE excludes them from Designer rewards and training experience; AutoEnvScaling records them as held out without measuring the band.
  Both generators stop writing manifests.
- Shared task-name and arm readers move to `reef/core/tasks/metadata.py`.
  Status exposes store counts, and the existing scenario archive path includes the store.

## Changes from the planned file layout

Harbor rejects extra files inside a task directory, so `reef.json` would make stored tasks unreadable.
Task metadata therefore lives in generation documents outside the Harbor directories.
Those documents own fixed split placements; `store.json` owns the version, fraction and seed.
A separate `split.json` would duplicate that state and is omitted.
The implementation plan now reflects this layout.

`PipelineContext` has no component identifier.
Composite components must configure distinct state directories; PR 1 does not expand that shared contract.

## Configuration and upgrade

Set `eval_fraction` and `split_seed` under `pipeline.record2task.config`, replacing generator-level `eval_fraction` and `seed`.
Defaults are 0.0 and 0, preserving training plays for every valid task unless a holdout is configured.
An existing store rejects a different fraction or seed.

The store lives under `state_dir/<scenario>/store/`, with raw Harbor files in `tasks/`, generation checkpoints in `generations/` and transitional round summaries in `plays/`.
The cumulative manifest is `store/split-manifest.json`, using `store/tasks/` as its task root.
Legacy round summaries and task files are preserved without importing them into the new store.
Existing inference records and committed training history are untouched.
Pydantic 2 is declared directly for checkpoint serialization; it was previously supplied through Harbor.

## Verification

The focused regression run passed 289 tests across task splitting, storage, worker integration, both generators, composite replay, trainer behavior and the unchanged batch goldens.
The final staging-cleanup check also passes in the store suite.
Repository-wide pre-commit passes, and targeted mypy passes all nine changed Python source files.

The full pytest run stopped during collection with 19 errors because this environment lacks Ray, Torch, Slime and `reef_client.record_import`; 15 tests skipped.
Full mypy reports 13 errors in the unchanged `reef/observability/open_telemetry.py` module with the installed dependencies.
These are limits on validation, not passing full-suite results.
Tests used stand-ins and local services, with no paid model calls.

Logs are beside this note: `record2batch-pr1-focused-tests.log`, `record2batch-pr1-full-tests.log`, `record2batch-pr1-pre-commit.log`, `record2batch-pr1-targeted-mypy.log` and `record2batch-pr1-mypy.log`.

## Remaining sequence

PR 1 recovers stored definitions and validation checkpoints; it does not promise exactly-once rollout or feedback across process restarts.
PRs 2 and 3 add pending source ownership, scheduling and durable delayed feedback.
PRs 4 through 7 add idempotent sandbox staging, rollout buffers, training acknowledgement and the final recipe switch.
The current combined worker still uses round summaries to advance rounds and can repeat plays when explicitly retried.

The user approved updating issues #831, #832 and #833, and their revised descriptions were saved and verified on GitHub.
The receipt is `issue-drafts/2026-10-16-record2batch-v2-issue-update-receipt.json`.
RFC and split coordination comments remain separate public actions before the affected PRs are opened.
