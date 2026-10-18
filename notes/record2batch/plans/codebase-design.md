# Generated-task pipeline: current code and proposed refactor

Local design for review against [RFC 805](https://github.com/Human-Agent-Society/reef/issues/805) and upstream main `b9d3ffb481e6f3a6be98da07f682d459d48e4727`, fetched on October 16, 2026.
This replaces the earlier six-task implementation breakdown with three milestones for the external-rollout training path.
Backend-native task execution, candidate evaluation, curriculum changes and episode compaction are deferred.
Names for new types and methods below are proposed APIs; selected operating decisions are recorded separately.

## Operating policy selected in discussion

Each data-loop run uses a fixed snapshot of recorded interactions; records arriving later wait for the next run.
Nightly execution is an example deployment schedule, not a pipeline requirement; runs may be invoked manually or through a configured trigger.
By default, a run finishes when its selected work has completed or been deferred; optional per-run time or work limits stop it earlier.
Preserve unfinished work when a limit is reached and retain the chosen limits with the run configuration for recovery.
Within a run, refill low rollout capacity using stored tasks first, then generation when more tasks are needed.
Measure active queue capacity in planned episodes, counting queued and running episodes together, with training and measurement plays both charged.
Persist validated tasks that do not fit in the active queue for later admission.
Episode queue capacity and optimizer batch size are separate: one update may collect task groups over multiple rollout waves under fixed sampling weights.
Production batch sizes and stability requirements will be informed by [the experiment plan](2026-10-16-record2task-experiments.md).
The numeric capacity, refill threshold and optimizer batch target have not been selected.
Rollout policy refresh and the training objective are research choices recorded in experiment 2, not selected architecture defaults.
Initial implementation uses the existing hosted inference endpoint as the rollout source, independently configured from the training backend.
No separate candidate sampler or per-update checkpoint reload is required; the hosted model is not assumed to be the latest training candidate.
Preserve behavior-policy identity and data needed by the selected learning method rather than requiring candidate inference.
Endpoint availability does not establish that the learning algorithm is on-policy; actual sampled weight versions determine that relationship.
Fixed older-policy replay requires explicit objective support and admission rules; it cannot be enabled by silently relaxing staleness checks.
Generation may run ahead using available feedback; persist late feedback against its original generation and task for subsequent use.
Task authoring remains configurable through the selected generator, preserving its existing prompt-based or agent-based approach.
Use a local file-backed task store by default, with an explicit interface and scenario recovery/archive integration.
Remote rollout receives staged files through the execution abstraction; a shared-storage backend remains a future deployment option.
For temporary generation-service or network failures, retry within a bounded budget; if the job still cannot complete, defer it durably and continue other eligible jobs.
Keep its source ownership, partial durable progress and attempt history recoverable; deferral must not mark the source successfully processed or turn an infrastructure failure into a scored solver failure.
Deferred jobs become eligible on the next run, regardless of its wall-clock schedule; they are not retried again later in the same run after their retry budget is exhausted.
Exact retry bounds and handling of service-wide outages remain to be selected.
Initially, a task failing validation is rejected with its reason recorded as feedback, and processing continues; the pipeline does not start a repair loop.
Later add bounded authoring repair, with its default budgets informed by experiment 3.
Infrastructure retries and authoring repairs are separate actions with separate counters; neither may reset the enclosing job or run budget.
Generation needs an enforceable job deadline in addition to attempt limits because even one authoring or validation attempt can take too long.
Exact deadline and cost/token limits remain configuration choices to establish through the budget study.
Source processing finishes after generation completion and its task definitions or rejection outcomes are durably persisted.
Pending task selection, rollout, feedback and training recover independently from the task store; source processing does not wait for their training acknowledgements.
Marking a source processed does not delete its original conversation or alter its retention policy.
Initially, each task has one training use; after that use is acknowledged, retain its definition for inspection but exclude it from automatic reselection.
Recovery of an interrupted attempt is separate from scheduling another training use, and committed training must never be repeated because an acknowledgement was lost.
Task reuse with fresh rollouts is a future policy to evaluate in experiment 4; it is distinct from replaying old trajectories in experiment 2.
For the initial upgrade, create a fresh task store for the refactored pipeline and preserve legacy records, task files and round summaries without automatically importing or resuming legacy rounds.
Keep the new store distinct from legacy state; its empty state must not reset existing committed training history or implicitly schedule historical work again.

## Current ownership

`GeneratedTaskRecipe.pipeline_spec()` already binds the three stages in `reef/recipe/generated_tasks/recipe.py`.
Most generated-task behavior lives together in `reef/recipe/generated_tasks/processor.py`.

| Component | Current behavior |
| --- | --- |
| `RoundRecord2Record` in `reef/record2record/rounds.py` | Holds source sessions and counts trained batches, but another stage calls its round lifecycle methods. |
| `GeneratedTaskRecord2Record` | Groups incoming training reports by task and closes groups using reported play counts. |
| `GeneratedTaskTasks` | Inherits `ReportTasks`; makes trajectories from play reports rather than returning the generated Harbor tasks. Its background worker generates, validates, writes and plays tasks, delivers feedback and writes round summaries. |
| `GeneratedTaskBatches` | Wraps already assembled trajectories in a `TrainingBatch`. |
| `TrainingPipeline` | Immediately settles `make_tasks()` results, reserves record sets and synchronously calls `make_batch()`. Its pending value is an already constructed batch waiting for acknowledgement. |
| `TaskPlayer` in `reef/harness/client/tasks.py` | Runs Harbor task episodes through reef-eval, captures inference receipts and posts scored reports. |
| `reef/sandbox` | Provides execution, transfer and cleanup through Harbor environment contracts; it does not select batches or assemble training samples. |
| `ScenarioStore` in `reef/storage/scenario.py` | Owns records and committed progress; there is no stored generated-task collection. |

The current flow is generation and execution inside record2task, followed by reports entering record2record again, trajectory assembly in record2task, and final wrapping in task2batch.
Generation already uses `BackgroundJobWorker`; introducing a worker is not the refactor.

## Target ownership and files

The three tracking issues are implementation children of #805, not new architecture RFCs.
#811 owns opt-in curriculum, feedback interpretation, difficulty changes and calibration policy; this refactor owns transport, deduplication and recovery of feedback needed by existing generators.
Agree shared identifiers and feedback fields with #811 rather than create parallel contracts.
#810 owns measured-usage events and reporters; episode queue counts, deadlines and task limits are execution controls and do not replace its usage/billing integrations.
The optional reporting work is not a prerequisite for this refactor.

Track delivery with three board-level issues, one per milestone, and PR-sized checklists within them.
The initial breakdown is task store/splits; pending contracts, buffer/backpressure, record scheduling and generation/feedback; episode runner, rollout assembly, admission/acknowledgement and recipe cutover.
These are nine proposed PR-sized work packages rather than nine separate tracking issues; refine their boundaries during review and include validation in each PR.
The buffer is an explicit work package, with coordination and final integration dependencies across the second and third milestones.

| Location | Proposed change | Milestone |
| --- | --- | --- |
| `reef/core/tasks/metadata.py` (new) | Shared names and validation for task identity, split, round, arms and batch/attempt labels. | 1 and 3 |
| `reef/core/tasks/` | Stored-task and execution-result values where multiple packages need them; reuse existing Harbor task and split types. | 1 and 3 |
| Task storage module, proposed under `reef/storage/` | Explicit task-store contract and local file implementation for task definitions, durable generation completion and execution state. | 1 |
| Stage construction and scenario lifecycle wiring | Share the task-store instance between consumers and register its directory for recovery/archive; no initial requirement to extend every `ScenarioStorage` backend. | 1 |
| `reef/record2record/sets.py`, `rounds.py` and `base.py` | Round input sets, pending settlement, selectable per-task units, complete round scheduling and completion-time sample admission. | 2 and 3 |
| `reef/record2task/base.py` | Add pending task results and identified completion delivery. | 2 |
| `reef/task2batch/base.py` | Add pending batch results, completion delivery and lifecycle/status support. | 3 |
| `reef/train/pipeline.py`, `trainer.py` | Drain completions, preserve ownership while work runs, route batch records, distinguish running rollout from a trainable batch and keep polling without new ingress. | 2 and 3 |
| `reef/recipe/generated_tasks/processor.py` | Reduce record2task to generation, validation and persistence; move repeated plays, report assembly and grouping to task2batch. | 2 and 3 |
| `reef/harness/` and existing `TaskPlayer` | Reuse or extract episode execution behind a shared runner; integrate task staging and sandbox environment selection. | 3 |
| `reef/core/tasks/generation.py` and affected recipe generators | Define feedback delivery after generation has returned stored tasks. | 2 and 3 |
| Recipe settings, examples and pipeline documentation | Move settings to their owners and document task storage, pending work and execution capabilities. | All |

Shared lifecycle mechanisms belong in the stage and pipeline packages.
Generated-task method policy stays in the recipe; the sandbox package must not import the training pipeline.
Splitting the recipe's large processor into record, generation and batch modules is reasonable if it clarifies these owners, but file splitting is not itself a delivery milestone.

## Milestone 1: persist tasks and split before execution

Add a scenario-owned task collection containing identity, digest, source references, generation identity, split, measurement-arm definitions and task-file references.
Keep execution attempts separate from task definitions, so retrying an episode cannot change the task or its split.
Insertion must be idempotent and recoverable with source progress.
Coordinate the split implementation with #799 and PRs #800/#801 rather than introducing a third competing algorithm.

For this milestone, the existing combined worker can still execute tasks, but it must store them first and only execute the train split.
Held-out tasks remain stored and unplayed.
This provides a useful intermediate change before the stage ownership switches.

Selected approach: use a local file-backed task store under scenario state and inject its narrow interface into stage construction.
`PipelineContext` currently exposes no task storage, so injection is an explicit wiring change.
Define atomic writes, interrupted-write recovery and archive/restart behavior; register owned directories through the existing scenario lifecycle mechanism such as `scenario_state_dirs()`.
Local files do not imply unmanaged recipe-private state: task2batch must consume the store interface rather than read another stage's private layout.
A generator-local path is not automatically usable in a remote sandbox, so stage task files through the execution abstraction.
This follows the RFC's file-first direction and does not require a shared database or changes to every installed record-storage backend.

Acceptance covers duplicate insertion, restart with stable splits, related-source assignments, conflicting split connections, missing task files and interrupted generation persistence.

## Milestone 2: record2record scheduling and generation-only record2task

`RoundRecord2Record` emits a round `RecordSet` containing its stable round identity, selected source sessions and eligible earlier feedback.
It owns the round limit, pacing, source lifetime and empty-round behavior.
It no longer relies on `GeneratedTaskTasks.poll()` calling `round_due()`, `start_round()`, `round_finished()` and `stop_rounds()`.

Keep the actual `make_tasks(RecordSet) -> TaskResult` interface and add `PENDING`.
The RFC describes its return as a tuple, but the current implementation already uses `TaskResult`.
The generation stage submits work and returns immediately.
The existing worker generates, validates and durably stores tasks, then returns train-split `TaskItem`s through a proposed `take_finished()` completion API keyed by the originating set.
The pipeline settles that completion exactly once.
Synchronous task stages retain their immediate result path.

One round can produce many tasks, while `RecordSetPool.reserve()` currently counts sets.
Recommendation: settling a round creates one selectable child unit per stored training task, with stable task identity and a parent-round link.
The pipeline records each child's `TaskItem`, and record2record owns child selection and retirement.
Thus `tasks_per_step` selects task units rather than whole generation rounds.
Selected approach: finish source processing after a durable generation-completion record links all persisted tasks and rejection outcomes.
Do not advance source progress before that checkpoint; a failed or deferred job retains its unfinished source work.
Child task units recover independently through the store after that checkpoint, without waiting for all siblings to train.
Persist the source links and any required feedback context before releasing active source ownership; this release is separate from deleting retained conversation records.

There are two distinct completions: generation is persisted, and its rollout feedback is complete.
They must not share one round-finished boolean.
AutoEnvScaling currently resets `self.pending` on each `generate()` call and its `record_results()` finalizes the last generation, so delayed results cannot simply be sent to it unchanged.
Selected approach: retain feedback state by generation/task, persist delivery progress and deduplicate retries so generation can run ahead with available feedback.
Held-out tasks being intentionally unplayed must not be reported as solver failures.
Any feedback contract change is an explicit amendment to the RFC's claim that `TaskGenerator` stays unchanged.

Acceptance covers slow generation without blocking ingress, completion without new records, correct task-level batching, source release only after durable completion, independent task recovery, empty rounds, feedback after a later generation, duplicate completion and scenario close.
The ownership switch in milestones 2 and 3 must be coordinated so a production recipe never starts returning tasks to the old trajectory-only consumer.

## Milestone 3: task2batch rollout, sandbox execution and batch assembly

`GeneratedTaskBatches` takes selected task units and launches the configured training and measurement plays on a worker through the shared episode runner.
It returns pending immediately and retains one rollout job per selected batch initially.
Repeated plays, actual reported group size and trajectory assembly move here from the old record/report path.
The runner handles one episode; task2batch decides how many episodes to run and how to group them.

Reuse `TaskPlayer`, `TaskLab`/reef-eval and `ExecutorEnvironment` rather than inventing another isolation abstraction.
Select execution environment independently from the inference endpoint and policy context.
Resolve and stage task files before execution, validate supported capabilities, preserve hidden-verifier boundaries and close owned resources on cancellation.
An executor implementing Harbor's environment interface does not necessarily build arbitrary Dockerfile tasks.
Document actual supported task/environment combinations and reject incompatible combinations before model calls.

The pipeline reserves task units while rollout is pending.
Add an explicit batch-result/completion contract; a proposed `BatchResult` distinguishes ready, pending and terminal no-batch outcomes.
The existing `TrainingPipeline.pending` meaning must be separated into running rollout state and a ready batch retained until acknowledgement.
The trainer must never receive a pending marker as a `TrainingBatch`, or start training simply because enough input tasks were selected.

Recommendation: forward batch-tagged inference and report records from ingress to their owning pending batch, as the RFC suggests.
These records cannot also form ordinary training groups.
Persist enough batch/attempt identity to reconstruct ownership after restart; late or duplicate reports cannot train again.
Measurement results may feed the generator but never enter training, and held-out evaluation outcomes remain outside generation feedback.

Once rollout finishes, apply a completion-time admission check owned by record2record to the actual producing policy versions.
The existing staleness helper skips `TaskItem`, and `drop_stale_sets()` skips checking while a reservation exists, so calling the existing pre-selection check is insufficient.
An explicit reserved-result check must retire or admit completed task groups without silently invalidating the pool reservation.
Do not duplicate staleness policy inside task2batch or the runner.
Task definitions have no sampled-policy age; their generated trajectories do.
If sampling uses a separate synced inference server, version identity must represent an acknowledged scenario policy load rather than an unrelated server-local label.

On completion, assemble a batch from admitted training episodes with actual reported group sizes.
Preserve the current requirement of at least two episodes for a relative-reward group.
Keep infrastructure failure, scored solver outcome and training rejection distinct.
Feedback completion and training acknowledgement are separate checkpoints: a stale trajectory can still describe a real solver outcome.
After the existing training/commit acknowledgement, retire the selected task uses and owned rollout records through an idempotent storage transition.
Recovery must reconcile committed training before retrying an unknown batch outcome.

After the switch, remove `GeneratedTaskTasks.played()`, the combined generate-and-play loop, `reported_play_counts`, `TaskGroupGrouping`, cross-stage round driving and the old round-summary dependency.
Task-store state and execution/feedback state replace their recovery responsibilities.

## Configuration and lifecycle

| Owner | Proposed settings |
| --- | --- |
| record2record | `rounds`, `batches_per_round`, source selection and scheduling limits. |
| record2task | Generator selection, description, validation and generation worker/buffer limits. |
| task2batch | `tasks_per_step`, `plays_per_task`, rollout worker, agent, environment and sampling configuration; trajectory assembly options. |
| Scenario-managed local task store | Task metadata, files or file references, generation completion, execution state and retention. |

Replace the old task-stage polling extension with identified completion delivery if adopting the RFC contract.
Internal pipeline polling/wakeup remains necessary; do not remove pending-work visibility before the trainer has an equivalent mechanism.
Include task2batch in close, status and pending-work reporting, which currently cover only the earlier stages.
Start with bounded generation and one pending rollout batch, without overlap between rollout and training.
Selected upgrade policy follows the RFC's fresh-task-store behavior, with legacy files and records preserved and no automatic legacy-round migration.

## Validation and boundaries

Extend existing task-split, generated-task, task-player, sandbox, pipeline and storage tests.
Use controlled workers to demonstrate source selection, generation, persistence, task-level selection, rollout, admission, batch assembly and one-time acknowledgement.
Include restart during generation, rollout, feedback delivery and training acknowledgement, plus a policy change while rollout is pending.
Preserve unaffected recipe batch goldens and manual/hybrid behavior; record intentional generated-task changes separately.
The RFC also requires a live SPADE or AutoEnvScaling comparison before merging the rollout change; that needs separately approved billing and budget.
No model calls, implementation edits or public GitHub writes are part of this design review.

The remaining implementation contracts include task-store state transitions, pending/completion APIs, completion-time admission and policy-version accounting for the configured learning objective.
Unselected recommendations above remain proposals for review.
