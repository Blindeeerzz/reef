# Record2batch implementation tasks and completion criteria

Status: v2.1, revised after [the audit](2026-10-16-record2batch-audit.md) and the follow-up review of pool interfaces, split conflicts and training recovery.
The user authorized these corrections and starting local implementation on October 16, 2026; public actions and the maintainer acknowledgements listed below remain separate.
v1 is kept at [archive/2026-10-16-record2batch-task-decomposition-v1.md](archive/2026-10-16-record2batch-task-decomposition-v1.md).

Implement the external-rollout training path proposed in [RFC 805](https://github.com/Human-Agent-Society/reef/issues/805) through three tracking issues and seven functional PRs.
This plan fixes the design decisions, state transitions, dependencies and completion criteria so the executor implements without having to decide architecture.
Where this plan differs from RFC #805 it says so (**RFC deviation**); each deviation must be posted on #805 and acknowledged before the PR that depends on it is opened.

The reviewer is Zijian (@BobbyZhouZijian).
Their [comment on #805](https://github.com/Human-Agent-Society/reef/issues/805#issuecomment-6209372153) approved the three-issue split only.
They have not answered four questions: source completion after saving, delayed feedback, post-rollout freshness and the `TaskGenerator` amendment.
Treat these as unagreed until they reply.

## Scope and references

| Tracking issue | Delivery |
| --- | --- |
| [831 Task storage and train/eval split](https://github.com/Human-Agent-Society/reef/issues/831) | PR 1: task store, split, shared metadata module. |
| [832 Scheduling generation and buffering](https://github.com/Human-Agent-Society/reef/issues/832) | PR 2: pending contract and round sets. PR 3: generation without play, delayed feedback, runs. |
| [833 Rollout and training batches](https://github.com/Human-Agent-Society/reef/issues/833) | PR 4: shared episode runner. PR 5: task-level rollout batches and capacity. PR 6: acknowledgement and recovery. PR 7: switch and removal. |

**Changed from v1:**

- The issue #832 checklist item "Buffer and backpressure" moves to #833 (PR 5), because capacity has no meaning until rollout is scheduled.
- The public issue descriptions were updated and verified on October 16, 2026, after the user explicitly approved the edits.

Code references are relative to the Reef repository and were checked against upstream main `b9d3ffb481e6f3a6be98da07f682d459d48e4727` on October 16, 2026; a read-only checkout is at `/mnt/data3/reef-r2d-eval/reef-audit`.
Before each PR:

- re-fetch upstream;
- rebase onto open PR [829](https://github.com/Human-Agent-Society/reef/pull/829) (`SampleAssemblyError`, `unassemblable_sets` in `reef/train/pipeline.py`) if it has merged, keeping its behaviour;
- recheck [800](https://github.com/Human-Agent-Society/reef/pull/800), [801](https://github.com/Human-Agent-Society/reef/pull/801) and RFC [787](https://github.com/Human-Agent-Society/reef/issues/787) (one worker concept).

The [codebase design](2026-10-16-record2task-codebase-design.md) holds background architecture; where it conflicts with this plan (task units reserved in `RecordSetPool`, records routed to a pending batch), this plan wins.
The [experiment plan](2026-10-16-record2task-experiments.md) defines the controlled studies.
The older [six-task plan](2026-10-16-record2task-pipeline-implementation.md) is historical.

Out of scope: backend-native rollout, candidate evaluation, automatic task repair, automatic task reuse, curriculum changes (#811), episode segmentation (#825, stop-gap #829).

## Current behavior (verified against b9d3ffb)

| Area | Current code |
| --- | --- |
| Rounds | `GeneratedTaskTasks.poll()` (`reef/recipe/generated_tasks/processor.py:410`) drives `RoundRecord2Record.round_due/start_round/round_finished/stop_rounds` (`reef/record2record/rounds.py`). One round = one `BackgroundJobWorker` job: generate, validate, `write_task`, play, `record_results`, write `round-NNNNN.json`. |
| Execution | `played()` calls `HttpGenerator.write_task` and `HttpGenerator.play` (`reef/record2task/generator/client.py:227,251`). The generator service runs `TaskPlayer` (`reef/harness/client/tasks.py:224`). The agent's model calls go through `CaptureProxy` into Reef's scenario proxy with tags `{labels…, task, episode}`, so the inference records and the scored report reach Reef as ordinary records. |
| Training data | Reports are grouped by **task name** (`TaskGroupGrouping.placement`, processor.py:197). A group closes at `plays_per_task` or at the reported count passed through `reported_play_counts`. Groups wait unreserved in `RecordSetPool`, and `ready()` is `len(pool) >= batch_size`. |
| Duplicates | The generator service refuses a task whose content hash it already holds (409, no path returned; `generator/service.py:631-643`). `played()` turns `DuplicateTask` into "refused". |
| Split | No split in the recipe. AutoEnvScaling and SPADE write the split manifest themselves in `record_results`, **after** the plays: AutoEnvScaling keeps only tasks inside its solver band (`recipes/beta/autoenvscaling/generator.py:703-714`), and SPADE keeps only measured tasks (`recipes/beta/spade/generator.py:288-295`). The manifest is read by the Cordis gate (`reef/recipe/cordis.py:385`) and the task-player command (`reef/harness/client/tasks.py:436`). |
| Durability | The commit log is the only durable consumption truth. `prepare_commit` writes `consumed | released` record IDs atomically with the algorithm state, and a stale drop writes a consumption receipt (`receipt_id` = batch ID, `metadata.outcome = "stale"`; `reef/train/trainer.py:568-700`). Restart replays retained, unconsumed records (`trainer.reingest`), and the union of commits and receipts is the exclusion set (`reef/scenario/factory.py:69`). Source records are released only inside a commit or receipt. |
| Freshness | `Record2Record.drop_stale_sets` checks held sets before reservation and returns early while a reservation exists. `TaskItem` contributes no version (#819). |
| Wake-up | `ready()` calls `record2task.poll()`. The trainer polls at a fixed interval while `pipeline.background_jobs_running()` is true (`trainer.py:294`, `dispatcher.py:1132`); that covers record2record and record2task only. |
| Archive | `Recipe.scenario_state_dirs` asks only the record2task class (`reef/recipe/base.py:137`); composite recipes concatenate per component. |

## Selected design

### Task pool and group pool (**RFC deviation D3**, option C′ from the 2026-10-16 discussion)

The record stage holds **two pools of the same generalized class**:

1. **Task pool.** One `TaskSet` per saved train task, holding its `TaskItem`. A settled round set fills it (one set per task, so a round's tasks are selected independently). Selection uses an ordinary `SelectionPolicy`, which is where #811's curriculum selection would later plug in. Starting a rollout reserves task sets from it and consumes that reservation at once. The task's durable fate then lives in the store as a task use.
2. **Group pool.** The existing pool. Rollout reports come back as ordinary records, and the record stage groups them by `(batch, task_use)`, as it groups reports by task name today. Ready groups wait **unreserved** until their batch's rollout is complete; then `build_batch` reserves exactly that batch's groups.

Effects:

- The existing `drop_stale_sets` checks real trajectory versions before the groups are reserved, so #819's contract is unchanged.
- Dropping part of a batch is natural: a stale group ends, and the batch reserves the survivors.
- The existing commit consumes the rollout records atomically, and restart rebuilds groups by replaying the reports.
- `Task2Batch.make_batch` stays synchronous, and `TrainingPipeline.pending` keeps its single meaning (a ready batch awaiting acknowledgement). The running rollout is separate pipeline state (`self.rollout`).
- RFC rules kept: tasks reach task2batch through a pool, so a later backend-native task2batch reserves from the task pool and passes the `TaskItem`s through (not implemented here, but the interface must allow it). A report with `labels.batch` trains only in that batch.
- RFC open question 1 ("how do rollout records reach task2batch") is answered by record-stage grouping instead of routing records to the pending batch.
- **Fallback if Zijian rejects D3:** the RFC's routing, plus an explicit record2record admission method for reserved results. Then PR 5 grows and PR 6 changes the #819 contract openly.

### Pool generalization

`RecordSetPool` and the selection policies read only `set_id`, `arrival_index` and `group_key`. Make that explicit instead of forcing task sets to pretend to be record sets:

- Define a small frozen dataclass ABC `PoolItem` with those three fields in a shared record-stage module that does not import the pool or selection policies. `RecordSet` and `TaskSet` explicitly inherit it. Do not use `typing.Protocol`, which repository policy forbids.
- Make the pool generic: `Pool[ItemT: PoolItem]`. Keep `RecordSetPool = Pool[RecordSet]` as the name the existing code uses, preserving existing construction and reservation behavior.
- Type `SelectionPolicy` (`reef/record2record/selection.py`) over `PoolItem`. Existing policies change only in annotations.
- Add `TaskSet` (frozen dataclass: `set_id` = task name, `arrival_index`, `group_key` = generation, `item: TaskItem`). It is **not** a `RecordSet`: it has no records, so it never appears in consumed or releasable record IDs.
- `Record2Record` gains `task_pool: Pool[TaskSet] | None = None`. Every existing stage leaves it `None` and behaves exactly as today.
- Add `reserve(max_count, *, where=None)` to the pool. `where` is an explicit selection-filter ABC with a batch-filter implementation, not a callable-valued policy. Filter candidates before applying the existing selection policy, so a batch reservation includes only that batch's groups.

### Generic stage contracts for rollout

The rollout loop must work for any recipe whose record2task returns `TaskItem`s, not only generated tasks. So the constraint on stages is a small set of data classes plus two explicit ABCs, and nothing generated-task-specific:

| Contract | Where | What it requires |
| --- | --- | --- |
| Data classes | `reef/core/batches.py` (`TaskItem`), `reef/record2record/pool.py` (`PoolItem`, `TaskSet`), `reef/core/tasks/metadata.py` (fields and rules) | Tasks arrive as `TaskItem`; rollout episodes carry `labels.batch` and `labels.task_use`. |
| `RolloutRecord2Record` ABC | new `reef/record2record/rollout.py` | `task_pool`; `plays_reported(batch, task_use, count)`; `batch_groups_settled(batch) -> bool`; grouping of rollout reports by `(batch, task_use)`; enforcement of the two metadata rules. |
| `RolloutTask2Batch` ABC | `reef/task2batch/base.py` | `submit(batch_id, items)`, `take_finished()`, `running()`, `close()`. |

- Implement the record-stage half **once**, as `RolloutReportRecord2Record` on top of `ReportRecord2Record`. It moves today's `TaskGroupGrouping` and `plays_reported` logic out of `processor.py`, generalized to the `(batch, task_use)` key and to counting distinct episodes.
- `RoundRecord2Record` extends it with rounds, runs and the store restore. Those parts stay generated-task-specific.
- Any record2task works: the loop only needs `TaskItem`s in its result. `GeneratedTaskTasks` is one implementation.
- The pipeline enters rollout mode only when its record stage is a `RolloutRecord2Record` **and** its task2batch is a `RolloutTask2Batch`; a mismatched pair fails at build time with a message naming both classes.
- Not generalized yet: the task store stays in `reef/recipe/generated_tasks/store.py` until a second task source needs it. A recipe whose tasks do not need persistence can fill the task pool without a store, but then loses task recovery.

### Rollout loop

The pipeline gains a second loop, used only in rollout mode (both ABCs above). Otherwise the existing linear cycle runs unchanged.

```
records ─► record2record ─► round set ─► record2task (PENDING → take_finished) ─► TaskItems
                                                                          │
                                       task pool ◄── one TaskSet per train task (and store restore)
                                           │ reserve tasks_per_step, consume at once, store: uses planned
                                           ▼
                               task2batch.submit(batch_id, task items) ─► runner ─► episodes via Reef proxy
                                                                          │
records (inference + reports with labels.batch / labels.task_use) ◄───────┘
   ─► record2record groups by (batch, task_use) ─► group pool (unreserved)
   task2batch.take_finished ─► per-use reported counts ─► groups close
   rollout complete + all its groups settled ─► drop_stale_sets ─► reserve(where=batch) ─► make_batch ─► pending ─► trainer
```

`TrainingPipeline.ready()` in rollout mode:

1. Absorb completions: `record2task.take_finished()` (settles round sets and fills the task pool) and `task2batch.take_finished()` (per-use counts go to record2record; the rollout becomes terminal).
2. If `pending` (a ready batch), return True.
3. If a rollout is running: when it is terminal and record2record says every group of its batch is settled, run `drop_stale_sets`.
   - If any of the batch's groups survive, return True; `build_batch` then reserves them.
   - If none survive, clear the rollout, mark its uses `untrainable` (PR 6 rule) and continue.
   - Otherwise return False.
4. If no rollout is running and the task pool holds at least one set, reserve up to `tasks_per_step` task sets.
   - Consume that reservation, have the store create the uses as `planned`, call `task2batch.submit(batch_id, items)` and record `self.rollout`.
   - Return False.

`build_batch()` in rollout mode reserves from the group pool with `where=` the rollout's batch, calls `make_batch`, moves the uses to `training(batch, report_ids)`, clears `self.rollout`, and sets `pending`.
`background_jobs_running()` is true while round sets are pending or a rollout is running.
Status adds `task_pool` (count), `rollout` (batch, uses, admitted and running episodes, capacity) and the store counts by state and split.

### Ownership

| Concern | Owner |
| --- | --- |
| Run start, source snapshot, round sets, generation pacing, refill rule | `RoundRecord2Record` (record2record) |
| Task pool and its selection policy; grouping rollout reports; group pool; freshness | record2record (`RolloutRecord2Record`, implemented once as `RolloutReportRecord2Record`) |
| Generate, validate, save; feedback delivery | `GeneratedTaskTasks` (record2task) worker |
| Rollout submission, execution waves, episode capacity, per-use reported counts | rollout task2batch (`GeneratedTaskBatches` implements `RolloutTask2Batch`) |
| The rollout loop: reserve tasks, submit, decide readiness, reserve groups | `TrainingPipeline` |
| Reservation and consumption of groups | existing `RecordSetPool` and commit path |
| Task definitions, generation jobs, task uses, feedback journal | task store (`reef/recipe/generated_tasks/store.py`, explicit ABC `TaskStore`, `FileTaskStore`) |
| "Did a use train?" | commit log and consumption receipts; the store flag is a cache reconciled from them |

`Record2Record` today "keeps only memory that a replay of the retained records rebuilds".
`RoundRecord2Record` becomes a documented exception: at construction it restores its task pool (the store's `available` train tasks), generation jobs and run state from the task store.
State this in the class docstring and in the pipeline developer guide, together with a diagram of the two loops.

### Shared metadata module (RFC, PR 1)

Create `reef/core/tasks/metadata.py` with the field names and readers/writers used by every stage.

| Field | Where | Meaning |
| --- | --- | --- |
| `task.name`, `task.digest` | report `metadata.task` (exists) | task identity; digest is `HarborTask.digest` |
| `episode.labels.arm`, `episode.labels.round` | report `metadata.episode.labels` (exist) | arm label; round number |
| `task.round`, `task.split`, `task.arms` | `TaskItem.metadata` | generation round, `train`/`eval`, measurement arms |
| `episode.labels.batch` | play reports and inference tags | the batch whose rollout produced the episode |
| `episode.labels.task_use` | play reports and inference tags | the task-use ID; the group key with `batch` (addition to the RFC; mention on #805) |

The two rules, enforced in record2record:

- A report carrying `labels.batch` trains only in that batch.
- A report whose arm is not the training arm never trains.

Move `reported_task_name`, `reported_arm` and `TRAINING_ARM` from `processor.py` into this module.

### Identities

| Object | Identity | Durable fields | Owner |
| --- | --- | --- | --- |
| Run | `run-NNNNN` | record-sequence watermark at start (the snapshot), config, limits (`max_rounds`, `max_seconds`), state `running/finished/limited` | record2record |
| Round set / generation job | `gen-NNNNN` (= RFC round number) | run, source record IDs (session references, **not** Designer receipts), attempts, deadline, state, candidate tasks, saved task IDs, rejections | record2record schedules; record2task executes |
| Saved task | task name (unique in the store) + digest | Harbor files, generation, original source record IDs, Designer receipt IDs (separate field), split, measurement arms, state | task store |
| Task use | `use-<task>-<n>` | task, batch, state, report record IDs once known, disposition | task store via task2batch and the pipeline |
| Rollout batch | the `TrainingBatch.batch_id` it will produce, `"{scenario}:generated-tasks:{n}"` | task uses, planned episodes per arm, state | task2batch |
| Feedback item | `(generation, task, result_id)` | result, delivery state | task store; delivered by record2task |

### Task store layout and file rules

```
state_dir/<scenario>/store/
  store.json                 version, eval_fraction and seed; refuse incompatible settings
  .lock                      fcntl.flock by the single writer; a second writer fails at construction
  .staging/                  temporary files and dirs; cleared on open
  tasks/<name>/              Harbor files only (write_harbor_task layout)
  generations/gen-NNNNN.json  candidates, metadata, outcomes, fixed splits and completion
  plays/round-NNNNN.json      transitional worker summaries
  uses/use-*.json
  feedback/gen-NNNNN.jsonl   append-only journal: one line per result, then one line per delivery
  runs/run-NNNNN.json
  split-manifest.json        cumulative manifest in reef.core.tasks.split format, root = tasks/
```

- **Atomic JSON:** every JSON file is written to `.staging/`, `fsync`ed, then `os.replace`d into place, with the containing directory synced for crash durability. Reuse the existing split-manifest replacement pattern, but add the durability steps it does not currently perform.
- **Atomic task directory:** `write_harbor_task` publishes only Harbor files using its existing temporary-directory rename. The store syncs those files before publishing the generation outcome that makes the task visible. Interrupted publication retries from the saved candidate. The strict Harbor reader rejects extra files, so generation documents own task metadata outside the task directories. Opening verifies every published task against its saved digest and rebuilds the manifest.
- **Same content, same name:** saving returns the existing task (idempotent).
- **Different content, same name:** `TaskConflict` with both digests. The generation worker records it as a rejection with reason `name conflict`; it does not crash the job.
- **One writer:** a scenario has one trainer process. Composite components must configure distinct `state_dir` values because `PipelineContext` does not expose a component identifier. The writer lock rejects accidental sharing. PR 1 does not expand the context contract.
- **Archive:** `GeneratedTaskTasks.scenario_state_dirs` returns the store's parent directory, so deleting a scenario archives it.
- **Upgrade:** a new store starts empty. Legacy `round-*.json`, task files and records stay untouched and are not imported. Committed history is not reset.

### State machines

```
generation job:  planned -> generating -> candidates_saved -> (validating) -> complete
                                     \-> deferred (infra retries or deadline exhausted; eligible next run)
saved task:      candidate -> available(train) | held_out(eval) | rejected(reason)
                 available -> in_use -> trained | available (after stale/no-update/cancel, attempts < max) | retired_untrained
task use:        planned -> running -> rolled_out -> training(batch, report_ids) -> trained | stale_rejected | cancelled | untrainable
```

- The output of `generate()` is persisted as candidates before any validation. A retry resumes validating the remaining candidates and never calls `generate()` again for that job.
- A generation job is **complete** when every candidate is `available`, `held_out` or `rejected`. Only then are its sources done; the job record is the authority for "do not regenerate from these sources".
- `max_task_attempts` (default 2) bounds how often a task returns to `available` after an untrained use.

### Recovery order (the six crash cases plus a seventh)

On open, after record replay has rebuilt pipeline memory:

1. **Generation partly saved (cases 2 and 3).**
   - A job in `generating` with no candidates: re-run from its sources.
   - A job in `candidates_saved`: resume validating.
   - Sources replayed from the record log are skipped when a complete or deferred job already names them.
2. **Sources replayed although done (case 7, new).** Generation can run ahead while nothing commits, so sources are never released in a commit and replay on restart. The job record, not record release, prevents regeneration. Record release in the next commit is retention cleanup only.
3. **Some tasks of a generation used (case 1).** Siblings stay `available` in the store and are restored into the task pool. Uses are separate records, so nothing about a sibling changes when one task trains.
4. **Rollout interrupted or duplicated (case 4).**
   - Uses in `planned` or `running` with no live job become `available` again (no training happened), their attempt counter increments, and their tasks return to the task pool.
   - Reports already received for that batch are released, not trained: their batch is not pending, so record2record ends them.
   - A duplicate or late worker completion whose job ID is not live is ignored.
5. **Committed, local acknowledgement lost (case 5).** For each use in `training(batch, report_ids)`:
   - A verified, non-pending training commit must identify the exact scenario component, batch, task use and reports that actually contributed training samples. Persist this association under a reserved `metrics.record2batch` object containing `batch_id` and per-use `task_use_id`/`training_report_ids`, populated by the trainer from the reserved pipeline batch before acknowledgement mutates it. Preserve the existing commit schema and exclude measurement, discarded and released-only reports from this metadata.
   - Mark a use `trained` only when this explicit association matches the stored use and its training report IDs are included in that commit's `consumed_ids`. Membership in `consumed_ids` alone is not proof of training: the trainer also adds released and skipped records there.
   - Mark a use `stale_rejected` only from a matching batch/component consumption receipt with `outcome = "stale"` and explicit per-use disposition metadata, not from a report ID appearing in an unrelated receipt's release cleanup.
   - If no matching commit or receipt exists and the reports remain unconsumed, replay rebuilds the same batch without a new rollout.
   - If reports are already consumed but the exact disposition cannot be proved, mark the use unresolved and fail recovery with an actionable diagnostic; do not infer success or launch another rollout.

   Read commits and receipts separately: `RecoveredTrainerState` merges them, so add a store-recovery helper that reads `committed_records` and `consumption_receipts` itself.
   In-memory acknowledgement can be provisional while a commit is prepared; on restart only the durable evidence above determines trained status, even if the store was updated before the commit completed.
6. **Feedback after a later generation, and across restart (case 6).**
   - Feedback items are journaled before delivery, and the delivery line is written after the generator returns.
   - A crash between the two re-delivers the item, so the generator deduplicates by `result_id`, persisted in its own `state_dir`.

## Decisions

| # | Decision | Status |
| --- | --- | --- |
| D1 | Follow the RFC: `TaskStatus.PENDING` plus `take_finished()` on record2task; a stage that overrides `poll` fails at construction. `TrainingPipeline.background_jobs_running()` stays as the trainer's wake-up signal, but is **derived** by the pipeline from pending sets, the running rollout and record2record's own jobs. It is no longer a stage method on record2task. | Matches the RFC; mention the derived signal on #805. |
| D2 | Split rule: #801's connected-component rule with persistent placements, plus #800's separation of Designer receipts from session sources. A task sharing sources with existing tasks inherits their split only when those placements agree. A task bridging existing train and eval components is rejected with reason `split conflict`, retains its candidate files for diagnosis, and enters neither split. Do not mutate existing placements or move previously used tasks across splits. Split assignment precedes any play. | Depends on #800/#801. Coordinate on #801 before opening PR 1; local implementation is authorized. If neither merges first, PR 1 extends `reef/core/tasks/split.py` and `task_groups`, rather than adding a third parallel split path. |
| D3 | Task pool and group pool with a generalized pool class, plus the rollout loop (option C′, above). Chosen over the RFC as written (which needs a new post-rollout staleness check and partial reservations), over v2's separate task queue (which loses selection policies and backend-native pass-through), and over one mixed pool (two kinds of set in one pool). | Decided with the user 2026-10-16. **RFC deviation**: post on #805 before PR 3. |
| D4 | Rollout model traffic always goes through the Reef scenario ingress (`CaptureProxy` → scenario proxy), so records with tokens, logprobs and `runtime_load_id` are captured. "Hosted endpoint" means the upstream that serves the scenario. A sampler whose versions are not in the scenario's served list is rejected at configuration time while no objective declares off-policy support. | Confirm on #805, because #818 lists "production weights or a separate synced inference server" as undecided. |
| D5 | Eval tasks are never played; the store writes the cumulative `split-manifest.json`, and the generators stop writing manifests. The external benchmark is the real evaluation: SWE-chat sessions are the source records, and SWE-Together is the benchmark. So the internal split is only a leakage guard: keep the mechanism (RFC #831), and let experiment configs set `eval_fraction` (0 is allowed). AutoEnvScaling's band filter on eval membership is dropped and the change documented, with no `heldout` arm. This matches RFC open question 2: "until candidate evaluation lands, nothing plays eval tasks". Contamination between SWE-chat sources and SWE-Together (shared repos or issues) is checked in the experiment plan, not in this code. | Decided by the user (2026-10-16). |
| D6 | `TaskGenerator` amendment: `record_results` may receive results of any earlier generation. `TaskPlayResult` gains `generation`, `result_id`, `split` (from #800) and an `infrastructure_failure` flag. Generators key state by generation and task and deduplicate by `result_id`. | **RFC deviation** (the RFC says `TaskGenerator` does not change); post on #805 and coordinate the fields with @Benjamin-eecs on #811. |
| D7 | A run starts when the previous run has finished (or at the first start) and records exist after its watermark. Runs can be spaced with `min_run_interval_seconds` (default 0). Manual or nightly triggers are out of scope. | Planner default. |
| D8 | Refill rule: start a round set only while `available` train tasks < `refill_tasks` (default `2 × tasks_per_step`), generation jobs in flight < `max_generation_jobs` (default 1), and the run has eligible sources and remaining limits. | Planner default; experiments tune it. |

### Remaining public actions and maintainer coordination

- Remaining public actions: post D1/D3/D4/D6 on #805 and coordinate D2 on #801. Issue descriptions #831–#833 were updated with user approval on 2026-10-16. Other public writes still require explicit approval.
- Whether to wait for Zijian's answer on D3 before opening PR 3, or to develop it locally in parallel. Recommended: develop locally, open the PR only after the answer.

These do not block the authorized local start on PR 1. They remain prerequisites for opening the affected public PRs; local implementation is not maintainer approval of the RFC deviations.

## PR sequence and dependencies

| PR | Issue | Depends on | Production caller after merge |
| --- | --- | --- | --- |
| 1 | #831 | D2 | The existing worker saves tasks to the store before play, plays train tasks only, and the store writes the manifest. |
| 2 | #832 | PR 1, D1 | The existing combined round runs as a pending round set, scheduled by record2record; `poll` driving is removed. |
| 3 | #832 | PR 2, D3 posted, D6 | Pool generalization and task pool. Generation saves without playing, and its tasks fill the task pool; a transitional play job plays tasks reserved from the task pool. Delayed feedback; runs and deferral. |
| 4 | #833 | PR 1 (can run alongside PRs 2–3) | The transitional play job executes through the runner with idempotent staging. |
| 5 | #833 | PRs 3, 4, D4 | The rollout loop: `RolloutTask2Batch`, `(batch, task_use)` grouping in the group pool, and capacity, behind `pipeline.task2batch.config.rollout: true` (default false). |
| 6 | #833 | PR 5 | Acknowledgement, dispositions and recovery from commit evidence; the opt-in path is safe to run. |
| 7 | #833 | PR 6, approved live comparison | The default recipe uses the new path; old orchestration and the transitional play job are removed. |

Every PR runs in production behaviour except PR 5, whose opt-in flag is turned on by PR 6's tests and made default by PR 7.
Work on an isolated worktree per PR, branched from upstream main (or the previous PR branch).
Commit locally when appropriate, without automatically adding an agent co-author trailer, following the user's explicit instruction.
Nothing is pushed without the user's OK.

## PR 1 Task store, split and shared metadata

**Issue:** #831.

**Code areas:** new `reef/core/tasks/metadata.py`; `reef/core/tasks/split.py` (persistent incremental placement); new `reef/recipe/generated_tasks/store.py` (`TaskStore` ABC, `FileTaskStore`); `reef/recipe/generated_tasks/processor.py` (`run_round`, `played`, `scenario_state_dirs`); `recipes/beta/autoenvscaling/generator.py` and `recipes/beta/spade/generator.py` (stop writing manifests; honour D5); `reef/core/tasks/generation.py` (`TaskPlayResult.split` only if #800 has not added it).

### Changes

- The metadata module and the field moves described above.
- `TaskStore` operations, only those PR 1 calls:
  - `open` (lock, version check, staging cleanup)
  - `begin_generation`
  - `save_candidates`
  - `publish_task` (validated → `available` or `held_out` by split)
  - `reject_task`
  - `complete_generation`
  - `tasks(split=…)`
  - `counts()` (by state and split, for status)
  - `write_manifest`
- Incremental split (D2): `assign_split(task, sources) -> split` over persisted placements; a conflicting train/eval bridge raises a specific split-conflict error that the worker records as a rejection. Check for conflicts before changing placements or publishing the task. The seed and eval fraction are stored in `store.json`, and opening with different values is refused. Persisted generation outcomes own the fixed placements; no second placement file is needed.
- `run_round`:
  - persist the generated candidates;
  - validate;
  - publish each task with its split;
  - play only `available` train tasks;
  - send eval tasks to the generator as results with `split="eval"` and no error;
  - write the manifest after the round.
- The store, not the generator, writes the manifest. The generators drop their `write_manifest` calls and their generator-level split settings. Configure `pipeline.record2task.config.eval_fraction` (default 0.0) and `split_seed` (default 0). AutoEnvScaling keeps its band measurements for train tasks in its generation report; eval tasks get no plays (D5).
- Status: `task_store: {counts by state and split}`.
- `scenario_state_dirs` includes the store. Document the fresh-store upgrade in the recipe docstring and `docs/` configuration page.

### Completion criteria

- Saving the same task twice returns the same entry; the same name with a different digest raises `TaskConflict` and becomes a recorded rejection.
- Kill-point tests (raise at each write boundary, reopen) never show an `available` task with missing files, and never a `complete` generation missing a saved task.
- A retried generation does not call `generate()` again and saves no duplicate.
- Tasks that share or bridge same-split sources obey D2; a task bridging train and eval is rejected, appears in neither manifest split and leaves existing placements unchanged. Check transitive and repeated bridges, including after reopen. No eval task appears in a training play (assert on the fake player's calls).
- After reopening, the store lists the same tasks and the manifest reads back with `manifest_task_paths(..., split="eval")`.
- Scenario delete archives the store and nothing else.
- Upgrading next to legacy `round-*.json` starts an empty store and leaves the legacy files and records untouched.
- The #766 batch goldens are unchanged.

**Validation:**

- `tests/core/test_task_split.py` (incremental placement, bridging);
- new `tests/reef_recipe/test_task_store.py` (publication, kill points, lock, version);
- `tests/reef_recipe/test_generated_tasks.py` (worker saves, then plays train only);
- the generator suites under `tests/recipes/{spade,autoenvscaling}/` (no manifest writes; eval tasks never played).

## PR 2 Pending contract and round sets

**Issue:** #832.

**Code areas:** `reef/record2record/base.py` (`TaskStatus.PENDING`; set held while pending), `reef/record2task/base.py` (`take_finished`, `poll` removed, construction check), `reef/train/pipeline.py` (settle completions, derived `background_jobs_running`, include task2batch in `close`/`status`/metrics), `reef/record2record/rounds.py` (round sets), `reef/recipe/generated_tasks/processor.py`.

### Changes

- `TaskStatus.PENDING`: the record stage holds the set (its records are not releasable) until settled.
- `Record2Task.take_finished() -> tuple[tuple[Hashable, TaskResult], ...]`: completions keyed by set ID.
  - The pipeline settles each completion once.
  - A completion for an unknown or ended set is logged and dropped.
- Remove `Record2Task.poll` and `Record2Task.background_jobs_running`. A subclass that defines either fails at construction with a message pointing to `take_finished`.
- The pipeline derives `background_jobs_running()` from: pending sets exist, record2record jobs, and (from PR 5) a running rollout.
- `RoundRecord2Record` forms a **round set**: round number, source sessions, and the feedback references to deliver (RFC). It owns pacing (`batches_per_round`, empty-round follow-up, round limit), so `round_due/start_round/round_finished/stop_rounds` are no longer called from record2task.
  - A round set that settles `READY` with no tasks or `SKIPPED` makes the next round due at once.
  - Settling releases the round's source sessions.
- `GeneratedTaskTasks.make_tasks(round set)` submits the existing combined round job and returns `PENDING`; `take_finished` returns its result. The old play path is unchanged in this PR: the result is `READY` with no tasks, because training data still comes back as reports.
- Closing a pipeline closes all three stages. After `close`, completions are dropped.

### Completion criteria

- A blocked fake generation worker does not block `ingest`, `ready` or `status`.
- A completion makes the pipeline progress with no new record (trainer test with background polling).
- A duplicate completion, or one for an ended set, does not settle twice or touch another set.
- An empty round makes the next round due at once without a busy loop.
- After `close`, a late completion changes nothing.
- A stage overriding `poll` fails at construction.
- Synchronous recipes, manual/hybrid modes and the #766 goldens are unchanged.
- Contract tests for the pending path of record2task.
- Pipeline developer guide (`docs/developer-guide/processors.rst`) and API reference (`docs/reference/python-api.rst`) updated.

**Validation:** `tests/train/test_trainer.py`, pipeline tests that build `TrainingPipeline` (e.g. `tests/reef_service/test_manual_training.py`, `tests/recipes/sdpo/test_sdpo_pipeline.py`), round tests, generated-task tests with controlled workers; no live model calls.

**Implemented locally (2026-10-17, branch `feat/record2batch-pending-rounds`, commit `05588ef4` on PR 1 `3e3a4206`).** Decisions made during implementation:

- The pipeline, not `Record2Record`, holds pending sets (`TrainingPipeline.pending_sets`) and subtracts their records in `releasable_record_ids`. Report stages override `releasable_record_ids` without calling the base, so one pipeline-level rule is the only general place.
- Round follow-up uses the settled status: anything but `READY` makes the next round due at once. The combined round returns `READY` with no tasks when it played a task, `SKIPPED` when it played none, and `READY` when its job failed (keeps the old "do not spin on a failing worker" behaviour). In PR 3, a generation with no train task must return `SKIPPED`.
- The round range comes from `RoundRecord2Record.resume_rounds(next_round=..., round_limit=...)`, called once by `GeneratedTaskTasks` at construction. The `rounds` key does not move until PR 7.
- A worker that refuses a round raises from `ready()`. After `close()`, `ready()` returns False.
- `Task2Batch.close()` exists and the pipeline calls it. Task2batch `status`/metrics wait for PR 5, which has a consumer.
- No pipeline-level `pending_sets` status key: exact-status tests pin the shape, and the round in flight already shows as `task_generation.in_flight`. Add a key in PR 5 together with the rollout status.
- `plays_reported` counts are still drained inside `GeneratedTaskTasks.take_finished` (old path; removed in PR 7).

## PR 3 Generation without play, delayed feedback, runs

**Issue:** #832.

**Code areas:** `reef/record2record/pool.py` and `selection.py` (generalization), `reef/record2record/sets.py` or a new `tasksets.py` (`TaskSet`), `reef/record2record/base.py` (`task_pool`), `processor.py` (generation job, transitional play job), `reef/record2record/rounds.py` (task pool fill and restore, runs, refill rule, deferral), store (`generations`, `feedback`, `runs`), `reef/core/tasks/generation.py` (D6 fields), `recipes/beta/autoenvscaling/generator.py`, `recipes/beta/spade/generator.py`.

### Changes

- **Pool generalization** as described under the selected design: `PoolItem`, generic `Pool`, `RecordSetPool` alias, `SelectionPolicy` over `PoolItem`, `TaskSet`, `Record2Record.task_pool`, and `reserve(..., where=)`. It lands here because this PR has the first consumer.
- **The generation job** does generate → persist candidates → validate → publish/reject. It returns `READY` with the train `TaskItem`s (metadata per the module), never plays, and completes the generation in the store before returning.
- **Settling a round set** `READY` ends the round set (its sources are released; the pipeline keeps no tasks under its ID) and puts one `TaskSet` per train task into the task pool. At construction, `RoundRecord2Record` restores the task pool from the store.
- **Retries:**
  - `generation_attempts` (default 3) and `generation_deadline_seconds` (default 1800) apply to infrastructure failures (exceptions from `generate`/`validate`, `GeneratorError`).
  - When exhausted, the job is `deferred`, its sources stay owned, and other jobs continue.
  - Invalid tasks are rejected with their reason; there is no repair.
- **Runs (D7):**
  - The snapshot is the record-sequence watermark when the run starts; sources after it wait.
  - `max_rounds_per_run` and `max_run_seconds` are optional.
  - Deferred jobs are not retried in the same run and become eligible at the next run.
- **The refill rule (D8)** decides when to start a round set.
- **The transitional play job** lives on the same worker as generation but is a separate job type.
  - While no earlier play job runs, the pipeline reserves up to `tasks_per_step` task sets from the task pool (arrival order), consumes the reservation, and submits them; the job plays them with the existing `played()`.
  - This puts the task pool in production use before PR 5. The old report grouping by task name still builds the batches.
  - It journals results as feedback items.
  - This keeps the default recipe training between PR 3 and PR 7. PR 7 deletes it.
- **Delayed feedback (D6):**
  - Results are journaled by `(generation, task, result_id)`.
  - The next round set carries the undelivered items, and record2task delivers them through `record_results` before calling `generate()`.
  - Delivery is journaled after the call returns.
  - AutoEnvScaling and SPADE replace `self.pending` (reset on each `generate()`) with state keyed by generation and task, persisted under their `state_dir`, and deduplicate by `result_id`.
  - A generation's feedback is **complete** when all its train tasks have a result (trained, failed, or retired untrained); eval tasks are excluded. AutoEnvScaling's `STAGE_UNREACHED` and SPADE's per-generation experience update fire on completion, not on the next `generate()`.
  - Infrastructure failures carry `infrastructure_failure=True` and are not counted as solver failures; eval outcomes are never sent.

### Completion criteria

- The generation path calls no player (fake player asserts zero calls) and returns references to saved train tasks.
- A transient failure retries within the attempt and deadline limits; exhaustion defers durably while another job progresses; the deferred job runs at the next run, not earlier.
- Records after a run's watermark are excluded from that run.
- One generation's tasks can be played by different play jobs; siblings survive a restart and are back in the task pool.
- Existing stages with `task_pool = None`, every existing `SelectionPolicy` and the #766 goldens are unchanged by the pool generalization (typing-only changes; Ruff and type checks pass).
- A restart after generation completion (no commit yet) does not regenerate from the replayed sources.
- Feedback for generation A applies after generation B starts and after restart; re-delivery after a crash has no duplicate effect.
- Eval tasks are not reported as failures; infrastructure failures are flagged.
- The existing SPADE and AutoEnvScaling suites pass, apart from the documented lifecycle changes.

**Validation:**

- `tests/recipes/{spade,autoenvscaling}/` with delayed and duplicated results;
- generated-task worker tests with service failures and deadlines;
- run/watermark tests;
- restart tests with replayed sources.

## PR 4 Shared episode runner and idempotent staging

**Issue:** #833.

**Code areas:** `reef/harness/client/tasks.py` (`TaskPlayer`), `reef/record2task/generator/client.py` and `service.py` (staging, `/plays`), `reef/sandbox/environment.py`, `reef/sandbox/process.py`, `reef/sandbox/config.py`, `processor.py` (transitional play job uses the runner).

### Changes

- **`EpisodeRunner`:** an ABC, with an HTTP implementation over the generator service's `/plays`.
  - Input: a stored task reference, scenario, batch, task use, episode ID, arm, plays, extra instruction files, limits and model.
  - It returns per-episode outcomes: reward, report record IDs, error, and `infrastructure_failure` (the episode could not run) as distinct from a scored failure.
  - Batch and task use go into the play labels, so inference tags and reports carry `labels.batch` and `labels.task_use`.
- **The store is canonical; the generator's tasks root is a staging cache.**
  - Before playing, the runner stages the task from the store.
  - The service's `POST /tasks` returns `200` with the existing path when a task with the same digest and name is present.
  - It keeps 409 for the same name with different content, and for the same content under a different name only if that is still wanted.
  - `DuplicateTask` no longer means "refused" in the play path.
  - `delete_task` is only called for tasks the runner staged.
- **Model traffic (D4):** the runner never bypasses `CaptureProxy`; only the model name or served upstream are configurable.
- Copy #803's constraints into the runner's docstring and checks:
  - ordinary cleanup leaves a borrowed E2B VM running and cancellation retires it;
  - synchronous bridges run outside an event loop;
  - another execution user, forced template rebuild, Daytona and Modal are unsupported.
  - Unsupported task/environment combinations fail before any model call.
- Repeated plays, waves and grouping stay outside the runner.

### Completion criteria

- A saved task runs through the runner (fake lab) and yields reward, report IDs and the batch/task-use labels in inference tags and the report.
- Re-staging after a simulated crash between staging and play succeeds; different content under the same name is refused.
- A redeployed generator (empty tasks root) can still play a stored task.
- Infrastructure failure and scored failure are distinguishable.
- Cancelling cleans owned resources and leaves borrowed ones.
- Existing task-player and sandbox tests pass.

**Validation:**

- `tests/harness/test_task_player.py`, `tests/sandbox/test_environment.py`, executor tests;
- generator service tests for staging;
- one smoke run with the fake lab and no paid inference.

## PR 5 The rollout loop: task-level rollout batches and capacity

**Issue:** #833.

**Code areas:** `reef/task2batch/base.py` (`RolloutTask2Batch` ABC: `submit`, `take_finished`, `running`, `close`), `processor.py` (`GeneratedTaskBatches` implements it with a rollout worker over the PR 4 runner; report grouping keyed by `(batch, task_use)`), new `reef/record2record/rollout.py` (`RolloutRecord2Record` ABC and `RolloutReportRecord2Record`: the `(batch, task_use)` grouping moved from `TaskGroupGrouping`, `plays_reported`, `batch_groups_settled`), `reef/record2record/rounds.py` (`RoundRecord2Record` now extends it), `reef/train/pipeline.py` (the rollout loop and `self.rollout`), store (`uses`). The PR 3 transitional play job keeps running when `rollout: false`.

### Changes

- **The rollout loop in the pipeline** (selected design, "Rollout loop"), when `rollout: true`:
  1. Reserve up to `tasks_per_step` task sets from the task pool, consume the reservation, create the uses as `planned` in the store, and `submit` the rollout batch with its batch ID.
  2. Episodes are admitted in waves: at most `episode_capacity` (default `tasks_per_step × plays_per_task`) queued plus running episodes, counting training and measurement plays.
     - Capacity, `execution_concurrency` and `tasks_per_step` are separate settings.
     - A task whose plays exceed the capacity is played across several waves.
  3. `task2batch.take_finished()` yields per-use outcomes. The pipeline hands each use's reported count to record2record (today's `plays_reported` mechanism, keyed by `(batch, task_use)`), which closes its group.
  4. The batch is ready when the rollout is terminal and every group of the batch is settled (`READY` or `DISCARD`).
     - Then `drop_stale_sets` runs on the group pool (nothing is reserved there yet), and `build_batch` reserves exactly the batch's surviving groups with `reserve(..., where=batch)`.
     - `make_batch` wraps them.
     - Use states move to `training(batch, report_ids)` **before** the batch is handed to the trainer.
     - If no group survives, the rollout is cleared and the loop continues with the next tasks.
- **Grouping:**
  - The group key is `(batch, task_use)`, not the task name.
  - The "two or more usable episodes" rule counts distinct `episode.id` values, not reports (#825).
  - Reports with a `labels.batch` that is not the running or ready batch are ended without training.
  - Measurement-arm reports never train; their rewards are journaled as feedback.
- **Measurement plays** run for train tasks as configured by the generator's arms.
- Status: `rollout: {batch, uses, admitted_episodes, running_episodes, capacity}`, plus the store counts.
- The flag defaults to false. Unit tests and PR 6 exercise it.

### Completion criteria

- Stored tasks go from the task pool, through a running rollout, to a ready batch from the group pool; each intended training episode is included at most once.
- `pending` never holds anything but a ready batch; the trainer never sees a running rollout as a batch.
- With `rollout: false`, or with a record stage without a task pool, the pipeline runs the existing cycle unchanged.
- A contract test runs the rollout loop with a minimal non-generated-task record2task (fixed `TaskItem`s), `RolloutReportRecord2Record` and a fake runner, showing the loop does not depend on generated-task code. A mismatched stage pair fails at build time.
- A partial report count closes a group; a group with fewer than two distinct episodes is discarded.
- Duplicate or late reports neither enlarge a group nor create another batch; they never enter another batch's groups.
- Measurement episodes contribute zero training samples.
- Admission never exceeds `episode_capacity`; a batch larger than the capacity completes across waves without deadlock.
- A model-version change while the rollout is pending makes `drop_stale_sets` drop the affected groups before reservation (#819 behaviour, unchanged code).
- Contract test for the rollout task2batch with a fake runner.
- The #766 goldens are unchanged.

**Validation:** trainer and generated-task integration tests with a fake runner, partial and duplicate reports, capacity smaller than the batch, and a policy change mid-rollout.

## PR 6 Acknowledgement, dispositions and recovery

**Issue:** #833.

**Code areas:** store (use reconciliation), a new store-recovery helper reading `committed_records` and `consumption_receipts` (`reef/scenario/factory.py`, `reef/storage/commits.py`), `reef/train/trainer.py` (`prepare_commit`, `reject_pending`), `reef/train/pipeline.py` (reserved training metadata, `acknowledge`, `release_batch`), `processor.py`.

### Changes

- On `acknowledge(trained=True)`, uses become `trained` and their tasks `trained` (one training use per task).
- On `acknowledge(trained=False)` (backend stale drop) and on `reject_pending`, uses become `stale_rejected` and the task returns to `available` with attempts + 1, up to `max_task_attempts`, then `retired_untrained`.
- On `release_batch`, the reservation is cleared and the uses stay `training`; the next `build_batch` re-reserves the same groups.
- Groups discarded or dropped as stale before reservation make their use `untrainable` with the same attempt rule.
- No disposition counts as a successful use except `trained`; solver rewards and feedback are journaled independently of the training disposition.
- Before acknowledgement mutates the reservation, derive the exact trained-use/report association from the ready batch and persist it in `metrics.record2batch` on successful training commits. Write matching per-use disposition metadata on stale receipts. These fields are owned by the trainer/pipeline, not supplied by the backend, and do not require a commit-schema change.
- Recovery case 5 above.
- The sampler check (D4): a batch whose trajectory versions include one outside the scenario's served list is already stale under #819. Additionally, refuse at configuration time a rollout model setting that names a sampler other than the scenario's serving runtime.

### Completion criteria

- A valid batch trains, and its uses and records are consumed exactly once through the scenario commit path.
- Kill tests:
  - after the commit but before the store update, recovery marks the uses `trained` and does not retrain;
  - after a provisional store update but before commit, recovery does not mark the use trained without matching durable evidence;
  - after the stale receipt, the uses are `stale_rejected`;
  - before either, the batch re-forms from replayed reports without a new rollout.
- A released, skipped or measurement report appearing in another batch's `consumed_ids` never proves that its task use trained. Test unrelated successful commits, unrelated stale receipts and composite-component mismatches.
- A matching commit's per-use metadata names only reports that contributed training samples, and every named report is present in that commit's consumption progress.
- Consumed reports with no provable matching disposition cause an explicit unresolved recovery failure, without another rollout or inferred training success.
- An unknown outcome is never retried by re-rolling out while its reports are unconsumed.
- A stale or no-update disposition does not corrupt rewards or feedback, and follows the attempt rule.
- Source records are not held until training.

**Validation:** scenario replay and recovery tests (`tests/scenario/`), crash injection around `prepare_commit`, `commit` and `reject_pending`, and focused store reconciliation tests.

## PR 7 Switch the recipe and remove old orchestration

**Issue:** #833.

### Changes

- `rollout: true` becomes the default for `GeneratedTaskRecipe`.
- Remove:
  - the combined round loop;
  - `GeneratedTaskTasks.played()`;
  - the PR 3 transitional play job;
  - `reported_play_counts`;
  - the task-name `TaskGroupGrouping` key;
  - `recorded_rounds` and `recorded_reported_plays` (round-summary recovery).
- Settings move to their owners:
  - `rounds`, `batches_per_round`, run limits and refill to `pipeline.record2record.config`;
  - the generator, description, attempts and deadline to `pipeline.record2task.config`;
  - `tasks_per_step`, `plays_per_task`, `episode_capacity`, `execution_concurrency`, the runner and the model to `pipeline.task2batch.config`.
  - Reef **refuses** the old key locations at startup with a message naming the new location, as in #766.
- Update the example deployments, the pipeline developer guide, the API reference and the configuration docs. Cover: supported sandbox/task combinations, staging, the hosted-model rule (D4), recovery, the fresh-store upgrade, and the D5 eval behaviour.
- Run the RFC's live SPADE or AutoEnvScaling comparison against the old path. Record the configuration, the intentional behaviour changes and the results under `handoff/notes/`.

### Completion criteria

- A controlled end-to-end run selects sources, generates and stores tasks, selects uses, rolls out, groups, checks freshness and acknowledges a batch through the default recipe.
- The same flow recovers from kills during generation, rollout, feedback delivery and acknowledgement.
- Eval isolation, one training use per task and the capacity bound hold end to end.
- No old orchestration remains; unaffected recipes and the #766 goldens are unchanged; the old keys are refused.
- The live comparison is complete before merge. **Its billing source and budget need the user's explicit approval before it runs** (per the standing rules: ChatGPT plan through the :8900 stack, no OpenAI credits; Tinker only if approved).
- Ruff, type checks and required PR checks pass; unavailable checks are recorded.

## Research controls

| Study | Controls exposed | Deferred decision |
| --- | --- | --- |
| Batch size and stability | `tasks_per_step` independent of `episode_capacity`/`execution_concurrency`; usable group and token counts in status | Production batch size |
| Fixed vs refreshed sampler | Producing versions on every trajectory; D4's configuration check | Objective and refresh frequency |
| Generation and repair budgets | `generation_attempts`, `generation_deadline_seconds`, run limits, failure and latency counts | Numbers; whether to add repair |
| Task reuse | `max_task_attempts`, retained `trained` tasks, separate uses | Whether to reselect trained tasks |

Resource note: validation sandboxes and rollout sandboxes share the provider quota (E2B team account). Document that `max_generation_jobs × validation sandboxes + episode_capacity` must fit within it.

## Executor rules

- Implement PR by PR in order; for each PR, write the failing tests from its completion criteria first.
- Follow the repository's `CLAUDE.md` in the worktree: minimal diffs, an ABC for replaceable behaviour, no unused frameworks.
- No model calls, paid runs or GitHub writes. Start PR 1 locally under the user's authorization and stop to resolve a contract only when it affects the current work:
  - an unanswered design decision affects the current PR;
  - the code contradicts this plan;
  - #800/#801/#829/#787 changes the ground.
- Record deviations from this plan in a short note under `handoff/notes/` per PR.
