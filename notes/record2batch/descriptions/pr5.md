## Motivation

Saved generated tasks left the task pool only as play sets: each task played on its own, and its episodes trained in whatever batch they reached.
A training batch therefore mixed episodes of tasks from different rollouts, and nothing bounded how many episodes ran at once.

This implements PR 5 of [#833](https://github.com/Human-Agent-Society/reef/issues/833), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): task-level rollout batches with an episode capacity.
It builds on #836 to #839. The rollout loop replaces the play-set path.

## Changes

- **The rollout loop** (`reef/train/pipeline.py`): when no rollout runs, the pipeline reserves up to `tasks_per_step` tasks from the record stage's task pool, submits them as one rollout batch, and only then takes them from the pool (a failed submit leaves them there). When the plays end, it hands each task use's training reports to the record stage. Once every group of the batch settled, the staleness check runs and the batch trains on its surviving groups; with none left, the batch ends and the next rollout starts.
- **Generic parts:**
  - `RolloutReportRecord2Record` and `RolloutGrouping` (`reef/record2record/rollout.py`): a report stage with a task pool (`Pool[TaskSet]`) that groups training reports by `(batch, task_use)` from the episode labels.
  - `RolloutTask2Batch` (`reef/task2batch/base.py`): `submit(batch, tasks) -> task_uses`, `take_finished() -> RolloutResult(batch, training_reports)`, `running()`, `rollout_trained(batch, task_uses)`, and `settle_timeout_seconds`.
  - `Pool.reserve` and `Pool.select` take an optional predicate (`where=`).
- **Settling:** a group settles when every report its plays returned joined it or ended without training (skipped, or refused by the report contract), or after `settle_timeout_seconds`. Several reports of one episode count as one episode; a group of fewer than two episodes is discarded. A report whose batch is not running, or whose arm is not the training arm, never trains; a report that arrives after its batch was built is released when the batch ends.
- **`GeneratedTaskBatches`** plays a rollout batch on its own worker through the episode runner, with at most `episode_capacity` episodes queued or running (0 = `tasks_per_step` x `plays_per_task`); a task with more plays plays in waves. Each episode carries the `batch` and `task_use` labels. If one task's play fails, the other tasks play on.
- **Training state:** the tasks of trained groups are marked trained in the task store. On a restart, train tasks that did not train go back to the task pool. The generator receives each task's first play result.
- **Removed:** play sets (`PlaySet`, the play job, `reported_play_counts`) and grouping by task name. A rollout that trained nothing lets the next generation round start at once.

### What a recipe can do now

```yaml
pipeline:
  record2record:
    config:
      plays_per_task: 8
  task2batch:
    config:
      tasks_per_step: 4
      episode_capacity: 16          # at most 16 episodes queued or running; 0 = 4 x 8
      settle_timeout_seconds: 600   # a batch waits this long for missing reports, then trains on what arrived
```

Any record-to-task stage that returns `TaskItem` values can feed the loop with its own rollout stage:

```python
class MyRollout(RolloutTask2Batch):
    def submit(self, batch, tasks):
        uses = tuple(f"{batch}/{task.task_path.name}" for task in tasks)
        for use, task in zip(uses, tasks):
            self.worker.submit(play(task, labels={"batch": batch, "task_use": use}))
        return uses

    def take_finished(self):
        # one result per ended batch: the report IDs of each use's training episodes
        return tuple(RolloutResult(batch, reports) for batch, reports in self.worker.take_done())

    def running(self):
        return self.worker.busy

    def rollout_trained(self, batch, task_uses):
        self.ledger.mark_trained(task_uses)

    def make_batch(self, tasks, batch_number):
        return TrainingBatch(f"rollout:{batch_number}", tasks)


stage = RolloutReportRecord2Record(context, RolloutGrouping(), exclusive_sources=True)
stage.add_tasks(saved_tasks, group_key=round_number)
pipeline = TrainingPipeline(context, record2record=stage, record2task=MyTasks(), task2batch=MyRollout(context, 4))
```

## Documentation impact

- Reader question: how do saved tasks become a training batch, and what limits the episodes that run at once?
- Updated pages: the pipeline guide (`docs/developer-guide/processors.rst`) describes the rollout loop, settling and restarts; the configuration reference lists `episode_capacity` and `settle_timeout_seconds`, and drops the play-set wording.
- Verification: checked against the implementation and tests. The docs site build was not run.

## Compatibility and operational impact

- The generated-task recipe now always rolls out; the play-set path and grouping by task name are gone. `GeneratedTaskBatches` needs the generated-task record and record-to-task stages.
- `play_and_record` and `played` take `labels` and return the training episodes' report IDs.
- The task store gains `trained_task_names` per generation; `available_tasks` means "not trained yet". Store counts show `trained/train`.
- The `generated_tasks` stream of the record-stage golden parity test is removed: it covered task-name grouping.
- Known limits:
  - A task is marked trained when its batch is acknowledged, before the commit is durable. A crash in between loses those marks (the tasks do not roll out again); a crash before acknowledgement replays the tasks.
  - A task whose play job failed reports no episodes; its late reports are discarded, and it rolls out again after a restart.

## Verification

- `pytest tests/reef_recipe tests/record2record tests/record2task tests/train tests/recipes tests/scenario --continue-on-collection-errors`: 1575 passed, 32 skipped. The 2 failures (reefine recipe imports) and 1 collection error (`sao_bridge`, training dependencies) also fail on the base.
- `pre-commit run --all-files`: passed. `mypy`: only the existing `reef/observability/open_telemetry.py` errors.

This change was written with AI assistance (Claude Code). I reviewed it.
