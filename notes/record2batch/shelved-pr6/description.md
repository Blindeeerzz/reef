## Motivation

PR 5 rolls saved tasks out as batches, but nothing durable records what became of each rollout.
After a restart, a task that trained could roll out again, a task whose batch was dropped stale was lost until its generation reopened, and a batch whose plays had ended had to roll out again.
A commit's consumed IDs cannot answer these questions, because the trainer also puts released and skipped records there.

This implements PR 6 of [#833](https://github.com/Human-Agent-Society/reef/issues/833), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): task uses, batch outcomes in commits, and recovery.
It builds on PR 5 (the rollout loop). The loop stays opt-in until PR 7.

## Changes

- **Task uses** (`reef/record2task/store.py`): each rollout of a saved train task is a `TaskUse`, saved under `uses/`.
  - States: `running` → `rolled_out` → `training` → `trained` | `stale_rejected` | `untrainable` | `cancelled`.
  - A task with an active or trained use does not roll out. After an untrained end it is available again, until `max_task_attempts` untrained uses (default 2) retire it.
  - Store counts show `in_use/train`, `trained/train` and `retired/train`.
- **Rollout hooks** (`RolloutTask2Batch`, `reef/task2batch/base.py`), with no-op defaults:
  - `rollout_training(batch, training_reports)`: the batch formed; these reports, by task use, make its samples.
  - `rollout_ended(outcome)`: the batch trained or was dropped. A use missing from the outcome could not train.
  - `recover(outcomes, consumed_ids) -> RecoveredRollout | None`: settle the uses of earlier runs before records replay.
- **Batch outcomes in commits**: the pipeline exposes `batch_metadata()` for the batch handed out. The trainer takes it at reservation and writes it to the commit's `metrics.record2batch`, or to the stale receipt's metadata. Only after the commit or the receipt is durable does the trainer call `TrainingPipeline.settle_batch`, which ends the uses; acknowledgement alone leaves them `training`:

  ```json
  {"batch_id": "coding:rollout:3f2a…", "uses": [{"task_use_id": "use-harbor-00000-000-000", "training_report_ids": ["rep-1", "rep-2"]}]}
  ```

- **Recovery** (`reef/scenario/factory.py`): `recovered_trainer_states` collects each trainer's `RolloutBatchOutcome`s:
  - from commits that consumed every named report (trained), including a commit held for review, whose reports never replay;
  - from stale receipts (dropped).

  `Scenario.recover_rollouts` hands them to the pipeline before records replay.
- **`GeneratedTaskBatches`** creates uses on submit, marks them rolled out when the plays end, and applies the hooks:
  - A task whose use ended untrained goes back to the task pool.
  - The generator's feedback comes from a task's first play. A task retired before any play is reported as an infrastructure failure, so its generation's feedback can complete.

### What a recipe can do now

A rollout stage that keeps its own task state follows each batch to its end and recovers after a restart:

```python
class MyRollout(RolloutTask2Batch):
    def rollout_training(self, batch, training_reports):
        self.ledger.mark_training(batch, training_reports)        # use → reports that made its samples

    def rollout_ended(self, outcome):
        for task_use in self.ledger.uses(outcome.batch):
            if task_use not in outcome.training_reports:
                self.ledger.end(task_use, "untrainable")
            else:
                self.ledger.end(task_use, "trained" if outcome.trained else "stale_rejected")

    def recover(self, outcomes, consumed_ids):
        # outcomes come from this trainer's commits and stale receipts, read before replay
        ...
        return RecoveredRollout(batch, task_uses, reported_plays)  # or None: no batch re-forms
```

## Recovery cases

| State at the stop | After restart |
| --- | --- |
| Use still playing | `cancelled`; the task returns to the task pool (one attempt). |
| Plays ended, batch not built, not committed, or its commit failed; reports unconsumed | The batch re-forms from the replayed reports, without another rollout. |
| Commit recorded the batch | Named uses `trained`; other uses of the batch `untrainable`. |
| Stale receipt recorded the batch | Named uses `stale_rejected`; their tasks return to the pool. |
| Training reports consumed, but no commit or receipt records the batch | Recovery stops with an error naming the use and batch. |

## Documentation impact

- Reader question: what happens to a rolled-out task after its batch trains, is dropped, or a restart interrupts it?
- Updated pages: the pipeline guide (`docs/developer-guide/processors.rst`) describes task uses, attempts and recovery; the configuration reference lists `max_task_attempts`.
- Verification: checked against the implementation and tests. The docs site build was not run.

## Compatibility and operational impact

- Off by default with `rollout: false`; the play-set path keeps its availability rule (a task without uses is available until it has a result).
- Commit metrics and stale receipts of rollout batches gain a `record2batch` object. No schema change.
- Task use IDs are `use-<task>-NNN`, and the episodes' `task_use` label is the use ID.
- A refused submit removes the uses it began; no attempt is counted.

## Verification

- `pytest tests/record2record tests/record2task tests/train tests/reef_recipe/test_generated_tasks.py tests/scenario`: all pass (including 19 new tests).
- Full suite with `--continue-on-collection-errors`: 5751 passed, 202 skipped, 12 failed, 19 collection errors. The 12 failures also fail on the PR 4 base: 11 environment failures (sandbox jail, vLLM, slime, recipe imports) and one flaky harness-wrapper terminal-kill test. The collection errors are missing training dependencies.
- `pre-commit run --all-files`: passed. `mypy`: only the existing `reef/observability/open_telemetry.py` errors.

This change was written with AI assistance (Claude Code). I reviewed it.
