## Motivation

A record-to-task stage could not return a result later.
Slow work, such as a task-generation round, had to run through `Record2Task.poll()` and `background_jobs_running()`, outside the `make_tasks` contract.
The generated-task stage also drove the record stage's rounds through `round_due`, `start_round`, `round_finished` and `stop_rounds`.

This implements PR 2 of [#832](https://github.com/Human-Agent-Society/reef/issues/832), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): pending results and round sets.
It builds on the task store PR for [#831](https://github.com/Human-Agent-Society/reef/issues/831).

## Changes

- Add `TaskStatus.PENDING` and `TaskResult.pending()`.
  `make_tasks` can return pending, and the stage hands the result back through the new `take_finished()`.
- The pipeline holds a pending set and its records until the set settles.
  It settles each set once, logs and ignores a duplicate or unknown result, and settles nothing after `close()`.
- `TrainingPipeline.background_jobs_running()` stays the trainer's signal, and the pipeline now derives it from pending sets and the record stage's jobs.
- Remove `Record2Task.poll` and `Record2Task.background_jobs_running`.
- `TrainingPipeline.close()` also closes the task-to-batch stage.
- `RoundRecord2Record` owns round pacing.
  When a round is due, it forms a `RoundSet` with the round number and its source sessions, and it stops at the round limit.
  The round ends when the set settles; a set that settles as anything but `READY` makes the next round due at once.
- `GeneratedTaskTasks` runs the existing combined round through this path.
  A played round settles `READY` with no tasks, because its episodes still train as reports. A round that played none settles `SKIPPED`.

### What a stage can do now

A record-to-task stage can run slow work, such as a model call or a sandbox check, without blocking the trainer.
It submits the work in `make_tasks`, returns pending, and hands the result back through `take_finished`:

```python
class ModelWrittenTasks(Record2Task, BackgroundJobRunner):
    """Turns each record set into one task that a slow model call writes."""

    def __init__(self) -> None:
        self.worker = BackgroundJobWorker(self, thread_name="model-written-tasks")
        self.set_ids: dict[str, Hashable] = {}

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        receipt = str(record_set.set_id)
        if not self.worker.submit(TaskJob(receipt=receipt, record_ids=record_set.record_ids())):
            raise RuntimeError("the task worker is closed")
        self.set_ids[receipt] = record_set.set_id
        return TaskResult.pending()

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        task_path = await write_task_from_model(job.record_ids)  # runs on the worker, off the trainer thread
        return TaskWritten(receipt=job.receipt, task=TaskItem(task_path, source_agent_record_ids=job.record_ids))

    def take_finished(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        finished = []
        for result in self.worker.poll():
            set_id = self.set_ids.pop(result.receipt)
            if isinstance(result, TaskWritten):
                finished.append((set_id, TaskResult.ready((result.task,))))
            else:
                finished.append((set_id, TaskResult.skipped()))
        return tuple(finished)

    def close(self) -> None:
        self.worker.close()
```

The pipeline then does the rest:

```python
pipeline.ingest(record)                 # make_tasks returns pending; the set and its records are held
pipeline.ready()                        # False while the job runs
pipeline.background_jobs_running()      # True, so the trainer keeps polling
# ... the job finishes on the worker ...
pipeline.ready()                        # take_finished settles the set; True without another record
pipeline.build_batch()                  # the batch holds the task the job wrote
```

A stage written for the old hooks fails at import with a direct instruction:

```text
TypeError: OldTasks defines poll, which Record2Task no longer calls:
return TaskResult.pending() from make_tasks and hand the result back through take_finished
```

Stages that return results immediately need no change.

## Documentation impact

- Reader question: how does a record-to-task stage run slow work without blocking the trainer?
- Updated pages: the pipeline guide (`docs/developer-guide/processors.rst`) has a new "Pending tasks" section and updated `ready()`, `close()` and generated-task round text. The API reference (`docs/reference/python-api.rst`) names the pending hooks.
- Verification: the text was checked against the implementation and the contract tests. The example above ran against this branch with a stand-in model call. The docs site build was not run.

## Compatibility and operational impact

- A custom `Record2Task` that defines `poll` or `background_jobs_running` must move to `take_finished`. In this repository, only `GeneratedTaskTasks` used these hooks.
- `RoundRecord2Record.start_round`, `round_finished` and `round_sources` are removed. `resume_rounds(next_round=..., round_limit=...)` sets the round range.
- A worker that refuses a round, because it is closed or broken, now raises from `ready()` instead of stopping the rounds silently.
- After `close()`, `ready()` returns False and starts no round.
- Configuration, records, commit rows, status keys and the task store format do not change.
  `rounds` stays in `pipeline.record2task.config` until a later PR of #833 moves it.

## Verification

- Focused suites for the changed contracts: 227 passed.
  These are `tests/reef_recipe/test_generated_tasks.py`, `tests/train/test_record_stage.py`, `tests/train/test_trainer.py`, `tests/reef_service/test_manual_training.py`, `tests/recipes/sdpo/test_sdpo_pipeline.py` and `tests/recipes/openclawrl/test_openclawrl.py`.
- Broad regression over `tests/core/test_task_split.py`, `tests/reef_recipe`, `tests/recipes`, `tests/train`, `tests/scenario`, `tests/record2record`, `tests/record2task` and `tests/reef_service/test_manual_training.py`: 1673 passed, 32 skipped, 1 failed, 1 collection error.
  - `test_reefine_recipe.py::test_profile_builds_without_tutorial_or_training_imports` fails on the base branch too: its isolated subprocess cannot import `reef` in this environment.
  - `tests/recipes/sao/test_sao_bridge.py` needs Ray, which this environment does not have.
- `pre-commit run --all-files`: passed.
- Targeted mypy on the seven changed source files: passed.
- No model calls, sandboxes or training runs.

## AI assistance

Claude Code implemented the changes, tests and documentation under the contributor's direction.
The contributor remains responsible for reviewing and understanding the changes.

## Checklist

- [x] The change is focused and contains no unrelated cleanup.
- [x] Changed code satisfies the style, code structure, and naming rules in `AGENTS.md`.
- [x] Tests cover behavior changes.
- [x] Public interface changes include contract tests (`tests/train/test_record_stage.py`).
- [x] Affected documentation is updated and reviewed against the Documentation review checklist in `AGENTS.md`.
- [x] Root README content is unchanged.
- [ ] An accepted RFC is linked: RFC #805 is linked; its later deviations still need maintainer acknowledgement.
- [x] Relevant pre-commit and test checks pass.
- [x] Non-trivial AI assistance is disclosed above.
