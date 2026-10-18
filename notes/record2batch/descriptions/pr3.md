## Motivation

A generated-task round generated, validated and played its tasks in one job, and sent the results to the generator at the end of the round.
So generation could not finish before its tasks ran, a failure to run stopped the round, and a generator lost its feedback when the next generation started or the process restarted.

This implements PR 3 of [#832](https://github.com/Human-Agent-Society/reef/issues/832), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): generation without play, task-level play from a task pool, retries, and delayed feedback.
It builds on the pending-result PR and the task store PR for [#831](https://github.com/Human-Agent-Society/reef/issues/831).

## Changes

- Generalize the pool: `PoolItem` is the base of `RecordSet` and the new `TaskSet` (one saved task). `Pool` is generic, `RecordSetPool` stays as its record-set form, and selection policies take any pool item.
- A generation job generates, validates and saves tasks, and plays none.
  A generator or check that fails to run is retried from the saved candidates, up to `generation_attempts` times within `generation_deadline_seconds`.
  After that the round settles as `SKIPPED`: it counts toward the `rounds` limit, so a generator that always fails stops there.
- `RoundRecord2Record` holds a task pool. A round that made train tasks puts them in it. When no play runs, the stage takes the pool's tasks into a `PlaySet`, and a play job plays them and saves each result in the store.
- The next round starts only while the task pool holds fewer than `refill_tasks` tasks, so saved tasks play before more are generated.
- The `rounds` limit counts finished generations. A round of replayed sessions does not use it up. A task that an earlier generation published is rejected as a duplicate.
- A task that cannot run at all has an infrastructure failure as its result (`TaskPlayResult.infrastructure_failure`).
- Delayed feedback: when every train task of a generation has a result, the next job sends the generation's results with `record_results(results, generation=...)`. The store records the delivery, so a restart sends only what the generator did not receive, and sends it at startup even when no other job is due. A generation whose delivery fails does not hold back later ones.
- SPADE and AutoEnvScaling keep each open generation in `open-<generation>.json` until its results arrive, so a generation can close after later ones started and after a restart. A second delivery does nothing. SPADE records each Designer report as it is sent, so a delivery that stopped part way does not report a proposal twice, and its experience never goes back to an older generation. Tasks that could not run get no Designer reward (SPADE) and count as an outage (AutoEnvScaling).
- A restart puts the unplayed train tasks back in the task pool, closes their groups from the saved reported counts, and does not generate again from replayed sessions that a complete generation used. A task whose play a stop interrupted is staged again and plays. The round summaries under `store/plays/` are gone; the store's generation documents hold the play results.

### What a deployment can do now

Generation and play are separate, and both survive a restart.
The new settings control pacing and retries:

```yaml
pipeline:
  record2record:
    config:
      plays_per_task: 4
      batches_per_round: 2
      source_tag: source=session
      source_records_per_round: 4
      refill_tasks: 8            # generate only while fewer than 8 saved tasks wait to play
  record2task:
    config:
      rounds: 100                        # generations to finish
      generation_attempts: 3             # a generator or check that fails to run is retried
      generation_deadline_seconds: 1800  # within this time; then the round is skipped
```

A generator now receives each generation's results by number, possibly after later generations and after a restart:

```python
class MyGenerator(TaskGenerator):
    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        tasks = await self.propose(request)
        self.save_open(request.generation, tasks)  # keep what record_results needs, across a restart
        return tasks

    async def record_results(self, results: tuple[TaskPlayResult, ...], *, generation: int) -> None:
        state = self.load_open(generation)
        if state is None:
            return  # already closed: a second delivery does nothing
        for result in results:
            if result.infrastructure_failure:
                continue  # the task could not run; not a solver failure
            ...
        self.close_open(generation)
```

## Documentation impact

- Reader questions: how do generation, play and feedback relate now, which settings pace them, and what must a generator keep?
- Updated pages: the pipeline guide (`docs/developer-guide/processors.rst`, generated tasks and the generator contract), the configuration reference (new record-stage and record-to-task settings, store behaviour), the API reference (`record_results` and `TaskGenerationRequest.generation`), and the SPADE and AutoEnvScaling READMEs (open generations, the pool-building script).
- Verification: checked against the implementation and tests. The README's pool-building script was not run against a live generator service. The docs site build was not run.

## Compatibility and operational impact

- Generator contract: `record_results` gains a required keyword `generation`, and `TaskGenerationRequest` gains `generation` (default 0). Every in-repository generator is updated. An external generator must accept the keyword and keep per-generation state.
- `TaskPlayResult.infrastructure_failure` (default False) is new. `rounds` now counts finished generations.
- `GeneratedTaskTasks.run_round`, `RoundJob` and `RoundOutcome` are replaced by `run`, `GenerationJob`/`GenerationOutcome` and `PlayJob`/`PlayOutcome`. Status `task_generation` drops `tasks_generated` and `tasks_played` (the `task_store` counts cover them) and adds the task pool size to the record stage's status.
- Behaviour changes: a generation that fails to run is retried and then skipped, instead of playing what it validated so far. A task whose play was running at a stop plays again after the restart; exactly-once plays come with the rollout PRs of #833.
- Known limits:
  - Rounds still wait for batches. When the played groups cannot fill a batch (fewer tasks than `tasks_per_step`, groups of one), no further round starts. The rollout loop of #833 replaces this pacing.
  - `refill_tasks` counts the tasks that wait in the pool. A play set takes every waiting task, so tasks in a queued or running play are not counted.
  - Deferred rounds are kept in memory. After a restart, their sessions come back through the record replay; if they group differently, a new generation starts and the old, incomplete one stays in the store.
  - Results go to the generator once per generation, not by result ID. A generator keeps open-generation state for deduplication.
  - AutoEnvScaling saves its session cursor after each session, so a retried generation continues with the next sessions instead of the ones a failed attempt took.
- New stores keep play results in their generation documents. Records, commit rows and the split manifest format do not change.
- The `rounds` key stays in `pipeline.record2task.config`; a later PR of #833 moves it.

## Verification

- `pre-commit run --all-files`: passed.
- Targeted mypy on the 21 changed source files: passed.
- Broad regression over `tests/core`, `tests/reef_recipe`, `tests/recipes`, `tests/train`, `tests/scenario`, `tests/record2record`, `tests/record2task`, `tests/contributing` and `tests/reef_service/test_manual_training.py`: 1900 passed, 32 skipped, 1 failed, 1 collection error.
  - `test_reefine_recipe.py::test_profile_builds_without_tutorial_or_training_imports` fails on the base branch too: its isolated subprocess cannot import `reef` in this environment.
  - `tests/recipes/sao/test_sao_bridge.py` needs Ray, which this environment does not have.
- New tests cover: generation without play, play from the task pool, a restart that plays only unplayed siblings, no regeneration from replayed sessions, retries that resume from saved candidates, a generator that always fails stopping at the round limit, the generation deadline, infrastructure failures, redelivery after a failed delivery, store results and lookups, the generic task pool, and delayed, restart-safe feedback in both generators.
- No model calls, sandboxes or training runs.

Logs: `handoff/notes/record2batch-pr3-{focused-tests,pre-commit,targeted-mypy}.log`.

## AI assistance

Claude Code implemented the changes, tests and documentation under the contributor's direction.
The contributor remains responsible for reviewing and understanding the changes.

## Checklist

- [x] The change is focused on #832's planned work.
- [x] Changed code satisfies the style, code structure, and naming rules in `AGENTS.md`.
- [x] Tests cover behavior changes.
- [x] Public interface changes include contract tests.
- [x] Affected documentation is updated and reviewed against the Documentation review checklist in `AGENTS.md`.
- [x] Root README content is unchanged.
- [ ] An accepted RFC is linked: RFC #805 is linked; the task pool (D3) and the `TaskGenerator` amendment (D6) still need maintainer acknowledgement.
- [x] Relevant pre-commit and test checks pass.
- [x] Non-trivial AI assistance is disclosed above.
