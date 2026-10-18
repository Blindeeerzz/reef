## Motivation

The generated-task play job wrote each task through the generator service and played it with the service client directly.
A second write of the same task was refused as a duplicate, so a play that a stop interrupted could not stage its task again.
The play results also mixed episodes that could not run with episodes that scored a failure.

This implements PR 4 of [#833](https://github.com/Human-Agent-Society/reef/issues/833), under [RFC #805](https://github.com/Human-Agent-Society/reef/issues/805): a shared episode runner and staging that you can repeat.
It builds on the PRs for [#831](https://github.com/Human-Agent-Society/reef/issues/831) and [#832](https://github.com/Human-Agent-Society/reef/issues/832).

## Changes

- Add `EpisodeRunner` (`reef/record2task/generator/runner.py`), which stages a saved task through the generator service and plays its episodes.
  - `play` returns one `Episode` per played episode: its ID, reward, report record IDs and error.
  - `Episode.is_infrastructure_failure` marks an episode with an error and no reward, which could not run.
  - `play` takes `labels`, which the task player puts on each episode's inference tags and report. The rollout PR passes the batch and task-use labels here.
- The generator service's `POST /tasks` answers 200 with the staged path when the same task (same name and digest) is already under its name, and 201 when it creates the copy. `WrittenTask.created` tells the two apart.
  A different task under the name is still refused (409), and the same content under another name is still a duplicate (409).
- SPADE and AutoEnvScaling validation delete their written copy only when they created it, so validation never removes a copy that a play may use.
- `POST /plays` checks its tags as the task player's labels and refuses `arm`, the player's own tags (`task`, `episode`) and empty values with 400, before any play runs.
- The task store stays the source of each task; the service's tasks root holds staged copies, which stay after a play because duplicate detection reads them. A play that a stop interrupted stages again, and a service that restarted with an empty tasks root can still play a saved task.
- The generated-task play job uses the runner. This replaces the delete-and-rewrite step after `DuplicateTask`. A task whose training episodes all could not run is an infrastructure failure; its staged copy stays, like every other.

### What a caller can do now

The rollout stage (and any other caller) plays a saved task without knowing where it runs:

```python
runner = EpisodeRunner(HttpGenerator(generator_url))

task_path = await runner.stage(saved.generated.task)   # the same path again after a stop or a redeploy
episodes = await runner.play(
    task_path,
    scenario="coding",
    arm="plain",
    plays=4,
    extra_instruction_files=(),
    labels={"batch": batch_id, "task_use": task_use_id},  # on every inference tag and report
    model=None,
)
rewards = [episode.reward for episode in episodes if not episode.is_infrastructure_failure]
reported = sum(episode.is_reported for episode in episodes)
```

## Documentation impact

- Reader question: how does a saved task reach the sandbox, and what happens when that is interrupted?
- Updated pages: the pipeline guide (`docs/developer-guide/processors.rst`) describes the episode runner, staging that you can repeat, and infrastructure failures. The generator service's module docstring describes the 200 answer of `POST /tasks`.
- Verification: checked against the implementation and tests. The docs site build was not run.

## Compatibility and operational impact

- `POST /tasks` with a task that is already staged under its name now answers 200 instead of 409. Callers that used the 409 to detect the very same task no longer get it; the same content under another name still answers 409.
- AutoEnvScaling's and SPADE's validation write a task, check it and delete it. A copy of the very same task, left by a stop, no longer makes validation refuse it as a duplicate.
- `POST /plays` now refuses tags that the task player would refuse, with 400 instead of a failed play job.
- No configuration, records or store format change.
- Known limits:
  - A staged copy whose files no longer match its digest still cannot be staged again: the service refuses the name, and the play records the task as refused. The play path does not delete a copy that may belong to another task.
  - Two Reef processes that share one generator service do not coordinate: the process that created a copy may remove it after an infrastructure failure while the other plays it.
  - Duplicate detection reads the staged copies, not the task store, so a service with an empty tasks root does not see earlier tasks as duplicates.
- Not in this PR:
  - The runner plays through the generator service, whose task player owns the sandboxes. The #803 constraints apply there unchanged: ordinary cleanup leaves a borrowed E2B VM running and cancellation retires it, synchronous bridges run outside an event loop, and another execution user, a forced template rebuild, Daytona and Modal are unsupported.
  - Checks that refuse an unsupported task and environment pair before any model call, and cleanup of owned versus borrowed sandboxes, belong to a runner that executes locally. None does yet.

## Verification

- Episode runner tests: staging again after a stop, staging on an empty root (a redeployed service), labels passed to the plays, infrastructure failures, a different task under a staged name, and unstaging only a copy that the runner created and no other stage reuses.
- Generator service tests: writing the same task again returns it with `created` False; plays with `arm`, a reserved tag or an empty value are refused with 400 and never run.
- Task-player test: `batch` and `task_use` labels reach the inference tags (`x-reef-tag-batch`, `x-reef-tag-task_use`) and the report's `episode.labels`.
- AutoEnvScaling test: a staged copy of the same task passes validation and stays staged.
- Generated-task tests: an interrupted play stages again without a delete, and the episode rewards and infrastructure failures.
- `pre-commit run --all-files`: passed.
- Targeted mypy on the changed source files: passed.
- Broad regression over `tests/core`, `tests/reef_recipe`, `tests/recipes`, `tests/train`, `tests/scenario`, `tests/record2record`, `tests/record2task`, `tests/harness/test_task_player.py`, `tests/sandbox`, `tests/contributing` and `tests/reef_service/test_manual_training.py`: 1952 passed, 32 skipped, 1 failed (`test_reefine_recipe`, also on the base: its subprocess cannot import `reef` here), 1 collection error (`test_sao_bridge` needs Ray).
- No model calls, sandboxes or training runs.

## AI assistance

Claude Code implemented the changes, tests and documentation under the contributor's direction.
The contributor remains responsible for reviewing and understanding the changes.

## Checklist

- [x] The change is focused on #833's planned work.
- [x] Changed code satisfies the style, code structure, and naming rules in `AGENTS.md`.
- [x] Tests cover behavior changes.
- [x] Public interface changes include contract tests.
- [x] Affected documentation is updated.
- [x] Root README content is unchanged.
- [ ] An accepted RFC is linked: RFC #805 is linked; the plan's deviations still need maintainer acknowledgement.
- [x] Relevant pre-commit and test checks pass.
- [x] Non-trivial AI assistance is disclosed above.
