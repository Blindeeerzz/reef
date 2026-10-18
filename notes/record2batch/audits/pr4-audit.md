# PR 4 adversarial audit (2026-10-17)

Audit of the PR 4 commit `6a6350b7` (fixes amended into `b84ba557`) by a separate reviewer agent; the implementing agent confirmed and fixed the findings.
Reproductions are in `/mnt/data3/reef-r2d-eval/.cache/audit4/test_audit4.py`.
The fixes are folded into the amended PR 4 commit, each with a test.

## Fixed

| Finding | Fix | Test |
| --- | --- | --- |
| Validation passed a task whose same copy was already staged, then deleted that copy (a play could lose its files) | `WrittenTask.created` (201 vs 200); SPADE and AutoEnvScaling validation delete only a copy they created | `test_a_staged_copy_of_the_same_task_passes_validation_and_stays_staged` |
| `unstage` could delete a copy another stage still used | `GeneratorEpisodeRunner` counts stages and removes a copy only when it created it and no later stage reused it | `test_unstage_removes_only_a_copy_this_runner_created_and_no_other_stage_reuses` |
| Reserved or invalid play tags (`task`, `episode`, empty value) failed the whole play job inside the job; `arm` was silently overwritten | `POST /plays` checks tags as task-player labels and refuses `arm`, with 400 before any play | `test_a_play_runs_the_arm_with_its_files_and_comes_back_as_episodes` (extended) |
| An `unstage` error after an infrastructure failure replaced the episode's error | The play path logs an unstage `GeneratorError` and keeps the episode error | reviewed |
| The runner docstring said a staged copy exists only while the task plays | Docstring and docs say staged copies stay, because duplicate detection reads them | reviewed |
| Nothing checked that `batch`/`task_use` labels reach inference tags and reports | Task-player test with those labels | `test_batch_and_task_use_labels_reach_the_inference_tags_and_the_report` |
| Test stub raised on an unreadable staged copy where the service answers "not staged" | Stub mirrors the service | stub |

## Confirmed unchanged

`Episode.is_infrastructure_failure` (no reward and an error) matches the old `rewards_of` filter and the old all-plays-failed check: played counts, reported counts and group completion are unchanged.

## Documented limits

- A staged copy whose files no longer match its digest still cannot be staged again; the play records the task as refused rather than deleting a copy that may belong to another task.
- Two Reef processes sharing one generator service do not coordinate staged copies.
- Duplicate detection reads staged copies, not the task store.
- Plan items deferred to a runner that executes locally: refusing an unsupported task and environment pair before any model call, and cleanup of owned versus borrowed sandboxes on cancel.
