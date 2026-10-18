# PR 5 adversarial audit (2026-10-17)

A separate reviewer agent audited the PR 5 commit `17b3d970`. The implementing agent confirmed the findings and fixed them in the amended commit `479e6fd2` (now `32905cc6` after the second audit) on `feat/record2batch-rollout-loop` (`reef-record2batch-pr5`).
Reproductions are in `/mnt/data3/reef-r2d-eval/.cache/audit5/test_audit5.py`.

## Fixed (confirmed bugs)

| Finding | Fix | Test |
| --- | --- | --- |
| A failed rollout job returned `RolloutResult(batch, {})`; no use got a count, so the batch never settled and no rollout ran again | The pipeline gives each task use without a count 0; a group with no report and count 0 settles | `test_a_rollout_that_reported_no_episode_settles_and_the_next_rollout_starts` |
| A skipped report counted in `reported_plays` but never joined its group, so the group waited forever | `RolloutReportRecord2Record.end_report` records the episode as ended; completion counts ended episodes | `test_a_skipped_episode_counts_toward_its_group` |
| A report that arrived while its batch waited for acknowledgement joined a group that nothing decided again | `end_rollout` discards the batch's leftover groups, which releases their reports | `test_a_report_that_arrives_while_its_batch_waits_is_released_when_the_batch_ends` |
| Reports without an episode ID each counted as a distinct episode, so segments of one episode made a READY group | Such a report never trains; it counts as ended by its report ID so the group still settles | `test_reports_without_an_episode_id_never_train_but_still_settle_the_group` |
| A rollout that trained nothing left `RoundRecord2Record` waiting for a batch, so the next round never started | `end_rollout(batch, trained=...)`; the round stage sets `follow_at_once` when not trained | `test_a_rollout_that_trains_nothing_lets_the_next_round_start` |

## Fixed (risks)

| Risk | Fix |
| --- | --- |
| A failed submit lost the reserved tasks | Reserve, submit, then consume; release the reservation when submit raises (`test_a_failed_submit_keeps_the_tasks_in_the_pool`) |
| `reported == 0` with leftover reports could train | `decide` needs `reported > 0` |
| Close order closed record-to-task before a rollout that still saves results | Close task-to-batch before record-to-task |
| A task store write error in a rollout job failed the whole batch | Superseded by the second audit's per-task catch (below) |
| In rollout mode, play-set reports still trained | `TaskGroupGrouping.trains_play_sets = False` with `forms_play_sets = False` |
| Status did not show admitted episodes | `rollout_capacity.admitted_episodes` |

## Deferred to PR 6

- A restart loses the running rollout batch; its tasks without a result come back from the task store at startup, and its reports end.
- The task store does not record task uses per rollout.

## Second audit (of `b8a17555`, after the `rollout_ready` refactor)

A second reviewer agent audited the fixed commit. Reproductions are in `/mnt/data3/reef-r2d-eval/.cache/audit5b/`. Its findings are fixed in `32905cc6`.

| Finding | Fix | Test |
| --- | --- | --- |
| A rollout job failed as a whole when one task's result save raised (a raw `OSError`, not `TaskStoreError`); the batch's tasks were neither played nor back in the pool, and the generation's feedback never went out | The rollout job catches each task's failure, logs it and counts 0 for it; when the rollout ends, `put_back_unplayed` returns every task of the batch without a saved result to the task pool | `test_a_task_whose_result_cannot_be_saved_goes_back_to_the_pool_and_the_batch_plays_on` |
| After a failed job, `asyncio.gather` left the sibling plays running while the next rollout started, past the episode capacity | Same per-task catch: the job no longer fails, so the batch waits for its siblings | covered by the test above |
| A training report that the report contract refused (or any other `end_report` path) never counted toward its group, so the loop stalled forever | `end_report` is overridden instead of `reject` and `handle_other_report`, so every ending path counts the episode | `test_a_training_report_that_the_contract_refuses_counts_toward_its_group` |

The `rollout_ready` refactor (dropping `is_finished`) was checked: no group can settle before its count arrives.

Known limits, documented and not fixed:

- An episode without an ID that sends two reports counts twice, which can close its group early and discard it. This needs malformed reports. The robust fix is to settle groups by the report IDs that each episode returned (`Episode.report_record_ids`), which changes the `RolloutResult` contract. Candidate for PR 6.
- A report that stays FORMED (for example, `make_sample` raises) keeps its group incomplete, and in rollout mode that stalls the whole loop. A settle deadline is a candidate for PR 6.
- `follow_at_once` after an untrained rollout can make the next round start early. This is a pacing trade-off.
- The rollout path does not check `training_mode`; `GeneratedTaskBatches` supports only `auto`.
