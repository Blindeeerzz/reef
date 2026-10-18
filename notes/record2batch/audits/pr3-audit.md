# PR 3 adversarial audit (2026-10-17)

Audit of the PR 3 commit `73a83c57` (fixes amended into `08e22cf0`) by a separate reviewer agent and by the implementing agent.
Reproductions are in `/mnt/data3/reef-r2d-eval/.cache/audit/test_repro_pr3.py` and `.cache/audit-own/test_suspects.py`.
The fixes are folded into the amended PR 3 commit; each fix has a regression test in the suite.

## Fixed

| Finding | Fix | Test |
| --- | --- | --- |
| A task published in two generations crashed the play job (duplicate result) and blocked the later generation's feedback | The store rejects a task that another generation published (`TaskConflict`), so the candidate is a recorded rejection | `test_a_task_that_an_earlier_generation_published_is_rejected_as_a_duplicate` |
| Results were stranded after a restart when no further job was due | `GeneratedTaskTasks` submits a `DeliveryJob` at construction when undelivered results exist | `test_a_restart_sends_results_that_the_generator_did_not_receive` |
| A generator that failed at once burned the round limit in a tight loop and then released the deferred sessions | A deferred round does not count toward `rounds` and does not follow at once; the next round waits `deferred_retry_seconds` (default 600) or for new sessions | `test_a_generator_that_fails_at_once_waits_between_deferred_rounds`, `test_a_generation_job_that_failed_is_deferred_and_retried_after_a_wait` |
| Replayed sessions of a complete generation used up the round limit after a restart | The round limit counts finished generations (`READY` or `SKIPPED`); a replayed round settles `INGESTED` and does not count; new generation numbers come from the store | `test_replayed_sessions_of_a_complete_generation_do_not_use_up_a_round` |
| A task whose play a stop interrupted was refused as a duplicate on restart and counted as a solver failure | The play path deletes and rewrites a `DuplicateTask` before playing | `test_a_task_whose_play_a_stop_interrupted_is_staged_again_and_plays` |
| SPADE re-reported Designer rewards when a delivery stopped part way | SPADE saves each Designer report ID in `open-<n>.json` as it is sent and skips reported proposals | `test_a_delivery_that_stopped_part_way_does_not_report_a_proposal_twice` |
| One generation whose delivery always fails blocked all later deliveries | `deliver_results` continues past a failed generation | `test_a_generation_whose_results_fail_does_not_hold_back_later_ones` |
| SPADE experience could go back to an older generation when generations closed out of order | SPADE updates its experience only from a newer generation | covered by the restart test |
| `status()` iterated the store's generations without its lock while the worker writes them | `FileTaskStore.complete_generation_count()` reads under the lock | reviewed |

## Documented limits (not fixed in PR 3)

- Rounds wait for batches: when played groups cannot fill a batch, no further round starts. The rollout loop in PR 5 replaces this pacing.
- `refill_tasks` counts waiting pool tasks, not tasks in a queued or running play. A play set takes every waiting task (the plan said up to `tasks_per_step`).
- Deferred rounds live in memory; after a restart they rely on the record replay forming the same rounds. Otherwise the incomplete generation stays orphaned in the store.
- Feedback is delivered per generation; there is no per-result ID (a plan deviation, recorded in the PR description).
- AutoEnvScaling saves its cursor after each session, so a retried generation skips the sessions a failed attempt took.
