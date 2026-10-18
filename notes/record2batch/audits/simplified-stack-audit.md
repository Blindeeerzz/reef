# Audit of the simplified record2batch stack (2026-10-18)

A reviewer agent audited `review/record2batch-stack` at `b722f4f8` after the design-review simplification (PR 3 without runs or deferral, PR 4 without unstaging, PR 5 rollout-only with report-ID settling).
Reproductions: `/mnt/data3/reef-r2d-eval/.cache/audit7/test_audit7.py`.
Fixes: PR 3 amended to `fa37cf32`, PR 4 rebased to `185116c4`, PR 5 amended to `0e60f0a5`.

## Fixed

| Finding | Fix | Test |
| --- | --- | --- |
| The pipeline stopped for good when the task pool emptied before `batches_per_round` batches trained (the shipped examples' pacing) | `RoundRecord2Record.round_due` is also true when the task pool is empty and no rollout runs | `test_rounds_continue_when_the_task_pool_drains_before_batches_per_round` |
| A failing measurement arm discarded the training arm's report IDs and result (saved an infrastructure failure; the group settled empty) | Training reports are taken before the measurement arms; a measurement arm's `GeneratorError` is logged and the arm measures nothing | `test_a_failed_measurement_arm_keeps_the_training_reports_and_result` |
| A result that failed to save dropped the task's training reports and held back the generation's feedback until a restart, forever if the task trained | `play_and_record` keeps the reports when the save fails and queues the result; the generation worker saves queued results before each delivery. On restart, a trained task without a result gets an infrastructure-failure result | `test_a_task_whose_result_cannot_be_saved_still_trains_and_the_next_job_saves_it`, `test_a_generation_whose_task_result_failed_to_save_once_still_gets_its_feedback`, `test_a_task_that_trained_before_its_result_was_saved_gets_a_lost_result_after_a_restart` |
| A skipped generation left published tasks that never rolled out (and counted as available) | A description-only round resumes the newest incomplete description-only generation (PR 3); store counts show tasks of unfinished generations as `waiting/train` | `test_a_skipped_generation_is_finished_by_the_next_round`, store test |
| Reports without an episode ID made a group from one episode's segments (regression of the first PR 5 audit fix) | `RolloutGrouping.is_training_report` requires an episode ID; such reports end and count toward their group | `test_reports_without_an_episode_id_never_train_but_still_settle_their_group` |
| Docs said the recipe trains without rounds, described deferred rounds, and "tasks that have no result" | Recipe docstring, configuration reference and pipeline guide updated; example deployments' batch comment updated | docs |

## Known limits (unchanged)

- Tasks are marked trained at acknowledgement, before the commit is durable; a crash in between loses those marks.
- A task refused at staging never trains, so every restart pools it and refuses it again.
- There is no deadline before a rollout's plays end; a stuck play holds back the batch until its job times out.
- A round whose generation ran out of attempts counts toward `rounds` in memory; a restart counts complete generations only.
