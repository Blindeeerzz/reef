# PR 6 adversarial audit (2026-10-18)

A separate reviewer agent audited the PR 6 commit `6548e08b`. The fixes are amended into `9a214097` on `feat/record2batch-dispositions` (`reef-record2batch-pr6`).
Reproductions are in `/mnt/data3/reef-r2d-eval/.cache/audit6/test_audit6_recovery.py`. Two of them still fail after the fixes, but only because they assert the old state (`trained` right after acknowledgement; `cancelled` uses after a refused submit); the behaviour they check is fixed.

## Fixed

| Finding | Fix | Test |
| --- | --- | --- |
| The store marked uses `trained` at acknowledgement, which `prepare_commit` calls before the commit is durable; recovery trusted it, so a crash or a failed commit lost the batch (released untrained, tasks never rolled out again). The plan's kill test failed | The pipeline no longer ends uses at acknowledgement. `Trainer.commit` (after the scenario commit settles) and `Trainer.reject_pending` (after the receipt is written) call `TrainingPipeline.settle_batch`, which hands the outcome to the rollout stage. Until then the uses stay `training`, and a restart re-forms the batch | `test_a_stop_after_acknowledgement_but_before_the_commit_re_forms_the_batch` |
| A failed commit, `retry_pending`, then a stale drop left the uses `trained` while the receipt said stale | Same fix: the stale receipt settles the batch as `stale_rejected` | `test_a_batch_whose_commit_failed_and_that_was_then_dropped_stale_ends_stale` |
| A refused submit marked the uses `cancelled` (an attempt) while the tasks stayed pooled; tasks could retire in the pool without a result, every later look raised, and the generation's feedback never completed | A refused submit abandons the uses it began (`FileTaskStore.abandon_uses`, no attempt). `begin_uses` checks every task before saving any. Every retired task without a result gets an infrastructure-failure result whenever tasks return | `test_a_refused_submit_counts_no_attempt_and_the_tasks_roll_out_once_the_worker_takes_jobs` |
| A rollout pipeline without a task store (`rounds: 0`) crashed on its first look | `recover` returns None without a store | `test_a_rollout_pipeline_without_rounds_has_no_task_store_and_opens` |

## Risks addressed

- **Pending commits:** a commit held for review now counts as proof that its named uses trained, when it consumed their reports. Its reports never replay, so the batch could not re-form. Without this, deferring the store write would have made such a crash stop every restart. This deviates from the plan's "verified, non-pending" wording; the consumed-reports check remains.
- **Outcome order:** if a batch has both a commit and a receipt outcome, the commit wins.
- **Partial settle on raise:** recovery checks the unresolved-consumption error for every use before it changes any use.

## Known limits

- A resumed batch re-forms only if every reported episode replays. A report that `Trainer.ingest_retained` skips (a missing or consumed reference) never completes its group. This predates PR 6.
- The D4 configuration check (refuse a rollout model that is not the scenario's serving runtime) is not implemented. `served_model` is only a model name, and rollout traffic always goes through the scenario ingress, so a trajectory version outside the served list is already stale under #819.
- PR 5 carry-overs still open: a settle deadline for groups whose reports never settle, and settling groups by report IDs instead of episode counts.

## Checked and fine

- Nothing calls `pipeline.ready()` before the factory's `recover_rollouts`, so the implicit recover in `ready()` never locks in empty evidence in a real scenario.
- The report IDs in the metadata (`formed_set_ids`) are report record IDs, so the factory's consumed-IDs check is valid.
- Use IDs do not collide after reopening. Default hook no-ops leave non-generated recipes unchanged.
