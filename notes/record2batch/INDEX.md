# record2batch review branch

This branch (`review/record2batch-stack`) is for review only; it is not meant to become a pull request.
It holds the whole record2batch stack, PR 1 to PR 5, as one commit each on top of `main` at `5f0f1bfb`, plus this `notes/` directory.

| PR | Commit subject | Upstream |
| --- | --- | --- |
| 1 | feat(generated-tasks): save tasks with their split before rollout; refactor(record2task): keep the task store in record2task | #836 (two commits) |
| 2 | feat(pipeline): settle slow record-to-task work through pending results | #837 |
| 3 | feat(generated-tasks): generate without play, play from a task pool, and deliver feedback late | #838 |
| 4 | feat(record2task): play saved tasks through an episode runner with repeatable staging | #839 |
| 5 | feat(pipeline): roll saved tasks out as batches and train on their labelled episodes | local only |

The open PRs #837 to #839 still show their earlier versions; this branch has the simplified ones (see `stack-design-review.md`).
See each PR's change with `git show <commit>`, or the whole stack with `git diff 5f0f1bfb..HEAD -- . ':!notes'`.

## Contents

- `plans/experiment-plan.md`: the four planned studies (batch size, fixed vs refreshed sampler, generation budgets, task reuse), updated to the pipeline as built.
- `plans/task-decomposition.md`: the original PR plan (written before the simplification; PR 5 to PR 7 changed since).
- `plans/codebase-design.md`: the record2task codebase design notes.
- `stack-design-review.md`: an independent design review of PRs 1 to 6 and what was cut after it.
- `descriptions/pr1.md` to `pr5.md`: PR descriptions, updated to the simplified code.
- `audits/`: adversarial audits of PRs 1, 3, 4 and 5, and `simplified-stack-audit.md` of the stack after the simplification (all its findings fixed). The earlier audits describe the code at the time of each audit; some findings refer to code removed later.
- `shelved-pr6/`: the shelved PR 6 (durable task uses and commit-based recovery), with its description, audit and patch. It is not on this branch's code.

## What changed after the design review

- PR 2: removed the class-definition check for the removed `poll` hooks.
- PR 3: removed runs, deferred rounds and `TaskStatus.DEFERRED`; a generation that still fails after its attempts settles as skipped.
- PR 4: `EpisodeRunner` is a plain class; no unstaging.
- PR 5: the rollout loop replaces play sets; groups settle by the report IDs the plays returned, with a settle deadline; trained tasks are marked in the store and untrained ones return after a restart; single-implementation ABCs removed.
- PR 6: shelved.
- Not yet done: PR 1's single-implementation `TaskStore` ABC and fsync chain (a maintainer is pruning #836 directly).
