# Record2batch stack design review (2026-10-18)

An independent reviewer agent (Fable 5.1) read PRs 1–6 (diff 5f0f1bfb..pr6, the PR descriptions) and the PR 7 plan section, with no prior context, and judged the design from an ML-framework perspective.

## Verdict per PR

| PR | Verdict |
| --- | --- |
| 1 task store | Keep, simplify: one-implementation `TaskStore` ABC, fsync of every file and directory |
| 2 pending results | Keep; drop the `__init_subclass__` check |
| 3 generation without play | Split or shrink; drop runs, deferred-round memory and `DEFERRED` |
| 4 episode runner | Merge into PR 5; drop the `EpisodeRunner` ABC and refcounted `unstage` |
| 5 rollout loop | Keep, de-generify (one-implementation `RolloutRecord2Record`, `RolloutTask2Batch`, `PoolFilter`) |
| 6 dispositions | Drop or defer: durable per-use state to avoid one re-rollout after a crash; commit metrics used as a recovery log |

## Required before merge, by value

1. Shelve PR 6; on restart, re-pool tasks without a committed group.
2. Cut runs / deferral / `DEFERRED` from PR 3; fix or remove `TaskResult.deferred()`. Verified: base `Record2Record.settle` raises `TypeError` for `DEFERRED` (`reef/record2record/base.py:120-126`).
3. Collapse the single-implementation ABCs; keep `Pool[ItemT]`.
4. Land PR 7's deletion with PR 5, so `GeneratedTaskBatches` stops setting flags on sibling stages and the Optional guards go.
5. Add a settle deadline for rollout groups.
6. Remove the `__init_subclass__` check and the directory-fsync chain.
7. Merge PR 4 into PR 5.

ML-framework note: verl, OpenRLHF and slime keep no durable per-prompt use state; they recover by checkpoint plus resampling, and implement curricula as stateless filters. The only durable fact needed is "a commit consumed this report", which Reef already records.

Context the reviewer did not have: runs and the refill rule were plan decisions D7/D8; the two pools and generic contracts were option C′ chosen with the user; commit-evidence recovery was plan case 5. The decision on what to cut is pending with the user.
