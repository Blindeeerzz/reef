# Record2task experiments and controlled studies

Experiments for the recorded-interaction to task, rollout and model-training loop.
Nightly operation is an example schedule; the studies apply to manually or automatically triggered runs.
Each study records its research question, experimental conditions, controls and validation before execution.

Updated 2026-10-18 to match the pipeline as built (PR stack #836–#839 plus local PR 5), after a design review that removed run scheduling, deferral and durable per-use records.

## Setting and data

- **Sources:** SWE-chat sessions are the recorded interactions that tasks are generated from.
- **Evaluation:** SWE-Together is the external held-out benchmark. The internal train/eval split (`eval_fraction`, may be 0) is only a leakage guard; eval tasks are never played or trained.
- **Contamination:** check overlap between SWE-chat sources and SWE-Together (shared repositories or issues) in each experiment's setup, not in the pipeline.
- **Billing:** model calls go through Reef on the ChatGPT plan (no OpenAI API credits); training runs on Tinker. Each run's billing source and budget need approval before it starts.

## Pipeline as built: the controls the studies vary

| Control | Setting | What it does |
| --- | --- | --- |
| Rollout batch size | `pipeline.task2batch.config.tasks_per_step` | Saved tasks per rollout batch; one optimizer update per batch, on all of the batch's surviving task groups. |
| Plays per task | `pipeline.record2record.config.plays_per_task` | Training episodes per task in a rollout; the task group's size. |
| Execution capacity | `pipeline.task2batch.config.episode_capacity` | Episodes queued or running at once; a task with more plays plays in waves. Independent of batch size. |
| Settle deadline | `pipeline.task2batch.config.settle_timeout_seconds` | How long a batch waits for missing episode reports before it trains on what arrived. |
| Staleness | recipe `max_staleness` | Groups whose samples lag the serving version by more are dropped before the batch forms. |
| Generation pacing | `rounds`, `batches_per_round`, `refill_tasks`, `source_records_per_round` | When a generation round starts, how many sessions it reads, and how many saved tasks may wait. |
| Generation budget | `generation_attempts`, `generation_deadline_seconds` | Infrastructure retries and the deadline of one round; a round that runs out settles as making no task. |
| Train/eval split | `eval_fraction`, `split_seed` | Fixed per store; eval tasks are saved but never played. |

Each task trains at most once: a task is marked trained when its group trains. On a restart, tasks that did not train go back to the task pool and roll out again. The generator receives each task's first play result.

## Experiment 1 Optimizer batch size and training stability

Status: planned.

### Research question

How does the number of task groups per optimizer update affect learning stability, held-out improvement and efficiency?
Can small execution queues supply sufficiently large training batches through multiple rollout waves?

### Experiment

Compare three batch sizes, initially **2, 8 and 32 task groups per update**, with a fixed number of training plays per task.
These are candidate study settings, subject to a feasibility pilot.

Run each condition from the same model checkpoint with multiple seeds and an equal total training-rollout budget.
Collect each batch under fixed sampling weights, update once, then synchronize before collecting the next batch.

Larger batches produce fewer optimizer updates under the same rollout budget.
This first study measures that combined effect.

### Controlled factors

- Same frozen task pool and source-disjoint train/eval split; disable adaptive task generation (`rounds` fixed, pool generated in advance).
- Batch size is `tasks_per_step`; hold `episode_capacity` fixed across conditions so the rollout waves differ only by batch size.
- Same model, objective, learning rate, sampling settings and plays per task.
- Same queue capacity, concurrency, timeouts, retry policy and reward verifiers.
- Same task-sampling policy and handling of constant-reward groups.
- Same evaluation tasks and evaluation budget across conditions.

### Validation

- **Learning:** held-out reward and task success rate versus the starting checkpoint, with uncertainty across tasks and seeds.
- **Stability:** variation between runs, reward regressions, and available update metrics such as KL divergence and gradient norms.
- **Effective signal:** nonconstant task groups, usable trajectories and training tokens per update; skipped batches and infrastructure failures.
- **Efficiency:** rollout cost, wall-clock time and completed updates.
- **Pipeline correctness:** rollout waves preserve group membership and policy versions; measurement/eval episodes never train; records are consumed once.

### Decision supported

Choose a batch-size range and minimum acceptable batch for subsequent experiments.
Queue capacity remains a separate execution setting.
No condition is considered successful solely because training loss decreases.

## Experiment 2 Fixed serving-model data versus refreshed candidate rollouts

Status: planned.

### Research question

Can a training-only candidate improve from rollouts generated by the fixed deployed model, without serving intermediate candidate checkpoints?
When does refreshing the rollout model justify its inference capacity, synchronization cost and operational disruption?

### Deployment constraint and terminology

The deployed inference engine may be the only available sampler; the training backend may support optimization and forward scoring without supporting candidate generation.
Keep that engine on the accepted release, collect and score a rollout dataset, train the candidate separately, then evaluate and promote only if it improves.
A fixed dataset reused during optimization is the offline setting; continued collection from unchanged serving weights is off-policy relative to an evolving candidate.
SFT, reward-based RL and distillation are separate objectives, not interchangeable names for that setting.
Standard OPD samples the current student and obtains teacher token scores; replaying older serving-model samples requires an explicitly off-policy distillation method.
See [the OPD method](https://thinkingmachines.ai/blog/on-policy-distillation/) and [offline RL definitions](https://arxiv.org/abs/2005.01643).

### Experiments

First compare feasible learning methods on the same frozen, scored serving-model rollout pool:

- **No-update control:** evaluate the accepted serving checkpoint.
- **Filtered SFT:** train on successful, validated trajectories, with filtering and token masks fixed in advance.
- **Offline reward-based training:** use an objective explicitly designed or validated for the recorded behavior policy and dataset reuse; retain behavior log-probabilities and policy identity.
- **Off-policy distillation, when feasible:** use a specified teacher-scoring objective and compatible token scores; record the teacher identity and scoring cost separately from the rollout model.

Then isolate rollout refresh frequency with the same reward-based objective that supports the compared policy lags: no refresh within a run, refresh every configured number of updates, and refresh after each update.
Hold the task pool fixed rather than requiring identical trajectories from different sampling policies.
Add standard OPD as a separate current-student-sampling comparison only when candidate inference and a meaningful teacher signal are available.
Imitating the initial model's unfiltered samples is not assumed to provide an improvement signal.

Refreshed comparisons require a temporary sampler, controlled engine reload or another demonstrated candidate-inference path.
Where the deployment cannot provide one, report that limitation and compare feasible methods without treating an unavailable arm as a learning failure.

### Controlled factors

- Same initial candidate checkpoint, frozen task pool, source-disjoint held-out evaluation, task verifiers and sampling configuration.
- Same batch/group configuration within the refresh comparison; fix it independently of experiment 1's sweep.
- Same objective, optimizer, data reuse and correction rules within the refresh comparison; method comparisons explicitly vary the objective.
- Paired seeds and a matched total compute/cost budget, counting collection, teacher scoring, training, checkpoint transfer and evaluation; report episodes, tokens and optimizer updates separately.
- Fixed success filtering for SFT and fixed teacher/context for distillation; separate serving-policy, candidate-policy and teacher identities.
- Explicit policy-lag handling; do not bypass the current staleness checks merely to make offline replay run.

### Validation

- **Quality:** held-out reward and success rate versus the accepted release, across seeds and successive runs, plus general-capability regression checks.
- **Stability:** learning curves, divergence/update metrics where available, degradation with repeated dataset reuse and increasing policy lag.
- **Feasibility and efficiency:** improvement per total cost, run completion time, peak resources, sampler availability and any serving interruption.
- **Correctness:** behavior tokens/log-probabilities and policy versions remain linked; held-out data never trains; offline eligibility is explicit; only evaluated candidates are promoted.

### Decision supported

Choose the initial supported training objective and whether sampling stays fixed during a run or refreshes during learning.
Candidate inference is an experimental capability, not an assumed prerequisite for the refactor.
Experiment 1's refresh-after-update protocol is a study condition, not a settled deployment requirement.

## Experiment 3 Generation and repair budgets

Status: planned.

### Research question

How much generation and repair work should record2task spend on one source before moving on?
When do additional attempts produce enough useful training tasks to justify their latency, cost and reduced coverage of other sources?

### Experiments

- **Baseline:** generate and validate once at the pipeline boundary; reject invalid tasks and continue, preserving the configured generator's authoring mode.
- **Bounded repair:** return validation feedback to the author with candidate limits of 1 and 3 repair attempts; finalize values after a feasibility pilot.
- Compare attempt limits under the same per-source deadline first, then vary the deadline at a fixed repair limit to distinguish extra attempts from extra elapsed time.
- Compare allocating the remaining run budget to repairing failed tasks versus attempting new sources.

Enforce a per-round deadline (`generation_deadline_seconds`) and a total budget over rounds (`rounds`, `source_records_per_round`), including generation, validation, oracle execution and repair.
Time or budget exhaustion counts as an outcome; exclude neither slow sources nor unsuccessful attempts from the reported cost.
Keep infrastructure retries (`generation_attempts`) separate from task repair and record both.

Implementation gap: the pipeline does not repair tasks; repair runs inside a generator (for example, an authoring agent that retries its own validation). The bounded-repair arms need that generator mode, or a repair step added before this study. Run-level scheduling (runs with their own round and time limits) was removed; allocate the budget across sources with the round settings instead.

### Controlled factors

- Same fixed source snapshot, paired source order/seeds, author model and mode, generation instructions, validation criteria and execution environment.
- Same total generation/validation budget and concurrency; candidate per-source limits are the treatment.
- Same fixed solver and rollout protocol for assessing task usefulness, with assessment cost reported separately.
- Same downstream training configuration and held-out evaluation when comparing learning benefit; never weaken a verifier to increase the acceptance rate.

### Validation

- **Yield and coverage:** valid unique tasks, sources attempted, repair recovery rate, duplicates and rejection reasons.
- **Latency:** median and tail source-processing time, time to the first complete training batch, and time rollout waits for tasks.
- **Efficiency:** valid tasks and useful solver groups per unit cost/time, marginal benefit of each repair, and budget spent on unsuccessful attempts.
- **Task quality and learning:** independent validity checks, solver difficulty, nonconstant reward groups and downstream held-out improvement.
- **Correctness:** deadlines cancel owned work, late results cannot mutate closed jobs, and persisted attempt/budget accounting survives restart without duplicate admission.

### Decision supported

Keep rejection without pipeline repair as the initial implementation.
Choose later repair limits, per-source deadlines and generation budgets from the quality/latency/cost tradeoff, with deployment overrides for different latency requirements.

## Experiment 4 Task reuse with fresh rollouts

Status: planned.

### Research question

Does selecting validated tasks again in later runs improve learning efficiency enough to justify reduced task diversity and possible overfitting?
How many training uses should a task receive before allocating work to new tasks instead?

### Experiments

- **Baseline:** one training use per task; later runs use previously unused tasks.
- **Bounded reuse:** allow candidate caps of 2 and 4 uses per stable task identity across runs, generating fresh rollout groups for every use.
- Compare new-only selection against mixtures of new and previously used tasks, with mixture settings chosen after a feasibility pilot.

Implementation gap: the pipeline keeps one trained mark per task, which supports the baseline. The reuse caps need a per-task count of trained uses in the task store (a small extension) before this study runs.

Use an equal total training-rollout budget for the primary comparison and report generation/validation savings separately.
Also compare held-out improvement per total cost, including task authoring, so any advantage from saved generation work remains visible.
Do not replay old trajectories in this study; that is experiment 2's separate treatment.

### Controlled factors

- Same source snapshots, initial checkpoint, generation/validation method and source-disjoint evaluation set.
- Same rollout refresh schedule, learning objective, batch size, plays per use, sampling settings, concurrency and retry rules.
- Paired seeds and the same selection rules apart from reuse eligibility, cap and mixture.
- Same evaluation protocol across successive runs; retain task identities and train/eval splits throughout.

### Validation

- **Learning:** held-out reward and success rate across runs, compared with performance on reused training tasks.
- **Diversity:** unique tasks and sources represented, repeated-task concentration and new-task coverage.
- **Signal and efficiency:** nonconstant groups, useful training tokens, generation/validation savings and improvement per total cost.
- **Correctness:** every reuse gets distinct batch and task-use identities and fresh policy-tagged trajectories; restart keeps the trained counts. A crash between a batch's acknowledgement and its commit can lose that batch's trained marks or replay its tasks; report how often that happens.

### Decision supported

Keep one training use per task initially.
Choose any later reuse cap and new-task mixture from generalization and cost results rather than improvements on repeated training tasks alone.
