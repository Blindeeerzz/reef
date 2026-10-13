# GPU handoff: SPADE vs AutoEnvScaling as RL training data

You are the agent on the GPU host. This file is your plan. Read all of it before you start. Stop and report to the user at each **Checkpoint**.

The data arrives in parts while the user's other agent still generates tasks. So you start with the work that needs no training data: Slime support (Step 1), then the evaluation sets and the base models' scores (Step 2).

## Goal

Find out if tasks from **AutoEnvScaling** (AES) make better RL training data for agentic coding than tasks from **SPADE** (its prompt Designer). Both methods build tasks from the same recorded coding-agent sessions (SWE-chat). Train the same model with the same RL budget on an equal number of tasks from each method. Then compare held-out scores.

What we know already is task quality, not training value:
- AES's agent mode writes 100% valid tasks; SPADE's prompt Designer writes 30%.
- With AES's checks, 80% of admitted tasks are fair, against 32% without them.
- Qwen3.5-9B solves 40% of its plays on checked tasks, against 13% on unchecked tasks.

## Arms

| Arm | Method | Tasks |
|---|---|---|
| A0 | Base model, no training | – (evaluation only) |
| A1 | **AES**: agent mode with the behaviour rule and the review | from the same SWE-chat sessions |
| A2 | **SPADE**: the prompt Designer, up to 3 tries per session | from the same SWE-chat sessions |

Each arm uses its own method's admission: the oracle passes, the no-op fails, and for A1 also the review. The comparison is between the methods as they are, so it includes their difference in task quality. A no-checks ablation comes later. It is not in this plan.

**Tasks per arm:** N = 100 for the main comparison, then N = 300 for the scaling point.
- 100 decides the main question. 3 epochs at 8 tasks per step is 38 steps and 2,400 episodes. About half of A1's tasks are in 9B's reach (it solved 1 to 3 of 4 plays on 15 of 31 measured tasks), so about half of each step's groups have a non-zero advantage.
- 300 tests if AES's lead grows with more data.

## The host

- 8× B200, **full fine-tuning**.
- **No Docker.** Slime runs from the user's Apptainer image.
- **Harbor tasks run in E2B.** The agent (terminus-2) runs on the host and sends its commands to the sandbox, so the sandbox does not need to reach the model server.

### E2B and task environments (read this before you play any task)

Harbor (0.20 to 0.23) gives E2B a task's Dockerfile as written. E2B's template builder does not apply the base image's `ENV`, the Dockerfile's `ENV` and `ARG`, or a `WORKDIR` with `${VAR}`. The verifier, the solution and the agent's shell also run without that environment. Many tasks fail for this reason only.

This branch has the fix: `reef.record2task.generator.e2b_environment.E2BTaskEnvironment`. Its `prepare(task_path)` writes a copy of the task that states the environment Docker would apply. It reads the base image's `ENV` from the image registry. The user's generation host runs every check and play through it.
- Prepare **every** task before you play it: training tasks, SWE-Together and SWE-bench Verified.
- Keep each copy's directory name the same as the original, so the manifests still apply.
- Play the copies with Harbor's E2B environment: `--environment e2b` on the task player, `-e e2b` on `harbor run`.
- A task that `prepare` refuses (`UnrunnableTask`, for example a base image E2B cannot pull) is left out. List it.

## Inputs

**Code:** this branch, `r2d/training`. It is Reef `main` (9336c39d) + PRs #789, #790 and #791 + the E2B task environment. Set up the environment as `AGENTS.md` says. Inside the Slime image, install with `--no-deps` as it says, and add the `e2b` extra.

**Data**, in parts. The user copies each archive to you. Check each archive's `SHA256SUMS` first.
1. **Now:** `r2d-handoff-1.tar.gz`
   - `dryrun/`: 8 A1 tasks for the Slime dry run, with their 9B rewards from the user's measurements;
   - `eval/swe-together/`: 47 strict and 15 capped tasks, with our verifier fixes already applied, and `oracle.jsonl`.
2. **When generation finishes:** `r2d-handoff-2.tar.gz`. It holds `train/A1-aes/` and `train/A2-spade/`: Harbor task folders, `manifest.json` (the train split, at least 100 tasks each), `generation.jsonl` (per task: the source session, the check results, the `gpt-6-luna` rewards, and the 9B rewards where measured) and `DATASHEET.md`.
3. **Later:** `r2d-handoff-3.tar.gz`, with 300 tasks per arm for the scaling point.

SWE-bench Verified is not in an archive. Use Harbor's registry dataset `swebench-verified@1.0`, all 500 tasks.

**Credentials:** the user gives you environment files, mode 600: `E2B_API_KEY`, and `HF_TOKEN` if a model download needs it.
- Never print or log a credential.
- Never use OpenAI or Anthropic API keys. No step here needs a hosted model.
- Ask the user for the E2B account's concurrent sandbox limit. Do not exceed it.

## Step 1: Slime support for the generated-task objective (start here)

**Problem:** the recipe `reef.recipe.generated_tasks:GeneratedTaskRecipe` trains with the objective `task_group` (`reef/recipe/generated_tasks/objective.py`). Its `loss_family` is `"importance_sampling"`. Only the Tinker backend implements that family. No Slime algorithm is registered for it, so the recipe cannot train on GPUs.

**What the objective gives:** one advantage per episode. It is the episode's reward, centered and scaled within its task's group. A constant group gets 0. A step where every group is constant is skipped. Tinker's loss puts this advantage on every response token, with the rollout log-probabilities as the reference.

**Do:**
1. Read the contract and examples:
   - `reef/train/slime_backend/algorithm.py`: the `SlimeAlgorithm` class, its attributes (`loss_type`, `requires_rollout_logprobs`, `advantages`, `allows_slime_advantage_computation`, `required_objective_hooks`), and the `@register_loss_family` decorator.
   - `reef/train/slime_backend/loss_families.py` and `reef/train/algos/registry.py` (`register_loss_family_ref`): how a family name finds its module.
   - Examples: `recipes/tttd/slime/__init__.py` (small), `recipes/openclawrl/slime/__init__.py` (external advantages with a GRPO branch), and `reef/train/slime_backend/distill/` (an algorithm inside `reef/`, not in `recipes/`).
2. Add a Slime algorithm for `importance_sampling` and register it. The recipe is in `reef/`, so the algorithm must be in `reef/` too: `reef/` must not import `recipes/`. Choose the module by the codebase map in `AGENTS.md`.
   - Use the advantages the objective gives (`advantages = "required"`). Do not let Slime compute its own.
   - Use the rollout log-probabilities as the reference (`requires_rollout_logprobs = True`).
   - Prefer Slime's own policy loss with these advantages to a custom loss.
   - Use PPO-style clipping (`eps_clip` 0.2, `eps_clip_high` 0.28) and no KL term. Full fine-tuning is less stable than Tinker's LoRA. Write this reason in a short comment.
3. Follow `AGENTS.md` strictly: its Python design rules, its naming rules and its writing style.
4. Add contract tests next to the existing ones (`tests/reef_service/test_slime_algorithm_contract.py`).
5. **Dry run on the GPUs:** Qwen3.5-4B, the Slime backend, the recipe with `rounds: 0`, 2 steps on 4 of the `dryrun/` tasks × 4 plays in E2B. Pick tasks whose 9B rewards are mixed, so the groups are not constant. The run must show:
   - a finite loss and a non-zero gradient norm;
   - a weight update, and the new weights served for the next plays;
   - every multi-turn episode assembled into token rows, with none dropped.

**Checkpoint 1:** report the diff, the test results and a dry-run log excerpt. The user decides if this becomes an upstream PR.

## Step 2: Evaluation sets and the base models (no training data needed)

1. Prepare SWE-Together (47 strict, 15 capped) and all 500 SWE-bench Verified tasks for E2B.
2. Run each task's oracle once in E2B. This also builds and caches each task's E2B template. A task whose oracle fails is excluded from every model's score alike. Write the list with the reason for each task.
3. Evaluate A0, the base 9B and 4B, as Step 6 describes.

**Checkpoint 2:** report the oracle pass counts, the excluded tasks, the template build time, the E2B cost so far, and the A0 scores. If many SWE-bench Verified tasks fail in E2B, stop. The user decides about another remote sandbox.

## Step 3: Configuration (the same for every arm)

- **Models:** `Qwen/Qwen3.5-9B`, then `Qwen/Qwen3.5-4B`. Full fine-tuning. Thinking off (`chat_template_kwargs.enable_thinking: false`). Context 65536.
- **Recipe:** `reef.recipe.generated_tasks:GeneratedTaskRecipe` with `pipeline.record2task.config.rounds: 0`. Nothing is generated during training.
- **How episodes reach training:** follow "Train from the pool" in `recipes/beta/autoenvscaling/README.md`. The task player (`python -m reef.harness.client.tasks`) plays the manifest's train side and reports each episode with `--label arm=plain`. A task's group trains when it has `plays_per_task` episodes.
- **Grouping:** `pipeline.record2record.config.plays_per_task: 8` and `pipeline.task2batch.config.tasks_per_step: 8`. So a step has 64 episodes.
- **Epochs:** write a small driver that plays each task 8 times per epoch, for 3 epochs. Shuffle the task order per epoch with seed 0. Use the same seed for both arms.
- **Plays in flight:** the task player plays while the model trains. Keep at most 2 steps of plays in flight (128 episodes, fewer if the E2B limit is lower). Then an episode is at most about 2 weight versions old. The importance-sampling ratio corrects for that.
- **Agent** (the same JSON for training plays and evaluation):
  - terminus-2 through `import_path: reef.record2task.generator.terminus:VerifiedTerminus2`;
  - `max_tokens` 4096, `enable_summarize: false`, `model_info.max_input_tokens` 60000, `record_terminal_session: false`;
  - in `llm_kwargs`: `"api_key": "{api_key}"` and the `enable_thinking: false` extra body. Without the API key, the plays fail with 401.
- **Optimizer:** learning rate 1e-6, 3 warm-up steps, then constant. Gradient clipping 1.0. The same for both arms. If the pilot is unstable, change the setting once, for both arms, and say why.
- **Budget:** N = 100 → 38 steps. N = 300 → 113 steps. Save a checkpoint every 5 steps and at the end.
- **Seeds:** 1 per run. Add a second seed for the 9B main runs if time allows.
- **Slime topology:** choose it from Slime's guidance for 8× B200: colocated, or GPUs split between rollout and training. Report the choice and the measured throughput.

## Step 4: Pilot (when `r2d-handoff-2` arrives)

1. Prepare the A1 and A2 tasks for E2B and run each oracle once, as in Step 2.
2. Run 9B on A1 for 10 steps.

**Checkpoint 3:** report:
- the time per step, and the episodes per hour;
- the share of non-constant groups per step (we expect about 40–60%);
- dropped or unassembled episodes (we expect 0);
- E2B errors, and the E2B cost per step;
- the loss and gradient-norm curves;
- a 10-task SWE-Together evaluation at step 0 and at step 10;
- your estimate of the time for each run in Step 5.

## Step 5: Runs

| Run | Model | Arm | N | Steps |
|---|---|---|---|---|
| 1 | 9B | A1 AES | 100 | 38 |
| 2 | 9B | A2 SPADE | 100 | 38 |
| 3 | 4B | A1 AES | 100 | 38 |
| 4 | 4B | A2 SPADE | 100 | 38 |
| 5 | 9B | A1 AES | 300 | 113 |
| 6 | 9B | A2 SPADE | 300 | 113 |

- Sample N tasks at random, with seed 0, from each arm's manifest. Write the chosen task names to `results/`.
- Order: runs 1 and 2, then their evaluations (**Checkpoint 4**), then runs 3 and 4, then runs 5 and 6 when `r2d-handoff-3` arrives.

## Step 6: Evaluation

- Use the same agent JSON and settings as the training plays, in E2B. Serve each model through an OpenAI-compatible endpoint.
- **SWE-Together:** the 47 strict tasks, 3 plays each. Report the mean reward and pass@1. Report the 15 capped tasks separately. For reference, the base 9B scored **0.191** mean reward on the strict set (served by Tinker).
- **SWE-bench Verified:** all 500 tasks (`swebench-verified@1.0`) less the oracle failures. 1 play each. Report pass@1.
- **Checkpoints to evaluate:** the base model, the final checkpoint, and the best checkpoint on SWE-Together. Choose by SWE-Together only. Report SWE-bench Verified for the chosen one.
- **Statistics:** the paired difference per task (A1 − A2, and each arm − A0), with a bootstrap 95% confidence interval over tasks. Also a paired sign test.
- **Contamination:** checked on 2026-10-13. The training repositories share none with SWE-Together or SWE-bench Verified. Do not add other training sources without a new check.

## Reporting and rules

- **Logs:** run every long job with its output through `tee <logfile>`. Give the user the `tail -f` command. Call a virtual environment's Python binary directly. Do not use `conda run`.
- **Results:** at each checkpoint, write one table to `results/` as markdown and JSON. Columns: run, model, arm, N, steps, SWE-Together mean and pass@1, SWE-bench Verified pass@1, and confidence intervals.
- **Git:** commit locally on `r2d/training`. Do not push, open PRs, or post anywhere without the user's OK.
- **Cleanup:** delete only the E2B sandboxes and templates that you made. Do a dry run of the deletion first.
- **When unsure,** stop at the next checkpoint and ask the user.
