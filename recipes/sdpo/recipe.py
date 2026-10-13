"""Self-Distillation Policy Optimization recipe."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from recipes.sdpo.processor import SDPOBatches, SDPOTasks, sdpo_grid, sibling_request
from recipes.sdpo.report import SDPOReport
from reef.core.reports import ReportBase
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.record2record import GroupDecision, Record2Record, ReportContext, ReportGrouping, ReportRecord2Record
from reef.train.algos import StepScheduling


class SDPOGrouping(ReportGrouping):
    """One sampling step is one group, and its (group, rollout) coordinates are the slots.

    A step is ready when its grid is complete, it was sampled from one policy
    release, and the rollouts of each question share one request. Otherwise the
    complete step is discarded and listed in ``failed_steps``.
    """

    def __init__(self, groups_per_step: int, rollouts_per_group: int) -> None:
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.failed_steps: dict[int, str] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed = context.parsed_report
        if not isinstance(parsed, SDPOReport):
            raise ValueError("SDPOBatches requires the SDPOReport schema")
        return parsed.step, (parsed.group, parsed.rollout)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) != self.groups_per_step * self.rollouts_per_group:
            return GroupDecision.INCOMPLETE
        if not isinstance(key, int):
            raise TypeError("SDPO step must be an integer")
        releases = {
            None if member.inferences[0].artifact_ref is None else member.inferences[0].artifact_ref.release_id
            for member in members
        }
        reason = ""
        if None in releases or len(releases) != 1:
            reason = "missing_or_mixed_release_ids"
        reports = [(member, parsed) for member in members if isinstance(parsed := member.parsed_report, SDPOReport)]
        for group in range(self.groups_per_step):
            siblings = [sibling_request(member) for member, parsed in reports if parsed.group == group]
            if any(request != siblings[0] for request in siblings):
                reason = "different_requests_in_group"
        if reason:
            self.failed_steps[key] = reason
            return GroupDecision.DISCARD
        return GroupDecision.READY

    def status(self) -> Mapping[str, Any]:
        return {
            "failed_steps": [{"step": step, "reason": reason} for step, reason in sorted(self.failed_steps.items())]
        }


def sdpo_record2record(context: PipelineContext) -> ReportRecord2Record:
    """SDPO's record stage: one group per sampling step, in step order; a terminal report owns its source."""
    groups_per_step, rollouts_per_group = sdpo_grid(context.config)
    return ReportRecord2Record(
        context.scenario,
        context.report_type,
        SDPOGrouping(groups_per_step, rollouts_per_group),
        exclusive_sources=True,
        ordered_groups=True,
    )


@dataclass(frozen=True, kw_only=True)
class SDPORecipe(WeightTrainingRecipe):
    """Self-Distillation Policy Optimization (arXiv:2601.20802) on Reef.

    The served model is its own teacher: it reads the question with a
    successful sibling's response, or the environment's feedback, and its
    next-token distributions over the student's on-policy response become the
    target of a per-token divergence. A step is a complete sampling grid,
    ``groups_per_step`` questions by ``rollouts_per_group`` attempts from one
    policy release; each report carries one rollout's receipt, its score and
    its coordinates, and the grid trains as one optimizer step once complete.

    ``tokenizer_path`` is the served model's tokenizer directory, which renders
    the teacher prompt with the chat template the engine applied. The rendered
    prompt is cut at ``max_teacher_prompt_tokens`` and the whole teacher
    sequence must fit ``max_teacher_tokens``, the trainer's window.
    ``success_reward_threshold`` picks the demonstrating rollouts;
    ``allow_own_success_as_demonstration`` lets a successful rollout read its
    own response (the reference does not); ``remove_thinking_from_demonstration``
    strips ``<think>`` blocks from the demonstration. ``include_environment_feedback``
    adds the report's ``teacher_context`` to the teacher's prompt, by default
    only for rollouts without a demonstration
    (``environment_feedback_only_without_solution``). ``enable_thinking`` is
    the chat template's switch, set as the engine sampled.

    Objective settings such as the divergence, the top-K and the teacher's
    update rate belong to the training backend; for Slime they are
    ``--sdpo-*`` flags in ``training.options``.
    """

    name: str = "sdpo"
    groups_per_step: int = config_field(32)
    rollouts_per_group: int = config_field(8)
    tokenizer_path: str = config_field("")
    max_teacher_prompt_tokens: int = config_field(10240)
    max_teacher_tokens: int = config_field(18432)
    success_reward_threshold: float = config_field(0.5)
    allow_own_success_as_demonstration: bool = config_field(False)
    remove_thinking_from_demonstration: bool = config_field(True)
    include_environment_feedback: bool = config_field(False)
    environment_feedback_only_without_solution: bool = config_field(True)
    enable_thinking: bool = config_field(False)

    @property
    def report_type(self) -> type[ReportBase]:
        return SDPOReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="sdpo",
            task2batch=SDPOBatches,
            # The grid is the optimizer step, however many rollouts it holds.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def build_record2record(self, context: PipelineContext) -> ReportRecord2Record:
        return sdpo_record2record(context)

    def build_record2task(self, context: PipelineContext, record2record: Record2Record) -> SDPOTasks:
        return SDPOTasks(context)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0 or self.rollouts_per_group < 2:
            raise ValueError("SDPO needs positive groups_per_step and at least two rollouts_per_group")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the teacher prompt")
        if not 0 < self.max_teacher_prompt_tokens <= self.max_teacher_tokens:
            raise ValueError("SDPO needs 0 < max_teacher_prompt_tokens <= max_teacher_tokens")
        if not math.isfinite(self.success_reward_threshold):
            raise ValueError("success_reward_threshold must be finite")

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError(
                "SDPO objective options are backend-owned; configure the Slime implementation with training.options"
            )
