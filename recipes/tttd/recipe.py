"""TTT-Discover backend recipe."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

from recipes.tttd.processor import TTTDBatches, TTTDTasks, inference_release_id, tttd_grid
from recipes.tttd.report import TTTDGroupedRolloutReport
from reef.core.reports import ReportBase
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.record2record import GroupDecision, ReportContext, ReportGrouping, ReportStage
from reef.train.algos import StepScheduling
from reef.train.types import PipelineContext

logger = logging.getLogger(__name__)


class TTTDGrouping(ReportGrouping):
    """One TTTD step is one group, and its (group, rollout) coordinates are the slots.

    A step is ready when every one of its ``groups_per_step`` x
    ``rollouts_per_group`` slots is filled. A complete step whose reports span
    releases is discarded wholesale, because mixed-version rewards are not
    comparable.
    """

    def __init__(self, groups_per_step: int, rollouts_per_group: int) -> None:
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.failed_step_versions: dict[int, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        parsed: Any = context.parsed_report
        return parsed.step, (parsed.group, parsed.rollout)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        # 1. The barrier: a step batches only when every one of its
        #    groups_per_step x rollouts_per_group slots is filled.
        expected_count = self.groups_per_step * self.rollouts_per_group
        if len(members) != expected_count:
            return GroupDecision.INCOMPLETE
        # 2. A step whose reports span releases is discarded
        #    wholesale — mixed-version rewards are not comparable.
        versions = {release_id for member in members if (release_id := inference_release_id(member)) is not None}
        if len(versions) <= 1:
            return GroupDecision.READY
        if not isinstance(key, int):
            raise TypeError(f"TTTD step key must be an integer, got {key!r}")
        step = key
        ordered_versions = tuple(sorted(versions))
        self.failed_step_versions[step] = ordered_versions
        logger.error(
            "TTTD step %d failed because its %d reports span releases %s",
            step,
            expected_count,
            list(ordered_versions),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        """Expose mixed-version steps that terminally violated TTTD's invariant."""
        return {
            "failed_steps": [
                {
                    "step": step,
                    "reason": "mixed_release_ids",
                    "release_ids": list(versions),
                }
                for step, versions in sorted(self.failed_step_versions.items())
            ]
        }


def tttd_record_stage(context: PipelineContext) -> ReportStage:
    """TTTD's record stage: one group per step, in step order; a terminal report owns its sources."""
    groups_per_step, rollouts_per_group = tttd_grid(context.config)
    return ReportStage(
        context.scenario,
        context.report_type,
        TTTDGrouping(groups_per_step, rollouts_per_group),
        exclusive_sources=True,
        ordered_groups=True,
    )


@dataclass(frozen=True, kw_only=True)
class TTTDRecipe(WeightTrainingRecipe):
    name: str = "tttd"
    groups_per_step: int = config_field(8, env="REEF_TTTD_GROUPS_PER_STEP")
    rollouts_per_group: int = config_field(64, env="REEF_TTTD_ROLLOUTS_PER_GROUP")

    @property
    def report_type(self) -> type[ReportBase]:
        # One rollout addressed into its exact step grid; the processor adds
        # the config-relative check against the served grid.
        return TTTDGroupedRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="tttd",
            task2batch=TTTDBatches,
            # One optimizer step over the whole grid, as --global-batch-size does for Slime.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def build_record_stage(self, context: PipelineContext) -> ReportStage:
        return tttd_record_stage(context)

    def build_record2task(self, context: PipelineContext) -> TTTDTasks:
        return TTTDTasks(context)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.groups_per_step <= 0:
            raise ValueError("groups_per_step must be positive")
        if self.rollouts_per_group < 2:
            raise ValueError("rollouts_per_group must be at least two")
