"""The CORAL weight-training recipe: what a deployment names in ``serve.yaml``.

Binds :class:`~recipes.beta.coral.processor.CoralProcessor` to the grouped
relative-reward training machinery. CORAL sibling groups have the same shape
as TTT-Discover steps — comparison sets of scored rollouts from one shared
starting point — so the recipe reuses the ``tttd`` training objective (grouped
adaptive-entropic leave-one-out advantages) and the ``tttd`` Slime loss
family rather than duplicating either. The recipe's own configuration is the
group barrier.

Deployment reference: ``recipes.beta.coral.recipe:CoralRecipe``. Run from
the repository checkout, which also provides the ``tttd`` registrations this
recipe reuses; ``reef serve`` finds it beside the example config.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

# The tttd training objective and loss-family reference live in the repository
# cookbook; importing them is what registers both names this recipe binds.
import recipes.tttd  # noqa: F401  (registration side effect)
from recipes.beta.coral.processor import CoralProcessor, attempt_release_ids, sibling_group
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.record2record import GroupDecision, ReportContext, ReportGrouping, ReportStage
from reef.train.algos import StepScheduling
from reef.train.types import ProcessorContext

logger = logging.getLogger(__name__)


class CoralGrouping(ReportGrouping):
    """The siblings of one parent commit form one group, and each attempt's commit hash is its slot.

    A group is ready once ``group_size`` siblings are present. A group whose
    attempts span releases is discarded and listed in ``discarded_groups``.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.mixed_release_groups: dict[str, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return sibling_group(context)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {release_id for member in members for release_id in attempt_release_ids(member)}
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self.mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(members),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self.mixed_release_groups.items())
            ],
        }


def coral_record_stage(context: ProcessorContext) -> ReportStage:
    """CORAL's record stage: sibling groups in the order that they close; a terminal report owns its sources."""
    return ReportStage(
        context.scenario,
        context.report_type,
        CoralGrouping(int(context.config.get("group_size", 4))),
        exclusive_sources=True,
    )


@dataclass(frozen=True, kw_only=True)
class CoralRecipe(WeightTrainingRecipe):
    name: str = "coral"
    #: Scored siblings of one parent commit required before their group trains.
    group_size: int = config_field(4, env="REEF_CORAL_GROUP_SIZE")

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective="tttd",
            processor=CoralProcessor,
            # Sibling groups train as one optimizer step, as TTTD's grid does.
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    def build_record_stage(self, context: ProcessorContext) -> ReportStage:
        return coral_record_stage(context)

    def __post_init__(self) -> None:
        # Validate our own field first: the group barrier is checkable
        # without a fully constructed training runtime.
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")
        super().__post_init__()
