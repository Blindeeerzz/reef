"""Reported feedback: assemble reports and their inference records into samples and batches.

The record-to-record stage (:class:`~reef.record2record.ReportStage`) reads
reports in append order. It validates them, places them in groups, and decides
which singleton reports and ready groups train. This processor turns each
report into a sample and each reserved selection of sets into a batch.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.record2record import (
    AssemblyStatus,
    GroupDecision,
    RecordSet,
    ReportCandidate,
    ReportContext,
    ReportGrouping,
    ReportStage,
    SetAssembly,
    accepted_by,
    report_sets,
)
from reef.train.processors.base import DataProcessor
from reef.train.processors.common import (
    make_multi_turn_policy_trajectory,
    make_policy_trajectory,
    sample_assembly_config_fields,
)
from reef.train.types import ProcessorContext, TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

__all__ = [
    "GroupDecision",
    "ReportContext",
    "ReportGrouping",
    "ReportedFeedbackProcessor",
    "SampleAssembly",
    "SkippedReport",
    "accepted_by",
]

#: Hooks that moved from the processor to ``ReportGrouping``. A subclass that still defines one fails at import.
MOVED_REPORT_HOOKS = ("grouping", "decide_group", "is_training_report")


@dataclass(frozen=True)
class SkippedReport:
    """``make_sample`` returns this for a valid report that cannot train. The stage releases the report."""

    reason: str


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """Assemble valid reports and their existing inference records into batches.

    Recipes implement ``make_sample`` and ``make_batch``. A grouped method
    also returns its :class:`~reef.record2record.ReportGrouping` from
    ``report_grouping``. The record-to-record stage owns deduplication, group
    slots, selection, consumption, and record release. Invalid references
    raise immediately; training data failures propagate instead of silently
    dropping reports.

    The units for a batch are accepted singleton reports and ready groups, in
    arrival order, or with ready groups in key order under ``ordered_groups``.
    """

    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: Whether a terminal report owns its referenced sources outright.
    exclusive_sources: bool = False
    #: Batch ready groups in group-key order instead of arrival order.
    ordered_groups: bool = False
    #: Units held in manual mode beyond this many batches are released, oldest first.
    manual_unit_cap_batches: int = 4

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        moved = [name for name in MOVED_REPORT_HOOKS if name in vars(cls)]
        if moved:
            raise TypeError(
                f"{cls.__name__} defines {', '.join(moved)}: these hooks moved to "
                "reef.record2record.ReportGrouping (is_training_report, placement, decide). "
                "Return the grouping from report_grouping()."
            )

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        # The sample-assembly settings belong to every reported-feedback processor's config
        # surface: validate them here so a bad deployment fails at construction
        # even for recipes that never assemble a policy sample.
        sample_assembly_config_fields(context.config)
        #: Assembled samples by report ID, until their report trains or ends.
        self.samples: dict[str, TrainDataItem] = {}
        self.record_stage.limit_available(self.available_set_limit())

    def build_record_stage(self) -> ReportStage:
        return ReportStage(
            self.scenario,
            self.context.report_type,
            self.report_grouping(),
            exclusive_sources=self.exclusive_sources,
            ordered_groups=self.ordered_groups,
        )

    @property
    def report_stage(self) -> ReportStage:
        stage = self.record_stage
        if not isinstance(stage, ReportStage):
            raise TypeError(f"{type(self).__name__} needs a ReportStage, not {type(stage).__name__}")
        return stage

    # ------------------------------------------------------- the recipe hooks

    def report_grouping(self) -> ReportGrouping:
        """Return this method's report policy. The default trains every report by itself."""
        return ReportGrouping()

    @abstractmethod
    def make_sample(self, context: ReportContext) -> TrainDataItem | SkippedReport:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Return
        ``SkippedReport`` for a valid report that cannot train.
        """

    @abstractmethod
    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        """Build a batch from selected items, in group and arrival order.

        The processor tracks consumption independently, including any selected
        items the recipe omits from the resulting batch.
        """

    # --------------------------------------------------------------- assembly

    def assemble(self, record_set: RecordSet) -> SetAssembly:
        if not isinstance(record_set, ReportCandidate):
            raise TypeError(f"{type(self).__name__} assembles report candidates, not {type(record_set).__name__}")
        context = record_set.context
        sample = self.make_sample(context)
        if isinstance(sample, SkippedReport):
            return SetAssembly(AssemblyStatus.REJECTED)
        if not isinstance(sample, (TrajectoryItem, TaskItem)):
            raise TypeError(f"{type(self).__name__}.make_sample must return a TrainDataItem")
        report_id = context.report.agent_record_id
        self.samples[report_id] = replace(sample, source_agent_record_ids=(*context.references, report_id))
        return SetAssembly(AssemblyStatus.READY)

    def forget_sets(self, set_ids: frozenset[Hashable]) -> None:
        for set_id in set_ids:
            if isinstance(set_id, str):
                self.samples.pop(set_id, None)

    def set_training_mode(self, training_mode: str) -> None:
        super().set_training_mode(training_mode)
        self.record_stage.limit_available(self.available_set_limit())
        self.record_stage.forget_ended(self)

    def available_set_limit(self) -> int | None:
        """Manual mode batches on instructions, not sets, so the held sets are bounded instead."""
        return self._batch_size * self.manual_unit_cap_batches if self.training_mode == "manual" else None

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one accepted singleton report or one ready group; the
    # engine's half of the shared cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return self.record_stage.available_count()

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        report_sets_reserved = report_sets(self.record_stage.reserve(self._batch_size))
        items = tuple(
            self.samples[entry.report_id] for report_set in report_sets_reserved for entry in report_set.entries
        )
        return self.make_batch(items, batch_number)

    def reserved_report_ids(self) -> set[str]:
        """Return the IDs of the reports in the reserved batch."""
        return self.report_stage.reserved_report_ids()

    def _consume_pending(self) -> frozenset[str]:
        consumed = self.record_stage.consume_reservation()
        self.record_stage.forget_ended(self)
        return consumed


# --------------------------------------------------------- shared helpers


def reported_task(metadata: object) -> dict[str, Any] | None:
    """The task a report names, so a recipe can group samples by task; None when the report names none.

    ``metadata.task`` is a mapping with at least a ``name`` (the task player also sends ``path`` and
    ``digest``); a Harbor report from the shipped harness names the task as ``metadata.harbor.task_name``.
    """
    if not isinstance(metadata, Mapping):
        return None
    task = metadata.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("name"), str) and task["name"]:
        return dict(task)
    harbor = metadata.get("harbor")
    if isinstance(harbor, Mapping) and isinstance(harbor.get("task_name"), str) and harbor["task_name"]:
        return {"name": harbor["task_name"]}
    return None


@dataclass(frozen=True)
class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    One model call becomes one sample via ``make_sample`` (default: the shared
    tensor reader); an ordered multi-reference report becomes one assembled
    multi-turn sample. ``accept_multi_turn`` gates consumption only —
    assembly always runs first. Unsupported or unassemblable trajectories
    raise a training data error and retain their source records.
    """

    accept_multi_turn: bool = False
    realign_threshold: int = 1024
    scaffold_tolerance: int = 0
    make_sample: Callable[[AgentRecord, float], TrajectoryItem] | None = None

    @classmethod
    def from_config(
        cls,
        context: ProcessorContext,
        make_sample: Callable[[AgentRecord, float], TrajectoryItem] | None = None,
    ) -> SampleAssembly:
        accept_multi_turn, realign_threshold, scaffold_tolerance = sample_assembly_config_fields(context.config)
        return cls(accept_multi_turn, realign_threshold, scaffold_tolerance, make_sample)

    def build(self, context: ReportContext, score: float) -> TrajectoryItem:
        """Build training data, raising if the recorded trajectory cannot be used."""
        inferences = context.inferences
        if inferences is None:
            raise RuntimeError("sample assembly requires resolved inferences")
        if len(inferences) == 1:
            sample: TrajectoryItem | None = (
                make_policy_trajectory(inferences[0], score)
                if self.make_sample is None
                else self.make_sample(inferences[0], score)
            )
        else:
            sample = make_multi_turn_policy_trajectory(
                inferences,
                score,
                source_agent_record_id=context.report.agent_record_id,
                realign_threshold=self.realign_threshold,
                scaffold_tolerance=self.scaffold_tolerance,
            )
        if sample is None:
            raise ValueError("training data cannot assemble the recorded multi-turn trajectory")
        if (sample.training.get("turn_count", 1) > 1) and not self.accept_multi_turn:
            raise ValueError("training data requires accept_multi_turn_policy_samples for this trajectory")
        fields: dict[str, Any] = {
            "feedback": context.report.payload.get("feedback"),
            "report_agent_record_id": context.report.agent_record_id,
            "references": list(context.references),
        }
        task = reported_task(context.report.payload.get("metadata"))
        if task is not None:
            fields["task"] = task
        return sample.with_metadata(**fields)
