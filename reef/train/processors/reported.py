"""Reported feedback: reports and their inference records, as tasks and batches.

The record-to-record stage (:class:`~reef.record2record.ReportStage`) reads
reports in append order. It validates them, places them in groups, and decides
which singleton reports and ready groups train. ``ReportTasks`` turns each
report into a task (record-to-task), and ``ReportedFeedbackProcessor`` turns
each reserved selection of sets into a batch (task-to-batch).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.record2record import (
    GroupDecision,
    Record2RecordStage,
    RecordSet,
    ReportCandidate,
    ReportContext,
    ReportGrouping,
    ReportStage,
    accepted_by,
    report_sets,
)
from reef.train.processors.base import DataProcessor, Record2Task, TaskResult
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
    "ReportTasks",
    "ReportedFeedbackProcessor",
    "SampleAssembly",
    "SkippedReport",
    "accepted_by",
]

#: Report policy that moved from the processor to the recipe's ``ReportStage``. A subclass that still
#: defines one fails at import, because it would otherwise train with the default policy without an error.
MOVED_REPORT_HOOKS = ("grouping", "decide_group", "is_training_report", "exclusive_sources", "ordered_groups")


@dataclass(frozen=True)
class SkippedReport:
    """``make_sample`` returns this for a valid report that cannot train. The stage releases the report."""

    reason: str


class ReportTasks(Record2Task, ABC):
    """Turn one valid report and its inference records into one task.

    A recipe implements ``make_sample``. The stage attaches the report ID and
    its ordered references to the task's ``source_agent_record_ids``.
    """

    @abstractmethod
    def make_sample(self, context: ReportContext) -> TrainDataItem | SkippedReport:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Return
        ``SkippedReport`` for a valid report that cannot train.
        """

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        if not isinstance(record_set, ReportCandidate):
            raise TypeError(
                f"{type(self).__name__} makes tasks from report candidates, not {type(record_set).__name__}"
            )
        context = record_set.context
        sample = self.make_sample(context)
        if isinstance(sample, SkippedReport):
            return TaskResult.skipped()
        if not isinstance(sample, (TrajectoryItem, TaskItem)):
            raise TypeError(f"{type(self).__name__}.make_sample must return a TrainDataItem")
        report_id = context.report.agent_record_id
        return TaskResult.ready((replace(sample, source_agent_record_ids=(*context.references, report_id)),))


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """Shape the tasks of valid reports into batches.

    Recipes implement ``make_batch``. The recipe's
    :class:`~reef.record2record.ReportStage`, passed as ``record_stage``,
    owns the report policy (its ``ReportGrouping``), deduplication, group
    slots, selection, consumption, and record release. Without one, every
    report trains by itself. The recipe's ``ReportTasks``, passed as
    ``record2task``, makes the tasks. Invalid references raise immediately;
    training data failures propagate instead of silently dropping reports.
    """

    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: Units held in manual mode beyond this many batches are released, oldest first.
    manual_unit_cap_batches: int = 4

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        moved = [name for name in MOVED_REPORT_HOOKS if name in vars(cls)]
        if moved:
            raise TypeError(
                f"{cls.__name__} defines {', '.join(moved)}: the report policy moved to the recipe. "
                "Build a reef.record2record.ReportStage with a ReportGrouping (is_training_report, "
                "placement, decide) and exclusive_sources/ordered_groups in the recipe's "
                "build_record_stage(), and pass it to the processor as record_stage."
            )

    def __init__(
        self,
        context: ProcessorContext,
        *,
        record_stage: Record2RecordStage | None = None,
        record2task: Record2Task | None = None,
    ) -> None:
        super().__init__(context, record_stage=record_stage, record2task=record2task)
        if not isinstance(self.record_stage, ReportStage):
            raise TypeError(f"{type(self).__name__} needs a ReportStage, not {type(self.record_stage).__name__}")
        # The sample-assembly settings belong to every reported-feedback processor's config
        # surface: validate them here so a bad deployment fails at construction
        # even for recipes that never assemble a policy sample.
        sample_assembly_config_fields(context.config)
        self.record_stage.limit_available(self.available_set_limit())

    def default_record_stage(self) -> ReportStage:
        return ReportStage(self.scenario, self.context.report_type, ReportGrouping())

    def default_record2task(self) -> Record2Task:
        raise TypeError(f"{type(self).__name__} needs the recipe's ReportTasks as record2task")

    @property
    def report_stage(self) -> ReportStage:
        stage = self.record_stage
        if not isinstance(stage, ReportStage):
            raise TypeError(f"{type(self).__name__} needs a ReportStage, not {type(stage).__name__}")
        return stage

    # ------------------------------------------------------- the recipe hooks

    @abstractmethod
    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        """Build a batch from selected items, in group and arrival order.

        The processor tracks consumption independently, including any selected
        items the recipe omits from the resulting batch.
        """

    def set_training_mode(self, training_mode: str) -> None:
        super().set_training_mode(training_mode)
        self.record_stage.limit_available(self.available_set_limit())
        self.forget_sets(self.record_stage.take_ended())

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
            task
            for report_set in report_sets_reserved
            for entry in report_set.entries
            for task in self.tasks[entry.report_id]
        )
        return self.make_batch(items, batch_number)

    def reserved_report_ids(self) -> set[str]:
        """Return the IDs of the reports in the reserved batch."""
        return self.report_stage.reserved_report_ids()

    def _consume_pending(self) -> frozenset[str]:
        consumed = self.record_stage.consume_reservation()
        self.forget_sets(self.record_stage.take_ended())
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
