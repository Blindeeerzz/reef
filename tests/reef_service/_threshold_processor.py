"""Shared policy processor test double: assemble every valid scored report."""

from __future__ import annotations

from reef.record2record import ReportStage
from reef.train.processors.base import Record2Task
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, ReportTasks, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class ThresholdTasks(ReportTasks):
    """One report becomes one sample, with no score filtering."""

    def __init__(self, context: ProcessorContext) -> None:
        self.assembly = SampleAssembly.from_config(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class ThresholdProcessor(ReportedFeedbackProcessor):
    """Batch the tasks of ``ThresholdTasks`` as they come."""

    output_schema = TrainingBatch

    def __init__(
        self,
        context: ProcessorContext,
        *,
        record_stage: ReportStage | None = None,
        record2task: Record2Task | None = None,
    ) -> None:
        super().__init__(context, record_stage=record_stage, record2task=record2task)

    def default_record2task(self) -> Record2Task:
        return ThresholdTasks(self.context)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(
            f"{self.scenario}:threshold:{batch_number}",
            items,
        )
