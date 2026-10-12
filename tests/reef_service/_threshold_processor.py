"""Shared policy processor test double: assemble every valid scored report."""

from __future__ import annotations

from reef.record2record import ReportStage
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class ThresholdProcessor(ReportedFeedbackProcessor):
    """One report becomes one sample, with no score filtering."""

    output_schema = TrainingBatch

    def __init__(self, context: ProcessorContext, *, record_stage: ReportStage | None = None) -> None:
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context, record_stage=record_stage)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self._assembly.build(context, context.require_score())

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(
            f"{self.scenario}:threshold:{batch_number}",
            items,
        )
