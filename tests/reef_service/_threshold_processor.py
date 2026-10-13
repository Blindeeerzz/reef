"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from reef.record2record import ReportContext, ReportGrouping, ReportStage
from reef.record2task import ReportTasks
from reef.train.pipeline import TrainingPipeline
from reef.train.processors.base import Task2Batch
from reef.train.processors.reported import SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class ThresholdTasks(ReportTasks):
    """One report becomes one sample, with no score filtering."""

    def __init__(self, context: ProcessorContext) -> None:
        self.assembly = SampleAssembly.from_config(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class ThresholdBatches(Task2Batch):
    """Batch the tasks of ``ThresholdTasks`` as they come."""

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:threshold:{batch_number}", tasks)


def threshold_pipeline(
    context: ProcessorContext,
    *,
    record_stage: ReportStage | None = None,
    task2batch: Task2Batch | None = None,
) -> TrainingPipeline:
    """Every report by itself (or ``record_stage``), one sample each, batched by ``ThresholdBatches`` (or ``task2batch``)."""
    return TrainingPipeline(
        context,
        record_stage=(
            ReportStage(context.scenario, context.report_type, ReportGrouping())
            if record_stage is None
            else record_stage
        ),
        record2task=ThresholdTasks(context),
        task2batch=ThresholdBatches(context) if task2batch is None else task2batch,
    )
