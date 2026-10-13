"""Shared policy pipeline test double: assemble every valid scored report, and batch the samples as they come."""

from __future__ import annotations

from reef.pipeline_context import PipelineContext
from reef.record2record import ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2task import ReportTasks
from reef.record2task.assembly import SampleAssembly
from reef.task2batch import Task2Batch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


class ThresholdTasks(ReportTasks):
    """One report becomes one sample, with no score filtering."""

    def __init__(self, context: PipelineContext) -> None:
        self.assembly = SampleAssembly.from_config(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class ThresholdBatches(Task2Batch):
    """Batch the tasks of ``ThresholdTasks`` as they come."""

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:threshold:{batch_number}", tasks)


def threshold_pipeline(
    context: PipelineContext,
    *,
    record2record: ReportRecord2Record | None = None,
    task2batch: Task2Batch | None = None,
) -> TrainingPipeline:
    """Every report by itself (or ``record2record``), one sample each, batched by ``ThresholdBatches`` (or ``task2batch``)."""
    return TrainingPipeline(
        context,
        record2record=(
            ReportRecord2Record(context.scenario, context.report_type, ReportGrouping())
            if record2record is None
            else record2record
        ),
        record2task=ThresholdTasks(context),
        task2batch=ThresholdBatches(context) if task2batch is None else task2batch,
    )
