"""Harness evolution: recorded requests as traces, with reported scores or unscored, batched unmodified."""

from __future__ import annotations

from reef.core import RequestType
from reef.core.trajectories import make_trajectory
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    InferenceRecord2Record,
    RecordSet,
    ReportContext,
    ReportGrouping,
    ReportStage,
    SingleRecordSet,
)
from reef.record2task import Record2Task, ReportTasks, TaskResult
from reef.record2task.assembly import reported_task
from reef.task2batch import Task2Batch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch, TrajectoryItem


class TraceTasks(ReportTasks):
    """One report's referenced requests, as served, as one trace task with the reported score."""

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)


class InferenceTasks(Record2Task):
    """One recorded inference as one unscored trace task."""

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        if not isinstance(record_set, SingleRecordSet):
            raise TypeError(
                f"{type(self).__name__} makes tasks from single-record sets, not {type(record_set).__name__}"
            )
        return TaskResult.ready((make_trajectory((record_set.record,)),))


class TraceBatches(Task2Batch):
    """Batch trace tasks unmodified; the backends consume them without more processing.

    In ``hybrid`` a queued instruction batches with the traces that an
    automatic batch would take next, up to ``batch_size``, so the proposer
    reads the request beside them; in ``manual`` it runs alone.
    """

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:harness_evolve:{batch_number}", tasks)


def trace_pipeline(context: PipelineContext, batch_policy: str) -> TrainingPipeline:
    """The harness-evolution pipeline of one scenario: traces from reports, or from recorded inference traffic.

    * ``reports``: pair recorded requests with reported scores. Requests are
      recorded post-transform, so a trace shows exactly what the backend
      served. Every valid report contributes one trace; several references
      become one trajectory sample.
    * ``records``: batch recorded inference traffic every ``batch_size``
      requests, unscored, so a deployment that only serves still evolves.
      Each recorded inference is one set, in arrival order, and reports are
      released untouched. The proposer contract requires handling unscored
      samples; a deployment with real outcome signal selects ``reports``,
      because a measured result beats model self judgment.
    """
    if batch_policy == "records":
        return TrainingPipeline(
            context,
            record2record=InferenceRecord2Record(),
            record2task=InferenceTasks(),
            task2batch=TraceBatches(context),
        )
    return TrainingPipeline(
        context,
        record2record=ReportStage(context.scenario, context.report_type, ReportGrouping()),
        record2task=TraceTasks(),
        task2batch=TraceBatches(context),
    )
