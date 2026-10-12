"""Harness-evolution processor: pair recorded requests with reported scores, unmodified."""

from __future__ import annotations

from collections.abc import Hashable

from reef.core import RequestType
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import make_trajectory
from reef.record2record import AssemblyStatus, InferenceStage, RecordSet, SetAssembly, SingleRecordSet
from reef.train.processors.base import DataProcessor
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, reported_task
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem


class CordisProcessor(ReportedFeedbackProcessor):
    """Pair recorded requests with reported scores and batch them unmodified.

    Requests are recorded post-transform, so a trace shows exactly what the
    backend served. Every valid report contributes a trace. A report may reference
    one request or a whole run's worth; several references become one
    trajectory sample. The backends consume
    the resulting trace batches without adding processor logic. In ``hybrid``
    a queued instruction batches with the reported traces an automatic batch
    would take next, up to ``batch_size``, so the proposer reads the request
    beside them; in ``manual`` it runs alone.
    """

    output_schema = TrainingBatch
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        if request is not None and self.training_mode == "manual":
            self.record_stage.reserve(0)
            return TrainingBatch(request.id, ())
        # In hybrid an instruction takes the units an automatic batch would, none included; the base attaches it.
        return self._make_pending(batch_number)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        sample = make_trajectory(context.inferences, context.require_score(), context.report.payload.get("feedback"))
        task = reported_task(context.report.payload.get("metadata"))
        return sample if task is None else sample.with_metadata(task=task)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:harness_evolve:{batch_number}", items)


class RecordDrivenTraceProcessor(DataProcessor):
    """Batch recorded inference traffic every ``batch_size`` requests, unscored.

    The report-free half of harness evolution: a deployment that only serves
    still evolves. Each recorded inference is one unit, in arrival order
    (:class:`~reef.record2record.InferenceStage`); when ``batch_size`` have
    accumulated they batch as trace samples with ``score=None``, and the
    proposer contract requires handling unscored samples. Reports that arrive
    under this policy are released untouched; a deployment with real outcome
    signal selects the reported policy instead, because a measured result
    beats model self judgment. In ``hybrid`` a queued instruction batches
    with the oldest held records, up to ``batch_size``, as an automatic batch
    would; in ``manual`` it runs alone.
    """

    output_schema = TrainingBatch
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})
    required_request_types = frozenset(RequestType)

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        #: Trace samples by inference record ID, until the record trains.
        self.samples: dict[str, TrajectoryItem] = {}

    def default_record_stage(self) -> InferenceStage:
        return InferenceStage()

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        if request is not None and self.training_mode == "manual":
            self.record_stage.reserve(0)
            return TrainingBatch(request.id, ())
        # In hybrid an instruction takes the records an automatic batch would, none included; the base attaches it.
        return self._make_pending(batch_number)

    def assemble(self, record_set: RecordSet) -> SetAssembly:
        if not isinstance(record_set, SingleRecordSet):
            raise TypeError(f"{type(self).__name__} assembles single-record sets, not {type(record_set).__name__}")
        self.samples[record_set.record.agent_record_id] = make_trajectory((record_set.record,))
        return SetAssembly(AssemblyStatus.READY)

    def forget_sets(self, set_ids: frozenset[Hashable]) -> None:
        for set_id in set_ids:
            if isinstance(set_id, str):
                self.samples.pop(set_id, None)

    def _ready_count(self) -> int:
        return self.record_stage.available_count()

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        selected = self.record_stage.reserve(self._batch_size)
        return TrainingBatch(
            f"{self.scenario}:harness_evolve:{batch_number}",
            tuple(self.samples[str(record_set.set_id)] for record_set in selected),
        )

    def _consume_pending(self) -> frozenset[str]:
        consumed = self.record_stage.consume_reservation()
        self.forget_sets(self.record_stage.take_ended())
        return consumed
