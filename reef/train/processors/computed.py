"""Computed-feedback engine: turn judged records into samples and batches.

The sibling of :class:`~reef.train.processors.reported.ReportedFeedbackProcessor`
for the other way feedback arrives. A reported-feedback recipe is handed
explicit reports; a computed-feedback recipe produces its signal from the
traffic itself. The record-to-record stage
(:class:`~reef.record2record.JudgingStage`) owns the record lifecycle, the
judging worker, and the eligibility checks. A recipe writes:

* a :class:`~reef.record2record.RecordJudging`: the method's correlation
  (``observe``, with the stage's ``dispatch``/``track``/``retire`` verbs),
  its ``judge`` coroutine on the worker thread, and optional ``expire``;
* ``make_sample(record, judgment)``: one judged record into a
  :class:`TrajectoryItem`, or ``None`` for a record that cannot train;
* ``make_batch(samples, batch_number)``: the selected samples into the
  recipe's batch type.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Any

from reef.background_jobs import BackgroundJobWorker, Failed, SupportsReceipt
from reef.core.records_types import AgentRecord
from reef.record2record import (
    AssemblyStatus,
    JudgedSet,
    JudgingStage,
    RecordJudging,
    RecordSet,
    SetAssembly,
    judged_sets,
)
from reef.train.processors.base import DataProcessor
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

# The worker types moved to reef.background_jobs. Recipes can still import them from here.
__all__ = [
    "BackgroundJobWorker",
    "ComputedFeedbackProcessor",
    "Failed",
    "SupportsReceipt",
]


class ComputedFeedbackProcessor(DataProcessor, ABC):
    """Assemble judged records into batches for computed-feedback recipes.

    The recipe's ``judging`` and ``worker`` build the processor's
    :class:`~reef.record2record.JudgingStage`. Judged sets wait in the order
    that their judgments arrived. The stage drops sets more than the
    context's ``max_staleness`` versions behind the newest one, by record
    arrival, while no batch is out.
    """

    def __init__(
        self, context: ProcessorContext, *, judging: RecordJudging, worker: BackgroundJobWorker | None
    ) -> None:
        #: The method's correlation and judge, and the worker that runs the judge. No worker: records
        #: track and expire, nothing is ever judged, everything can be released.
        self.judging = judging
        self.worker = worker
        super().__init__(context)
        #: Samples by receipt, until their record trains or ends.
        self.samples: dict[str, TrajectoryItem] = {}

    def build_record_stage(self) -> JudgingStage:
        return JudgingStage(self.judging, self.worker, max_staleness=self.context.max_staleness)

    @property
    def judging_stage(self) -> JudgingStage:
        stage = self.record_stage
        if not isinstance(stage, JudgingStage):
            raise TypeError(f"{type(self).__name__} needs a JudgingStage, not {type(stage).__name__}")
        return stage

    # ------------------------------------------------------- the recipe hooks

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> TrajectoryItem | None:
        """One judged record into a sample; ``None`` cannot train."""

    @abstractmethod
    def make_batch(self, samples: tuple[TrajectoryItem, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected samples into this recipe's batch type."""

    # --------------------------------------------------------------- assembly

    def assemble(self, record_set: RecordSet) -> SetAssembly:
        if not isinstance(record_set, JudgedSet):
            raise TypeError(f"{type(self).__name__} assembles judged sets, not {type(record_set).__name__}")
        sample = self.make_sample(record_set.record, record_set.judgment)
        if sample is None:
            return SetAssembly(AssemblyStatus.REJECTED)
        self.samples[record_set.receipt] = sample
        runtime_load_id = sample.training.get("runtime_load_id")
        return SetAssembly(AssemblyStatus.READY, runtime_load_id if isinstance(runtime_load_id, str) else None)

    def forget_sets(self, set_ids: frozenset[Hashable]) -> None:
        for set_id in set_ids:
            if isinstance(set_id, str):
                self.samples.pop(set_id, None)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one judged record's sample; the engine's half of the shared
    # cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return self.record_stage.available_count()

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        reserved = judged_sets(self.record_stage.reserve(self._batch_size))
        return self.make_batch(tuple(self.samples[judged_set.receipt] for judged_set in reserved), batch_number)

    def _consume_pending(self) -> frozenset[str]:
        consumed = self.record_stage.consume_reservation()
        self.forget_sets(self.record_stage.take_ended())
        return consumed
