"""Computed-feedback engine: judged records as tasks and batches.

The sibling of :class:`~reef.train.processors.reported.ReportedFeedbackProcessor`
for the other way feedback arrives. A reported-feedback recipe is handed
explicit reports; a computed-feedback recipe produces its signal from the
traffic itself. The record-to-record stage
(:class:`~reef.record2record.JudgingStage`) owns the record lifecycle, the
judging worker, and the eligibility checks. A recipe writes:

* a :class:`~reef.record2record.RecordJudging`: the method's correlation
  (``observe``, with the stage's ``dispatch``/``track``/``retire`` verbs),
  its ``judge`` coroutine on the worker thread, and optional ``expire``;
* a ``JudgedTasks`` with ``make_sample(record, judgment)``: one judged record
  into a :class:`TrajectoryItem`, or ``None`` for a record that cannot train;
* ``make_batch(samples, batch_number)`` on the processor: the selected
  samples into the recipe's batch type.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from reef.background_jobs import BackgroundJobWorker, Failed, SupportsReceipt
from reef.core.records_types import AgentRecord
from reef.record2record import JudgedSet, JudgingStage, RecordJudging, RecordSet, judged_sets
from reef.train.processors.base import DataProcessor, Record2Task, TaskResult
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

# The worker types moved to reef.background_jobs. Recipes can still import them from here.
__all__ = [
    "BackgroundJobWorker",
    "ComputedFeedbackProcessor",
    "Failed",
    "JudgedTasks",
    "SupportsReceipt",
]


class JudgedTasks(Record2Task, ABC):
    """Turn one judged record into one task. A recipe implements ``make_sample``."""

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> TrajectoryItem | None:
        """One judged record into a sample; ``None`` cannot train."""

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        if not isinstance(record_set, JudgedSet):
            raise TypeError(f"{type(self).__name__} makes tasks from judged sets, not {type(record_set).__name__}")
        sample = self.make_sample(record_set.record, record_set.judgment)
        if sample is None:
            return TaskResult.skipped()
        runtime_load_id = sample.training.get("runtime_load_id")
        return TaskResult.ready((sample,), runtime_load_id if isinstance(runtime_load_id, str) else None)


class ComputedFeedbackProcessor(DataProcessor, ABC):
    """Shape judged records' tasks into batches for computed-feedback recipes.

    The recipe builds the :class:`~reef.record2record.JudgingStage` with its
    ``RecordJudging`` and worker, and passes it as ``record_stage``, with its
    ``JudgedTasks`` as ``record2task``. Judged sets wait in the order that
    their judgments arrived. The stage drops sets more than ``max_staleness``
    versions behind the newest one, by record arrival, while no batch is out.
    """

    def __init__(self, context: ProcessorContext, *, record_stage: JudgingStage, record2task: JudgedTasks) -> None:
        super().__init__(context, record_stage=record_stage, record2task=record2task)

    @property
    def judging_stage(self) -> JudgingStage:
        stage = self.record_stage
        if not isinstance(stage, JudgingStage):
            raise TypeError(f"{type(self).__name__} needs a JudgingStage, not {type(stage).__name__}")
        return stage

    @property
    def judging(self) -> RecordJudging:
        """The method's correlation and judge, which the stage runs."""
        return self.judging_stage.judging

    # ------------------------------------------------------- the recipe hooks

    @abstractmethod
    def make_batch(self, samples: tuple[TrajectoryItem, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected samples into this recipe's batch type."""

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one judged record's sample; the engine's half of the shared
    # cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return self.record_stage.available_count()

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        reserved = judged_sets(self.record_stage.reserve(self._batch_size))
        tasks = tuple(task for judged_set in reserved for task in self.tasks[judged_set.set_id])
        samples = tuple(task for task in tasks if isinstance(task, TrajectoryItem))
        if len(samples) != len(tasks):
            raise TypeError("computed-feedback tasks must be trajectories")
        return self.make_batch(samples, batch_number)

    def _consume_pending(self) -> frozenset[str]:
        consumed = self.record_stage.consume_reservation()
        self.forget_sets(self.record_stage.take_ended())
        return consumed
