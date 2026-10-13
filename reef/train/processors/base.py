"""The task-to-batch contract: the tasks of selected record sets become one batch.

The training pipeline (:class:`~reef.train.pipeline.TrainingPipeline`) selects
the record sets of the next batch from the record-to-record stage, takes the
tasks that the record-to-task stage made from them, and gives the tasks to a
recipe's ``Task2Batch``. A task is a ``TrainDataItem``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.records_types import RequestType
from reef.observability import ExperimentLogger
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch


class Task2Batch(ABC):
    """The task-to-batch stage: shape the tasks of the selected record sets into one batch.

    A recipe implements ``make_batch``. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's batches.
    ``make_batch`` runs on the trainer thread without network or model calls.
    """

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: The training modes that the recipe's batches support. ``manual`` and
    #: ``hybrid`` batch on a training instruction.
    supported_training_modes: frozenset[str] = frozenset({"auto"})
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    def __init__(self, context: ProcessorContext) -> None:
        self.context = context
        self.scenario = context.scenario
        batch_size = int(context.config.get("batch_size", 1))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        #: The number of record sets in one batch.
        self.batch_size = batch_size

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    @abstractmethod
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        """Build a batch from the tasks of the selected record sets, in batch order.

        The pipeline tracks consumption by record set, including the tasks
        that the recipe leaves out of the batch. In ``manual`` mode an
        instruction's batch has no tasks.
        """

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class NoBatches(Task2Batch):
    """Make no batch: the no-update default. A pipeline with it never becomes ready."""

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces a training batch")
