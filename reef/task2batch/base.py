"""The task-to-batch contract: a buffer of samples, and the samples of the selected batch units become one batch.

The tasks of a record set enter the stage as samples when the record-to-task
stage makes them. The buffer holds the samples of the record sets that wait
for a batch, at most ``buffer_size``. The training pipeline
(:class:`~reef.train.pipeline.TrainingPipeline`) makes a batch when the
buffer holds the candidate backend's batch size: it selects whole batch units
from the record-to-record stage and gives their samples to a recipe's
``Task2Batch``. A task is a ``TrainDataItem``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.batches import TrainDataItem, TrainingBatch
from reef.core.config import config_option
from reef.core.plugins import NoSettings, Plugin, PluginConfigError, PluginSettings
from reef.core.records_types import RequestType
from reef.observability import ExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.record2record import Record2Record
from reef.record2task import Record2Task


#: The batches that a buffer holds when the deployment does not set ``buffer_size``.
DEFAULT_BUFFER_BATCHES = 4


@dataclass(frozen=True)
class BufferSettings(PluginSettings):
    """The config of a task-to-batch stage: the size of its sample buffer.

    Values: ``buffer_size`` is None or a positive int.
    Purpose: a stage's ``from_config`` parses it from
    ``pipeline.task2batch.config`` and gives ``buffer_size`` to
    ``Task2Batch.__init__``. None there means ``DEFAULT_BUFFER_BATCHES``
    batches of the backend's batch size. A stage settings class that has
    more keys extends it.
    """

    buffer_size: int | None = config_option(
        None, help="The most samples the buffer holds; at least the batch size. Omitted holds 4 batches."
    )

    def __post_init__(self) -> None:
        if self.buffer_size is not None and self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")


class Task2Batch(ABC, Plugin):
    """The task-to-batch stage: bound the sample buffer, and shape the samples of the selected batch units into one batch.

    Values: ``buffer_size`` is an int and ``buffer_size >= context.batch_size``;
    the constructor refuses a smaller size. The pipeline keeps at most
    ``buffer_size`` samples of held record sets. A recipe implements ``make_batch``. The class attributes declare what the
    pipeline and the trainer need to know about the recipe's batches.
    ``make_batch`` runs on the trainer thread without network or model calls.
    A deployment names a stage class in ``pipeline.task2batch`` and gives its
    config. The class parses the config into its settings dataclass and
    builds the stage in ``from_config``.
    """

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: The training modes that the recipe's batches support. ``manual`` and
    #: ``hybrid`` batch on a training instruction.
    supported_training_modes: frozenset[str] = frozenset({"auto"})
    #: The kinds of record that the trainer gives the pipeline.
    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    #: Whether the recipe trains on uploaded trajectories. Admission refuses an upload
    #: when no trainer of the scenario accepts uploaded trajectories.
    accepts_trajectories: bool = False

    def __init__(self, context: PipelineContext, buffer_size: int | None) -> None:
        self.context = context
        self.scenario = context.scenario
        #: The most samples that the buffer holds, at least ``context.batch_size``.
        self.buffer_size = DEFAULT_BUFFER_BATCHES * context.batch_size if buffer_size is None else buffer_size
        if self.buffer_size < context.batch_size:
            raise PluginConfigError(
                f"{type(self).__name__} buffer_size {self.buffer_size} is smaller than the batch size "
                f"{context.batch_size}"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the stage of one scenario from its config.

        ``record2record`` and ``record2task`` are the scenario's earlier stages, built first.
        ``environ`` holds the environment the deployment was read with.
        """
        raise PluginConfigError(f"{cls.__name__} cannot be built from a deployment file: it has no from_config")

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, pipeline, and backend."""
        return self.context.experiment_logger

    @abstractmethod
    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        """Build a batch from the tasks of the selected record sets, in batch order.

        Use ``batch_id`` as the ID of the returned batch. The pipeline assigns
        it. An automatic batch gets ``<scenario>:batch:<number>``. An
        instruction's batch gets ``<scenario>:instruction:<request id>``. The
        number counts the batches of this pipeline and starts again when the
        pipeline restarts. It is not a training step.
        The pipeline tracks consumption by record set, including the tasks
        that the recipe leaves out of the batch. In ``manual`` mode an
        instruction's batch has no tasks.
        """


class NoBatches(Task2Batch):
    """Make no batch: the no-update default. A pipeline with it never becomes ready."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context)

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context, None)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        raise RuntimeError(f"scenario {self.scenario!r} trains no weights and never produces a training batch")
