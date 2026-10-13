"""The record-to-task contract: one record set from the record-to-record stage becomes tasks.

A task is a ``TrainDataItem``: a ``TrajectoryItem`` that trains directly, or a
``TaskItem`` that the algorithm rolls out. The training pipeline calls ``make_tasks``
for each set that the record stage forms and gives the ``TaskResult`` back to
the stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.batches import TrainDataItem
from reef.record2record import AssemblyStatus, RecordSet, SetAssembly


@dataclass(frozen=True)
class TaskResult:
    """The tasks made from one record set, and what the record stage does with the set."""

    assembly: SetAssembly
    tasks: tuple[TrainDataItem, ...] = ()

    @classmethod
    def ready(cls, tasks: tuple[TrainDataItem, ...], runtime_load_id: str | None = None) -> TaskResult:
        """The set trains as ``tasks``. ``runtime_load_id`` is the version of their data, when known."""
        return cls(SetAssembly(AssemblyStatus.READY, runtime_load_id), tasks)

    @classmethod
    def skipped(cls) -> TaskResult:
        """The set cannot train. The record stage ends its records."""
        return cls(SetAssembly(AssemblyStatus.REJECTED))


class Record2Task(ABC):
    """The record-to-task stage: turn one record set into tasks.

    Runs on the trainer thread without network or model calls. Raise on a
    broken training contract.
    """

    @abstractmethod
    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        """Return the tasks of ``record_set``, or a skip."""

    def poll(self) -> None:
        """Absorb finished background work, such as a finished round.

        The pipeline calls it each time it looks for a batch, before the record stage's ``poll``.
        """
        return

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def derivation_pending(self) -> bool:
        """Return True while background work, such as a running round, can change what trains."""
        return False

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class NoTasks(Record2Task):
    """Make no tasks. The record stage forgets each set; the no-update default."""

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        return TaskResult(SetAssembly(AssemblyStatus.INGESTED))
