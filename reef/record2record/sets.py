"""Record sets and batch units: the input of the record-to-task stage, and what a batch takes.

Notation: ``refs(r)`` is the set of records that a record ``r`` references.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass

from reef.core.records_types import AgentRecord


@dataclass(frozen=True, kw_only=True)
class RecordSet(ABC):
    """The records that the record-to-task stage makes tasks from, together.

    Values: ``X`` is a record set if and only if ``records(X)`` is not empty and
    ``refs(r) ⊆ records(X)`` for each record ``r`` in ``records(X)``. ``set_id``
    identifies ``X`` among the sets of its record stage. ``arrival_index`` is
    the position in which the stage formed ``X``. No record is in two record
    sets of one stage (``Record2Record.form``).

    Purpose: the record stage forms it, and the pipeline pushes it to the
    record-to-task stage. Its one result settles all records of the set together.
    """

    set_id: Hashable
    arrival_index: int

    def __post_init__(self) -> None:
        records = self.records()
        if not records:
            raise ValueError("a record set must hold one or more records")
        missing = {reference for record in records for reference in record.references} - set(self.record_ids())
        if missing:
            raise ValueError(f"record set {self.set_id!r} does not hold the referenced records {sorted(missing)!r}")

    @abstractmethod
    def records(self) -> tuple[AgentRecord, ...]:
        """Return the records of the set."""

    def record_ids(self) -> tuple[str, ...]:
        """Return the IDs of the records of the set, in the order of ``records``."""
        return tuple(record.agent_record_id for record in self.records())


@dataclass(frozen=True, kw_only=True)
class SingleRecordSet(RecordSet):
    """One record that references no other record. ``set_id`` is the record ID."""

    record: AgentRecord

    def records(self) -> tuple[AgentRecord, ...]:
        return (self.record,)


@dataclass(frozen=True, kw_only=True)
class BatchUnit(ABC):
    """What a batch takes all of or none of.

    Values: ``U`` is a batch unit if and only if ``consumed_ids(U)``, the IDs
    that consuming ``U`` writes to the commit, is not empty. ``unit_id``
    identifies ``U`` among the units of its kind. ``arrival_index`` orders the
    units of a pool. ``group_key`` names the group of a unit that is a group.
    It is None for any other unit. A unit has one of two forms:
    ``RecordSetUnit``, the record sets of a record stage, and
    ``reef.train.pipeline.DataItemUnit``, one uploaded data item.

    Purpose: a batch is a sequence of batch units, and ``batch_size`` counts
    them. The pipeline writes ``consumed_ids(U)`` of each unit of an
    acknowledged batch to the commit.
    """

    unit_id: Hashable
    arrival_index: int
    group_key: Hashable | None = None

    def __post_init__(self) -> None:
        if not self.consumed_ids():
            raise ValueError(f"batch unit {self.unit_id!r} must consume one or more IDs")

    @abstractmethod
    def consumed_ids(self) -> frozenset[str]:
        """Return the IDs that consuming the unit writes to the commit."""


@dataclass(frozen=True, kw_only=True)
class RecordSetUnit(BatchUnit):
    """Record sets that must train in the same batch.

    Values: ``U`` is a record set unit if and only if ``record_sets(U)`` is not
    empty. Its record sets are distinct and share no record, because a record
    stage forms each record into one set only. ``consumed_ids(U)`` is the set
    of the IDs of the records of its record sets.

    Purpose: the record stage puts it in its pool when the tasks of its record
    sets are ready. The pipeline reserves it from the pool for a batch and
    gives the consumed units back to the stage.
    """

    @abstractmethod
    def record_sets(self) -> tuple[RecordSet, ...]:
        """Return the record sets of the unit, in batch order."""

    def set_ids(self) -> tuple[Hashable, ...]:
        """Return the IDs of the record sets of the unit, in batch order."""
        return tuple(record_set.set_id for record_set in self.record_sets())

    def consumed_ids(self) -> frozenset[str]:
        return frozenset(record_id for record_set in self.record_sets() for record_id in record_set.record_ids())


@dataclass(frozen=True, kw_only=True)
class SingleSetUnit(RecordSetUnit):
    """A batch unit of one record set, with the ID and the arrival index of that set. ``of`` makes it."""

    record_set: RecordSet

    @classmethod
    def of(cls, record_set: RecordSet) -> SingleSetUnit:
        """Return the unit of ``record_set``."""
        return cls(unit_id=record_set.set_id, arrival_index=record_set.arrival_index, record_set=record_set)

    def record_sets(self) -> tuple[RecordSet, ...]:
        return (self.record_set,)
