"""The record-to-record stage: records in, record sets out, for one processor.

``Record2RecordStage`` is the interface. ``PassThroughStage`` is the default implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.record2record.buffer import UnitBuffer
from reef.record2record.eligibility import EligibilityCheck
from reef.record2record.selection import SelectionPolicy
from reef.record2record.units import BatchUnit


@dataclass(frozen=True, kw_only=True)
class RecordSet(BatchUnit, ABC):
    """Records that are used together, such as one record or a report with the records it references.

    A record set is a ``BatchUnit``, so the buffer, the selection
    policies, and the eligibility checks can hold and order it.
    """

    @abstractmethod
    def record_ids(self) -> tuple[str, ...]:
        """Return the IDs of the records in the set."""


@dataclass(frozen=True, kw_only=True)
class SingleRecordSet(RecordSet):
    """One record by itself. ``unit_id`` is the record's ID."""

    record: AgentRecord

    def record_ids(self) -> tuple[str, ...]:
        return (self.record.agent_record_id,)


def record_sets(units: Sequence[BatchUnit]) -> tuple[RecordSet, ...]:
    """Return pool units as record sets. A stage puts only record sets in its pool."""
    selected = tuple(unit for unit in units if isinstance(unit, RecordSet))
    if len(selected) != len(units):
        raise TypeError("a record-to-record pool must hold only record sets")
    return selected


class AssemblyStatus(Enum):
    """What the processor did with a formed record set."""

    #: The processor made training data from the set. The stage holds the set until a batch uses it.
    READY = "ready"
    #: The set cannot train. The stage ends its records.
    REJECTED = "rejected"
    #: The processor keeps the records itself. The stage forgets the set.
    INGESTED = "ingested"


@dataclass(frozen=True)
class SetAssembly:
    """The result of one assembly.

    ``runtime_load_id`` is the runtime load ID of the training data, if the
    processor knows it. The stage's eligibility checks read it.
    """

    status: AssemblyStatus
    runtime_load_id: str | None = None


class SetAssembler(ABC):
    """Turn record sets into training data. A processor implements this interface."""

    @abstractmethod
    def assemble(self, record_set: RecordSet) -> SetAssembly:
        """Make training data from one formed set. Raise on a broken training contract."""

    def forget_sets(self, set_ids: frozenset[Hashable]) -> None:
        """Forget the training data of sets that the stage ended. The default keeps nothing."""
        return


class Record2RecordStage(ABC):
    """Turn records into record sets for one processor, and decide which sets train and in what order.

    The processor gives the stage every record that it reads, except training
    instructions, in append order. The stage forms record sets. The
    processor assembles each formed set (``assemble_formed``). The stage holds
    each assembled set in its pool until the processor reserves it for a
    batch. The pool applies the stage's selection policy and eligibility checks.

    A record that the stage ends, for example a record that can no longer
    train, appears in ``releasable_record_ids``. The trainer adds those IDs to
    the next commit, so the record never trains after a restart.

    The stage keeps only memory that a replay of the retained records
    rebuilds, as processor memory does.
    """

    def __init__(self, selection: SelectionPolicy | None = None, checks: Sequence[EligibilityCheck] = ()) -> None:
        self.pool = UnitBuffer(selection, checks)
        #: Formed sets that ended after assembly. The assembler forgets their training data.
        self.ended_set_ids: set[Hashable] = set()
        #: Whether the processor handed out a batch that the trainer has not acknowledged or released yet.
        self.batch_out = False

    @abstractmethod
    def ingest(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    @abstractmethod
    def take_formed(self) -> tuple[RecordSet, ...]:
        """Return the record sets formed since the last call, in the order they formed."""

    def poll(self, now: float) -> None:
        """Absorb background work, such as finished judgments. ``now`` is a monotonic time."""
        return

    def assemble_formed(self, assembler: SetAssembler) -> None:
        """Give each formed set to ``assembler``, and keep or end the set from the result."""
        for record_set in self.take_formed():
            assembly = assembler.assemble(record_set)
            if assembly.status is AssemblyStatus.READY:
                self.accept(record_set, assembly)
            elif assembly.status is AssemblyStatus.REJECTED:
                self.reject(record_set)
            elif assembly.status is not AssemblyStatus.INGESTED:
                raise TypeError("assemble must return a SetAssembly with an AssemblyStatus")
        self.forget_ended(assembler)

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        """Hold an assembled set in the pool. The default stage never forms a set that it must hold."""
        raise NotImplementedError(f"{type(self).__name__} does not hold assembled sets")

    def reject(self, record_set: RecordSet) -> None:
        """End a set that cannot train. The default stage never forms a set that it must end."""
        raise NotImplementedError(f"{type(self).__name__} does not end rejected sets")

    def forget_ended(self, assembler: SetAssembler) -> None:
        """Tell ``assembler`` which formed sets ended since the last call."""
        if self.ended_set_ids:
            ended, self.ended_set_ids = frozenset(self.ended_set_ids), set()
            assembler.forget_sets(ended)

    # ---------------------------------------------------------- reservation

    def available_count(self) -> int:
        """Return the number of sets that a batch can use."""
        return len(self.pool)

    def mark_batch_out(self, batch_out: bool) -> None:
        """Record whether a batch is out. A stage does not drop held sets under a batch that is out."""
        self.batch_out = batch_out

    def reserve(self, max_set_count: int) -> tuple[RecordSet, ...]:
        """Select at most ``max_set_count`` sets as the next batch, in batch order, and keep them reserved."""
        return record_sets(self.pool.reserve(max_set_count))

    def reserved_sets(self) -> tuple[RecordSet, ...]:
        """Return the reserved sets. Return an empty tuple if no batch is reserved."""
        return record_sets(self.pool.reserved_units())

    def consume_reservation(self) -> frozenset[str]:
        """End the reserved sets as trained, and return the IDs of the records that they consumed."""
        if self.pool.reserved is None:
            raise RuntimeError("cannot consume a batch before sets are reserved")
        return self.mark_trained(record_sets(self.pool.consume_reserved()))

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        """End sets that a committed batch trained on, and return the IDs of their consumed records."""
        raise NotImplementedError(f"{type(self).__name__} does not hold assembled sets")

    def limit_available(self, max_set_count: int | None) -> None:
        """Hold at most ``max_set_count`` unreserved sets. ``None`` removes the limit."""
        return

    # ------------------------------------------------------------- lifecycle

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the report contract that the scenario admits when it is wider than the processor's own."""
        return

    def releasable_record_ids(self) -> frozenset[str]:
        """Return the IDs of records that the stage ended and that no live set references."""
        return frozenset()

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Forget released records. Stored records do not change."""
        return

    def derivation_pending(self) -> bool:
        """Return True while background work can form sets without a new record."""
        return False

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that the processor adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the processor adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class PassThroughStage(Record2RecordStage):
    """Hand each record to the processor by itself, in arrival order, and decide nothing.

    The processor keeps the records itself (``AssemblyStatus.INGESTED``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.formed: list[RecordSet] = []
        self.arrival_count = 0

    def ingest(self, record: AgentRecord) -> None:
        self.arrival_count += 1
        self.formed.append(
            SingleRecordSet(unit_id=record.agent_record_id, arrival_index=self.arrival_count, record=record)
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed, self.formed = tuple(self.formed), []
        return formed


class InferenceStage(Record2RecordStage):
    """Hold each inference record as one set, in arrival order, and end every other record untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.formed: list[RecordSet] = []
        #: Records that ended: records that are not inferences, and trained inferences.
        self.terminal: set[str] = set()

    def ingest(self, record: AgentRecord) -> None:
        if record.request_type is not RequestType.INFERENCE:
            self.terminal.add(record.agent_record_id)
            return
        self.formed.append(
            SingleRecordSet(
                unit_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record
            )
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed, self.formed = tuple(self.formed), []
        return formed

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        self.pool.put(record_set)

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        consumed = frozenset(record_id for record_set in trained_sets for record_id in record_set.record_ids())
        self.terminal |= consumed
        self.ended_set_ids.update(record_set.unit_id for record_set in trained_sets)
        return consumed

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset(self.terminal)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.terminal -= agent_record_ids
