"""The record-to-record stage: records in, record sets out, for one pipeline.

``Record2Record`` is the interface. ``PassThroughStage`` is the default implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.record2record.eligibility import EligibilityCheck
from reef.record2record.pool import RecordSetPool
from reef.record2record.selection import SelectionPolicy
from reef.record2record.sets import RecordSet, SingleRecordSet


class AssemblyStatus(Enum):
    """What the record-to-task stage made of a formed record set."""

    #: The set has tasks. The stage holds the set until a batch uses it.
    READY = "ready"
    #: The set cannot train. The stage ends its records.
    REJECTED = "rejected"
    #: The set makes no task, as in a pipeline that trains no weights. The stage forgets the set and keeps its records.
    INGESTED = "ingested"


@dataclass(frozen=True)
class SetAssembly:
    """The result of one assembly.

    ``runtime_load_id`` is the runtime load ID of the training data, if the
    pipeline knows it. The stage's eligibility checks read it.
    """

    status: AssemblyStatus
    runtime_load_id: str | None = None


class Record2Record(ABC):
    """Turn records into record sets for one pipeline, and decide which sets train and in what order.

    The pipeline gives the stage every record that it reads, except training
    instructions, in append order. The stage forms record sets
    (``take_formed``). The pipeline assembles each formed set and gives the
    result back (``settle``). The stage holds each assembled set in its pool
    until the pipeline reserves it for a batch. The pool applies the stage's
    selection policy and eligibility checks. The stage never calls the pipeline.

    A record that the stage ends, for example a record that can no longer
    train, appears in ``releasable_record_ids``. The trainer adds those IDs to
    the next commit, so the record never trains after a restart.

    The stage keeps only memory that a replay of the retained records
    rebuilds, as pipeline memory does.
    """

    def __init__(self, selection: SelectionPolicy | None = None, checks: Sequence[EligibilityCheck] = ()) -> None:
        self.pool = RecordSetPool(selection, checks)
        #: Formed sets that ended after assembly. The assembler forgets their training data.
        self.ended_set_ids: set[Hashable] = set()
        #: Whether the pipeline handed out a batch that the trainer has not acknowledged or released yet.
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

    def settle(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        """Keep or end a formed set from the pipeline's assembly result."""
        if assembly.status is AssemblyStatus.READY:
            self.accept(record_set, assembly)
        elif assembly.status is AssemblyStatus.REJECTED:
            self.reject(record_set)
        elif assembly.status is not AssemblyStatus.INGESTED:
            raise TypeError("an assembly result must have an AssemblyStatus")

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        """Hold an assembled set in the pool. The default stage never forms a set that it must hold."""
        raise NotImplementedError(f"{type(self).__name__} does not hold assembled sets")

    def reject(self, record_set: RecordSet) -> None:
        """End a set that cannot train. The default stage never forms a set that it must end."""
        raise NotImplementedError(f"{type(self).__name__} does not end rejected sets")

    def take_ended(self) -> frozenset[Hashable]:
        """Return the IDs of the formed sets that ended since the last call. The pipeline forgets their data."""
        ended, self.ended_set_ids = frozenset(self.ended_set_ids), set()
        return ended

    # ---------------------------------------------------------- reservation

    def available_count(self) -> int:
        """Return the number of sets that a batch can use."""
        return len(self.pool)

    def mark_batch_out(self, batch_out: bool) -> None:
        """Record whether a batch is out. A stage does not drop held sets under a batch that is out."""
        self.batch_out = batch_out

    def reserve(self, max_set_count: int) -> tuple[RecordSet, ...]:
        """Select at most ``max_set_count`` sets as the next batch, in batch order, and keep them reserved."""
        return self.pool.reserve(max_set_count)

    def reserved_sets(self) -> tuple[RecordSet, ...]:
        """Return the reserved sets. Return an empty tuple if no batch is reserved."""
        return self.pool.reserved_sets()

    def consume_reservation(self) -> frozenset[str]:
        """End the reserved sets as trained, and return the IDs of the records that they consumed."""
        if self.pool.reserved is None:
            raise RuntimeError("cannot consume a batch before sets are reserved")
        return self.mark_trained(self.pool.consume_reserved())

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        """End sets that a committed batch trained on, and return the IDs of their consumed records."""
        raise NotImplementedError(f"{type(self).__name__} does not hold assembled sets")

    def mark_batch_dropped(self) -> None:
        """The backend dropped the batch that is out as stale. Its acknowledgement follows."""
        return

    def limit_available(self, max_set_count: int | None) -> None:
        """Hold at most ``max_set_count`` unreserved sets. ``None`` removes the limit."""
        return

    # ------------------------------------------------------------- lifecycle

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the report contract that the scenario admits when it is wider than the pipeline's own."""
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
        """Return JSON-safe state that the pipeline adds to its status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the pipeline adds to its metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class PassThroughStage(Record2Record):
    """Hand each record to the pipeline by itself, in arrival order, and decide nothing.

    With ``NoTasks`` each set is ``AssemblyStatus.INGESTED``: the stage forgets it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.formed: list[RecordSet] = []
        self.arrival_count = 0

    def ingest(self, record: AgentRecord) -> None:
        self.arrival_count += 1
        self.formed.append(
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.arrival_count, record=record)
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed, self.formed = tuple(self.formed), []
        return formed


class InferenceStage(Record2Record):
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
            SingleRecordSet(set_id=record.agent_record_id, arrival_index=self.pool.next_arrival_index(), record=record)
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed, self.formed = tuple(self.formed), []
        return formed

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        self.pool.put(record_set)

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        consumed = frozenset(record_id for record_set in trained_sets for record_id in record_set.record_ids())
        self.terminal |= consumed
        self.ended_set_ids.update(record_set.set_id for record_set in trained_sets)
        return consumed

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset(self.terminal)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.terminal -= agent_record_ids
