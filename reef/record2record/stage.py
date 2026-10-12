"""The record-to-record stage: records in, record sets out, for one trainer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.records_types import AgentRecord
from reef.record2record.sets import RecordSet, SingleRecordSet


class RecordStage(ABC):
    """Turn records into record sets for one trainer, and own what the stage decides about each record.

    The trainer gives the stage every record that its processor accepts,
    except training instructions, in append order. The trainer then hands each
    formed set to the processor. A record that the stage ends without a set,
    for example a record that can no longer train, appears in
    ``releasable_record_ids``. The trainer adds those IDs to the next commit,
    so the record never trains after a restart.

    The stage keeps only memory that a replay of the retained records
    rebuilds, as processor memory does.
    """

    @abstractmethod
    def ingest(self, record: AgentRecord) -> None:
        """Read one record. The stage must not block on network or model calls."""

    @abstractmethod
    def take_formed(self) -> tuple[RecordSet, ...]:
        """Return the record sets formed since the last call, in the order they formed."""

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
        """Return JSON-safe state that the trainer adds to the processor status."""
        return {}

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Return numeric state that the trainer reports as processor metrics."""
        return {}

    def close(self) -> None:
        """Release resources that the stage owns. Calling it more than one time is safe."""
        return


class PassThroughStage(RecordStage):
    """Hand each record to the processor by itself, in arrival order, and decide nothing."""

    def __init__(self) -> None:
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
