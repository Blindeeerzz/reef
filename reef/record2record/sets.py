"""Record sets: the records that the record-to-record stage hands to a processor together."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.records_types import AgentRecord
from reef.record2record.selection import ExperienceUnit


@dataclass(frozen=True, kw_only=True)
class RecordSet(ExperienceUnit, ABC):
    """Records that are used together, such as one record or a report with the records it references.

    A record set is an ``ExperienceUnit``, so the buffer, the selection
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
