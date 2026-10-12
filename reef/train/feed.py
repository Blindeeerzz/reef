"""The trainer's record path: records through the record-to-record stage into the processor."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.record2record.stage import Record2RecordStage
from reef.train.processors.base import DataProcessor


class TrainingFeed:
    """Move one trainer's records through its record-to-record stage into its processor.

    Training instructions go straight to the processor. Every other record
    goes to the stage, and each record set that the stage forms goes to
    ``processor.assemble``. The feed also merges what the stage and the
    processor report: records to release, background work, status, and
    metrics. The trainer calls the feed under its lock.
    """

    def __init__(self, stage: Record2RecordStage, processor: DataProcessor) -> None:
        self.stage = stage
        self.processor = processor

    def ingest(self, record: AgentRecord) -> None:
        """Read one record that the processor accepts."""
        if record.request_type is RequestType.TRAIN:
            self.processor.ingest(record)
            return
        self.stage.ingest(record)
        self.assemble_formed()

    def assemble_formed(self) -> None:
        """Hand every newly formed record set to the processor."""
        for record_set in self.stage.take_formed():
            self.processor.assemble(record_set)

    def ready(self) -> bool:
        """Whether the processor can build a batch now."""
        self.assemble_formed()
        return self.processor.ready()

    def releasable_record_ids(self) -> frozenset[str]:
        return self.processor.releasable_record_ids() | self.stage.releasable_record_ids()

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.processor.release_records(agent_record_ids)
        self.stage.release_records(agent_record_ids)

    def derivation_pending(self) -> bool:
        return self.processor.derivation_pending() or self.stage.derivation_pending()

    def status(self) -> Mapping[str, Any]:
        return {**self.processor.status(), **self.stage.status()}

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {**self.processor.operational_metrics(), **self.stage.operational_metrics()}

    def close(self) -> None:
        """Close the stage, then the processor, even when the stage fails to close."""
        try:
            self.stage.close()
        finally:
            self.processor.close()
