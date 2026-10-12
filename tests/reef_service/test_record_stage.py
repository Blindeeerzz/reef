"""Record-to-record stage contracts: what the trainer's feed sends where, and what a commit records."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from reef.core import AgentRecord, RequestType
from reef.record2record.sets import RecordSet, SingleRecordSet
from reef.record2record.stage import PassThroughStage, Record2RecordStage
from reef.storage.sqlite import SQLiteRecordStore
from reef.train import ProcessorContext, Trainer
from reef.train.feed import TrainingFeed
from reef.train.processors import DataProcessor

from ._threshold_processor import ThresholdProcessor
from .test_record_stage_batch_parity import CountingBackend, inference, instruction, report


class RecordingProcessor(DataProcessor):
    """Keeps every record it ingests; never becomes ready."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        self.ingested: list[str] = []
        self.closed = False

    def ingest(self, item: AgentRecord) -> None:
        super().ingest(item)
        self.ingested.append(item.agent_record_id)

    def close(self) -> None:
        self.closed = True


class WithholdingStage(Record2RecordStage):
    """Passes records through, except records whose ID starts with ``drop``: it ends those itself."""

    def __init__(self) -> None:
        self.formed: list[RecordSet] = []
        self.ended: set[str] = set()
        self.released: set[str] = set()
        self.pending = False
        self.fail_close = False

    def ingest(self, record: AgentRecord) -> None:
        if record.agent_record_id.startswith("drop"):
            self.ended.add(record.agent_record_id)
            return
        self.formed.append(SingleRecordSet(unit_id=record.agent_record_id, arrival_index=0, record=record))

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed, self.formed = tuple(self.formed), []
        return formed

    def releasable_record_ids(self) -> frozenset[str]:
        return frozenset(self.ended)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.released |= agent_record_ids
        self.ended -= agent_record_ids

    def derivation_pending(self) -> bool:
        return self.pending

    def status(self) -> Mapping[str, Any]:
        return {"ended_records": len(self.ended)}

    def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("stage close failed")


def feed(stage: Record2RecordStage) -> tuple[TrainingFeed, RecordingProcessor]:
    processor = RecordingProcessor(ProcessorContext("s", training_mode="hybrid"))
    return TrainingFeed(stage, processor), processor


@pytest.mark.unit
def test_the_pass_through_stage_forms_one_set_per_record_in_arrival_order() -> None:
    stage = PassThroughStage()
    for record_id in ("i1", "r1", "i2"):
        stage.ingest(inference(record_id))
    formed = stage.take_formed()
    assert [record_set.record_ids() for record_set in formed] == [("i1",), ("r1",), ("i2",)]
    assert [record_set.arrival_index for record_set in formed] == [1, 2, 3]
    assert stage.take_formed() == ()
    assert stage.releasable_record_ids() == frozenset() and not stage.derivation_pending()


@pytest.mark.unit
def test_the_feed_sends_instructions_to_the_processor_and_other_records_through_the_stage() -> None:
    stage = WithholdingStage()
    training_feed, processor = feed(stage)
    training_feed.ingest(inference("i1"))
    training_feed.ingest(inference("drop-1"))
    training_feed.ingest(instruction("train-1"))
    assert processor.ingested == ["i1", "train-1"]
    assert processor.buffered_requests() == 1
    assert training_feed.releasable_record_ids() == {"drop-1"}


@pytest.mark.unit
def test_the_feed_merges_background_work_and_status_and_closes_the_processor_after_a_failed_stage() -> None:
    stage = WithholdingStage()
    training_feed, processor = feed(stage)
    assert not training_feed.derivation_pending()
    stage.pending = True
    assert training_feed.derivation_pending()
    assert training_feed.status() == {"ready_units": 0, "batch_size": 1, "buffered_requests": 0, "ended_records": 0}
    stage.fail_close = True
    with pytest.raises(RuntimeError, match="stage close failed"):
        training_feed.close()
    assert processor.closed


@pytest.mark.unit
def test_a_record_the_stage_ends_is_consumed_by_the_next_commit_and_then_released() -> None:
    records = SQLiteRecordStore()
    for record in (inference("i1"), inference("drop-1"), report("r1", "i1")):
        records.append(record)
    stage = WithholdingStage()
    trainer = Trainer.build(
        "s",
        records,
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        candidate_backend=CountingBackend(),
        record_stage=stage,
    )
    batch = trainer.reserve_training_batch()
    assert batch is not None
    execution = trainer.execute_reserved_step(0)
    assert execution.result is not None
    prepared = trainer.prepare_commit(execution.result)
    assert prepared.consumed_ids == {"i1", "r1", "drop-1"}
    trainer.commit(prepared)
    assert stage.released == {"i1", "r1", "drop-1"} and not stage.ended
    assert "drop-1" in trainer.consumed_record_ids


@pytest.mark.unit
def test_the_trainer_reports_the_stages_background_work() -> None:
    stage = WithholdingStage()
    trainer = Trainer.build(
        "s",
        SQLiteRecordStore(),
        processor_factory=lambda context: ThresholdProcessor(context.with_config({"batch_size": 1})),
        record_stage=stage,
    )
    assert not trainer.derivation_pending()
    stage.pending = True
    assert trainer.derivation_pending()


@pytest.mark.unit
def test_a_processor_built_with_the_wrong_mode_closes_the_stage_too() -> None:
    closed: list[str] = []

    class ClosingStage(PassThroughStage):
        def close(self) -> None:
            closed.append("stage")

    with pytest.raises(ValueError, match="training_mode"):
        Trainer.build(
            "s",
            SQLiteRecordStore(),
            processor_factory=lambda context: ThresholdProcessor(ProcessorContext("s")),
            training_mode="manual",
            record_stage=ClosingStage(),
        )
    assert closed == ["stage"]


@pytest.mark.unit
def test_the_default_processor_assembly_takes_only_single_record_sets() -> None:
    class Pair(RecordSet):
        def record_ids(self) -> tuple[str, ...]:
            return ("a", "b")

    processor = RecordingProcessor(ProcessorContext("s"))
    with pytest.raises(TypeError, match="single-record sets"):
        processor.assemble(Pair(unit_id="pair", arrival_index=1))
    record = AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload={}, agent_record_id="i1")
    processor.assemble(SingleRecordSet(unit_id="i1", arrival_index=1, record=record))
    assert processor.ingested == ["i1"]
