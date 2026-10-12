"""Computed-feedback engine: judge records asynchronously, batch the results.

The sibling of :class:`~reef.train.processors.reported.ReportedFeedbackProcessor`
for the other way feedback arrives. A reported-feedback recipe is handed
explicit reports; a computed-feedback recipe produces its signal from the traffic itself — a
record only becomes judgeable when some LATER record completes it, and
the judgment itself calls models, so it runs asynchronously on a private
worker. This engine owns everything between those facts and the trainer:
the record lifecycle, the judging worker, the batch cycle, buffer release, and
crash-replay semantics. A recipe subclasses it and writes four methods:

* ``ingest(record)`` — the method's own correlation, written with the
  engine's verbs: start with ``catch_up``, ``dispatch`` each completed
  record's job, then ``track`` the record or ``retire`` it;
* ``judge(job)`` — an async coroutine producing the recipe's judgment,
  run on the worker's thread;
* ``make_sample(record, judgment)`` — one judged record into a
  :class:`TrajectoryItem`, or ``None`` for a record that cannot train;
* ``make_batch(samples, batch_number)`` — the selected samples into the
  recipe's batch type.

The ingest stays recipe code on purpose: for a computed-feedback method the
correlation IS the method, so its whole line reads in the recipe file.

Everything here is reconstructible state: after a crash the trainer replays the
un-acknowledged tail and judging re-runs, recompute bounded by the
training backlog; a record whose judgment was in flight is lost — one
record's exposure.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.records_types import AgentRecord
from reef.record2record import BatchUnit, StalenessCheck, UnitBuffer
from reef.record2record.worker import Failed, JobRunner, JudgingWorker, SupportsReceipt
from reef.train.processors.base import DataProcessor
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

# The worker types moved to reef.record2record.worker. Recipes can still import them from here.
__all__ = [
    "CandidateUnit",
    "ComputedFeedbackProcessor",
    "Failed",
    "JudgingWorker",
    "SupportsReceipt",
    "candidate_units",
]


@dataclass(frozen=True, kw_only=True)
class CandidateUnit(BatchUnit):
    """A judged record's sample in the unit buffer; ``unit_id`` is the record's receipt."""

    receipt: str
    sample: TrajectoryItem


def candidate_units(units: Sequence[BatchUnit]) -> tuple[CandidateUnit, ...]:
    """Return buffer units as candidate units. The computed-feedback engine puts only candidate units in its buffer."""
    selected = tuple(unit for unit in units if isinstance(unit, CandidateUnit))
    if len(selected) != len(units):
        raise TypeError("the computed-feedback unit buffer must hold only candidate units")
    return selected


class ComputedFeedbackProcessor(DataProcessor, JobRunner, ABC):
    """Record lifecycle and batch cycle for computed-feedback recipes.

    Every ingested receipt sits in exactly one state: tracked
    (awaiting future traffic) → in-flight (judging) → candidate | terminal
    | trained. Terminal records are released from memory instead of
    ever entering a batch. By default, only candidates within
    ``max_staleness`` versions of the newest runtime load ID (by record
    arrival, not judgment arrival) batch: continual serving advances the
    version every step, and stale pending candidates would deadlock the FIFO.

    Candidates wait in a :class:`~reef.record2record.UnitBuffer` in
    the order that their judgments arrived. A ``StalenessCheck`` with the
    context's ``max_staleness`` drops candidates while no batch is out.

    The line, in reading order: the recipe's ``ingest`` (catch_up →
    dispatch → track), its ``judge`` on the worker, then here —
    ``_collect_judgments`` absorbs what landed, ``build_batch`` calls the
    recipe's ``make_batch``, ``acknowledge`` releases.
    """

    def __init__(self, context: ProcessorContext, *, worker: JudgingWorker | None) -> None:
        super().__init__(context)
        #: No worker: records track and expire, nothing is ever judged,
        #: everything can be released.
        self._worker = worker

        # Every ingested receipt is in exactly one state.
        self._tracked: dict[str, AgentRecord] = {}
        self._in_flight: dict[str, AgentRecord] = {}
        #: Candidates by receipt, and the reserved batch.
        self.unit_buffer = UnitBuffer(checks=(StalenessCheck(context.max_staleness),))
        #: Terminal receipts: retired or trained, both releasable and never
        #: distinguished by anything that reads them.
        self._terminal: set[str] = set()
        # Record-arrival order per live receipt: the stale-version policy
        # needs generation recency, which judgment completion order is not.
        self._arrival_order: dict[str, int] = {}
        self._next_order = 0

    # ------------------------------------------------------- the recipe hooks

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {
            **super().operational_metrics(),
            "tracked_records": len(self._tracked),
            "judging_records": len(self._in_flight),
            "unreserved_candidates": len(self.unit_buffer) - len(self.unit_buffer.reserved_units()),
            "reserved_candidates": len(self.unit_buffer.reserved_units()),
        }

    @abstractmethod
    def ingest(self, item: AgentRecord) -> None:
        """The method's own correlation, one record at a time, written with
        the engine's verbs (``catch_up`` first, then ``dispatch`` /
        ``track`` / ``retire``)."""

    @abstractmethod
    async def judge(self, job: Any) -> SupportsReceipt:
        """The recipe's judgment for one job.

        Awaited on the worker thread after your ``ingest`` dispatched the
        job — never on the trainer's — so it may call models and take
        minutes. It carries the ``receipt`` of the record it judges; that
        is all the engine reads of it.
        """

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker: the recipe's ``judge``."""
        return await self.judge(job)

    @abstractmethod
    def make_sample(self, record: AgentRecord, judgment: Any) -> TrajectoryItem | None:
        """One judged record into a sample; ``None`` cannot train."""

    @abstractmethod
    def make_batch(self, samples: tuple[TrajectoryItem, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected samples into this recipe's batch type."""

    def expire(self, now: float) -> tuple[str, ...]:
        """Tracked receipts whose wait for future traffic is over; they
        retire terminal. The default recipe never expires anything."""
        return ()

    # ------------------------------------------- the engine's verbs for ingest

    def catch_up(self, now: float) -> None:
        """Retire expired records, absorb finished judgments. Every ingest
        starts here — expiry runs before the record is read, so an expired
        context can never capture it."""
        for receipt in self.expire(now):
            self.abandon(receipt)
        self._collect_judgments()

    def track(self, item: AgentRecord) -> None:
        """This record awaits future traffic."""
        self._next_order += 1
        self._arrival_order[item.agent_record_id] = self._next_order
        self._tracked[item.agent_record_id] = item

    def tracked_record(self, receipt: str) -> AgentRecord | None:
        """The record still awaiting completion under ``receipt``, if any."""
        return self._tracked.get(receipt)

    def dispatch(self, job: SupportsReceipt) -> None:
        """A tracked record was completed: hand its judgment to the worker.
        ``job.receipt`` names the tracked record it judges."""
        record = self._tracked.pop(job.receipt, None)
        if record is None:
            return
        if self._worker is not None and self._worker.submit(job):
            self._in_flight[job.receipt] = record
        else:
            self.retire(job.receipt)

    def retire(self, receipt: str) -> None:
        """This receipt is terminal: released from memory, never batched."""
        self._terminal.add(receipt)
        self._arrival_order.pop(receipt, None)

    def abandon(self, receipt: str) -> None:
        """A tracked record's future never came: stop awaiting it, retire it.
        (``retire`` alone would leave the record protected as tracked.)"""
        self._tracked.pop(receipt, None)
        self.retire(receipt)

    def _collect_judgments(self) -> None:
        """Absorb judgments the worker finished since the last entry."""
        if self._worker is None:
            return
        for judgment in self._worker.poll():
            # 1. Only records still in flight resolve; anything else was
            #    already released or closed.
            record = self._in_flight.pop(judgment.receipt, None)
            if record is None:
                continue
            # 2. A failed or declined judgment retires the record; so does a
            #    sample the recipe cannot train on — never a failed training
            #    step.
            sample = None if isinstance(judgment, Failed) else self.make_sample(record, judgment)
            if sample is None:
                self.retire(judgment.receipt)
                continue
            # 3. The record is a batch candidate now. The eligibility checks
            #    drop candidates (by default, those too far behind the newest
            #    weight version by record arrival) — but never under an
            #    emitted-but-unacknowledged batch, whose samples reference the
            #    pending candidates.
            buffer = self.unit_buffer
            receipt = judgment.receipt
            buffer.put(
                CandidateUnit(
                    unit_id=receipt,
                    arrival_index=buffer.next_arrival_index(),
                    runtime_load_id=sample.training.get("runtime_load_id"),
                    source_index=self._arrival_order[receipt],
                    receipt=receipt,
                    sample=sample,
                )
            )
            if self._pending is None:
                for unit in candidate_units(tuple(dropped.unit for dropped in buffer.drop_ineligible())):
                    self.retire(unit.receipt)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one judged record's sample; the engine's half of the shared
    # cycle in base.py is the three methods below. ``ready`` also absorbs
    # whatever the worker finished, since judgments land without a record
    # ever arriving to trigger it.

    def ready(self) -> bool:
        self.catch_up(time.monotonic())
        return super().ready()

    def _ready_count(self) -> int:
        return len(self.unit_buffer)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        units = candidate_units(self.unit_buffer.reserve(self._batch_size))
        return self.make_batch(tuple(unit.sample for unit in units), batch_number)

    def _consume_pending(self) -> frozenset[str]:
        consumed = frozenset(unit.receipt for unit in candidate_units(self.unit_buffer.consume_reserved()))
        for receipt in consumed:
            self._arrival_order.pop(receipt, None)
            self._terminal.add(receipt)
        return consumed

    # -------------------------------------------------------- background work

    def derivation_pending(self) -> bool:
        """In-flight judgments resolve and tracked records expire without a
        new record ever waking the training worker; poll while either exists."""
        return bool(self._in_flight or self._tracked)

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    # ---------------------------------------------------------- buffer release

    def releasable_record_ids(self) -> frozenset[str]:
        candidates = frozenset(unit.receipt for unit in candidate_units(self.unit_buffer.units()))
        protected = frozenset(self._tracked) | frozenset(self._in_flight) | candidates
        return frozenset(self._terminal) - protected

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self._terminal -= agent_record_ids
        for agent_record_id in agent_record_ids:
            self._tracked.pop(agent_record_id, None)
            self._in_flight.pop(agent_record_id, None)
            self.unit_buffer.remove(agent_record_id)
            self._arrival_order.pop(agent_record_id, None)
