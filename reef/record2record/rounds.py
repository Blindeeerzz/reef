"""Round requests: the record stage decides when a task-generation round is due.

Task generation from a description reads no records. Its record-to-task stage
still takes a record set: an empty ``RoundRequest``. ``RoundStage`` forms one
when a round is due and otherwise works as a ``ReportStage`` for the reports
that the rounds' plays produce.

A round is due when no round is running and one of these is true:

* the stage has formed no request yet (the first round after a start);
* the last round played no task, so it makes no batch to wait for;
* ``batches_per_round`` batches trained since the last request. A batch that
  the backend dropped as stale trained nothing and does not count.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase
from reef.record2record.reports import ReportGrouping, ReportStage
from reef.record2record.sets import RecordSet
from reef.record2record.stage import SetAssembly


@dataclass(frozen=True, kw_only=True)
class RoundRequest(RecordSet):
    """A request for one task-generation round. It holds no records."""

    def record_ids(self) -> tuple[str, ...]:
        return ()


class RoundStage(ReportStage):
    """A report stage that also forms a ``RoundRequest`` each time a task-generation round is due.

    The record-to-task stage starts a round from each request. It returns
    ``READY`` when the round started, and a skip when no round can start (no
    rounds are left, or its worker stopped). After a skip, the stage forms no
    more requests. The processor reports each finished round with
    ``round_finished``.
    """

    def __init__(
        self,
        scenario: str,
        report_type: type[ReportBase] | None,
        grouping: ReportGrouping,
        *,
        batches_per_round: int,
        exclusive_sources: bool = False,
    ) -> None:
        super().__init__(scenario, report_type, grouping, exclusive_sources=exclusive_sources)
        if batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        self.batches_per_round = batches_per_round
        self.request_count = 0
        self.round_running = False
        self.follow_at_once = False
        self.rounds_stopped = False
        # Batches that trained since the last request; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.batch_dropped = False

    def round_due(self) -> bool:
        if self.rounds_stopped or self.round_running:
            return False
        return self.request_count == 0 or self.follow_at_once or self.trained_batches >= self.batches_per_round

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed: tuple[RecordSet, ...] = super().take_formed()
        if self.round_due():
            self.request_count += 1
            request = RoundRequest(set_id=("round-request", self.request_count), arrival_index=self.request_count)
            formed = (*formed, request)
        return formed

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        if not isinstance(record_set, RoundRequest):
            super().accept(record_set, assembly)
            return
        # The round started. The batches that its plays make count toward the next one.
        self.round_running = True
        self.follow_at_once = False
        self.trained_batches = 0
        self.ended_set_ids.add(record_set.set_id)

    def reject(self, record_set: RecordSet) -> None:
        if not isinstance(record_set, RoundRequest):
            super().reject(record_set)
            return
        self.rounds_stopped = True
        self.ended_set_ids.add(record_set.set_id)

    def round_finished(self, played_none: bool) -> None:
        """A round finished. A round that played no task makes no batch, so the next one is due at once."""
        self.round_running = False
        self.follow_at_once = played_none

    def mark_batch_dropped(self) -> None:
        self.batch_dropped = True

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        if self.batch_dropped:
            self.batch_dropped = False
        else:
            self.trained_batches += 1
        return super().mark_trained(trained_sets)
