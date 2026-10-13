"""Round requests: the record stage decides when a task-generation round is due, and from which sessions.

Task generation reads a description and, optionally, recorded sessions. Its
record-to-task stage takes a record set: a ``RoundRequest``. ``RoundStage``
forms one when the pipeline looks for a batch (``poll``) and a round is due.
Otherwise it works as a ``ReportStage`` for the reports that the rounds' plays
produce.

A round is due when no round is running and one of these is true:

* the stage has formed no request yet (the first round after a start);
* the last round played no task, so it makes no batch to wait for;
* ``batches_per_round`` batches trained since the last request. A batch that
  the backend dropped as stale trained nothing and does not count.

A source session is a report that does not train and that the recipe's
``RoundGrouping.is_source_report`` accepts, with its inference records. With
``sources_per_round`` above zero, the stage holds source sessions in arrival
order. Each request takes at most ``sources_per_round`` of them; a request
without a waiting session has no sources. The stage keeps a session's records
until the round that uses it finishes, then releases them.
"""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.records_types import AgentRecord
from reef.core.reports import ReportBase
from reef.record2record.reports import ReportContext, ReportGrouping, ReportStage
from reef.record2record.sets import RecordSet
from reef.record2record.stage import SetAssembly


class RoundGrouping(ReportGrouping):
    """The report policy of a task-generation recipe, and which reports are source sessions."""

    def is_source_report(self, context: ReportContext) -> bool:
        """Whether a report that does not train is a recorded session to generate tasks from. The default is no."""
        return False


@dataclass(frozen=True, kw_only=True)
class RoundRequest(RecordSet):
    """A request for one task-generation round, with the source sessions that the round uses."""

    sources: tuple[ReportContext, ...] = ()

    def record_ids(self) -> tuple[str, ...]:
        return tuple(
            record_id
            for context in self.sources
            for record_id in (*context.references, context.report.agent_record_id)
        )


class RoundStage(ReportStage):
    """A report stage that also forms a ``RoundRequest`` each time a task-generation round is due.

    The record-to-task stage starts a round from each request. It returns
    ``READY`` when the round started, and a skip when no round can start (no
    rounds are left, or its worker stopped). After a skip, the stage forms no
    more requests and holds no more source sessions. The pipeline reports
    each finished round with ``round_finished``.
    """

    def __init__(
        self,
        scenario: str,
        report_type: type[ReportBase] | None,
        grouping: RoundGrouping,
        *,
        batches_per_round: int,
        sources_per_round: int = 0,
        exclusive_sources: bool = False,
    ) -> None:
        super().__init__(scenario, report_type, grouping, exclusive_sources=exclusive_sources)
        if batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        if sources_per_round < 0:
            raise ValueError("sources_per_round must be non-negative")
        self.round_grouping = grouping
        self.batches_per_round = batches_per_round
        self.sources_per_round = sources_per_round
        self.request_count = 0
        self.round_running = False
        self.follow_at_once = False
        self.rounds_stopped = False
        # Batches that trained since the last request; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.batch_dropped = False
        self.waiting_sources: list[ReportContext] = []
        self.round_sources: tuple[ReportContext, ...] = ()
        self.formed_request: RoundRequest | None = None

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round finishes.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.round_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    def round_due(self) -> bool:
        if self.rounds_stopped or self.round_running:
            return False
        return self.request_count == 0 or self.follow_at_once or self.trained_batches >= self.batches_per_round

    def poll(self, now: float) -> None:
        # Only a look for a batch forms a request. Thus a restart replays the records before the first round.
        super().poll(now)
        if self.formed_request is not None or not self.round_due():
            return
        self.request_count += 1
        sources = tuple(self.waiting_sources[: self.sources_per_round])
        self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.formed_request = RoundRequest(
            set_id=("round-request", self.request_count), arrival_index=self.request_count, sources=sources
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed: tuple[RecordSet, ...] = super().take_formed()
        if self.formed_request is not None:
            formed, self.formed_request = (*formed, self.formed_request), None
        return formed

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        if not isinstance(record_set, RoundRequest):
            super().accept(record_set, assembly)
            return
        # The round started. The batches that its plays make count toward the next one.
        self.round_running = True
        self.follow_at_once = False
        self.trained_batches = 0
        self.round_sources = record_set.sources
        self.ended_set_ids.add(record_set.set_id)

    def reject(self, record_set: RecordSet) -> None:
        if not isinstance(record_set, RoundRequest):
            super().reject(record_set)
            return
        # No round runs again, so no session waits for one.
        self.rounds_stopped = True
        self.waiting_sources = []
        self.ended_set_ids.add(record_set.set_id)

    def round_finished(self, played_none: bool) -> None:
        """A round finished. A round that played no task makes no batch, so the next one is due at once."""
        self.round_running = False
        self.follow_at_once = played_none
        self.round_sources = ()

    def mark_batch_dropped(self) -> None:
        self.batch_dropped = True

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        if self.batch_dropped:
            self.batch_dropped = False
        else:
            self.trained_batches += 1
        return super().mark_trained(trained_sets)

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in (*self.waiting_sources, *self.round_sources)
            for record_id in (*context.references, context.report.agent_record_id)
        }
        return super().releasable_record_ids() - held

    def operational_metrics(self) -> dict[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}
