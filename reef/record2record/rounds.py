"""Rounds: the record stage decides when a task-generation round is due, and from which sessions.

Task generation reads a description and, optionally, recorded sessions.
``RoundRecord2Record`` paces the rounds and holds the source sessions for
them. Otherwise it works as a ``ReportRecord2Record`` for the reports that
the rounds' plays produce.

A round is not a record set and does not train. When a round is due, the
stage makes a ``TaskGenerationRound`` and gives it to its ``RoundRunner``, the
record-to-task stage of generated tasks. The runner returns a ``RoundResult``
when the round ends.

A round is due when no round is running, the round limit is not reached, and
one of these is true:

* no round has started yet (the first round after a start);
* the last round ended with ``wait_for_batches`` False, for example a round
  that played no task, so it makes no batch to wait for;
* ``batches_per_round`` batches trained since the last round started. A batch
  that the backend dropped as stale does not count.

A source session is a report that does not train and that the recipe's
``RoundGrouping.is_source_report`` accepts, with its inference records. With
``sources_per_round`` above zero, the stage holds source sessions in arrival
order. Each round takes at most ``sources_per_round`` of them; a round
without a waiting session has no sources. The stage holds the records of a
round's sources until the round ends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.records_types import AgentRecord
from reef.pipeline_context import PipelineContext
from reef.record2record.reports import ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import BatchUnit


class RoundGrouping(ReportGrouping):
    """The report policy of a task-generation recipe, and which reports are source sessions."""

    def is_source_report(self, context: ReportContext) -> bool:
        """Whether a report that does not train is a recorded session to generate tasks from. The default is no."""
        return False


@dataclass(frozen=True)
class TaskGenerationRound:
    """One task-generation round.

    Values: ``round_number`` is the number of the round. Rounds of one stage
    have the numbers 0, 1, 2, and so on, in order. ``sources`` holds at most
    ``sources_per_round`` source sessions, in arrival order.
    Purpose: the round stage makes it when a round is due. A ``RoundRunner`` runs it.
    """

    round_number: int
    sources: tuple[ReportContext, ...]

    def record_ids(self) -> frozenset[str]:
        """Return the IDs of the records of the sources, which the stage holds while the round runs."""
        return frozenset(
            record_id for source in self.sources for record_id in (*source.references, source.report.agent_record_id)
        )


@dataclass(frozen=True)
class RoundResult:
    """How a round ended.

    Values: ``wait_for_batches`` is True when the next round waits for
    ``batches_per_round`` trained batches. It is False when the next round can
    start at once, because the round made no batch.
    """

    wait_for_batches: bool


class RoundRunner(ABC):
    """Runs the task-generation rounds of a round stage, one at a time."""

    @abstractmethod
    def start_round(self, generation_round: TaskGenerationRound) -> bool:
        """Start ``generation_round`` on a worker. Return False when no worker can run it."""

    @abstractmethod
    def finished_rounds(self) -> tuple[RoundResult, ...]:
        """Return the results of the rounds that ended since the last call. The stage calls it in ``advance``."""


class RoundRecord2Record(ReportRecord2Record):
    """A report stage that paces task-generation rounds and holds their source sessions.

    The stage starts no round until ``resume_rounds`` gives it a runner, the
    first round number and the round limit. At the limit, the stage stops the
    rounds and holds no more source sessions. The subclass's settings check
    that ``batches_per_round`` is positive and ``sources_per_round`` is not negative.
    """

    def __init__(
        self,
        context: PipelineContext,
        grouping: RoundGrouping,
        *,
        batches_per_round: int,
        sources_per_round: int = 0,
        exclusive_sources: bool = False,
    ) -> None:
        super().__init__(context, grouping, exclusive_sources=exclusive_sources)
        self.round_grouping = grouping
        self.batches_per_round = batches_per_round
        self.sources_per_round = sources_per_round
        self.runner: RoundRunner | None = None
        self.next_round = 0
        self.round_limit = 0
        self.running_round: TaskGenerationRound | None = None
        # The first round after a start is due at once.
        self.follow_at_once = True
        self.rounds_stopped = False
        # Batches that trained since the last round started; the next round is due at batches_per_round.
        self.trained_batches = 0
        self.waiting_sources: list[ReportContext] = []

    def resume_rounds(self, runner: RoundRunner, *, next_round: int, round_limit: int) -> None:
        """Set the runner, the next round number and the round limit. After a restart, the next round follows the recorded ones."""
        self.runner = runner
        self.next_round = next_round
        self.round_limit = round_limit

    def handle_other_report(self, report: AgentRecord) -> None:
        # The report ends here. releasable_record_ids holds back a session's records until its round ends.
        super().handle_other_report(report)
        if self.sources_per_round == 0 or self.rounds_stopped:
            return
        if any(reference not in self.inferences for reference in report.references):
            return
        context = self.report_context(report, None)
        if self.round_grouping.is_source_report(context):
            self.waiting_sources.append(context)

    # ---------------------------------------------------------------- rounds

    def advance(self, now: float) -> None:
        """End the rounds that finished, then start the next round when it is due, or stop the rounds at the limit."""
        super().advance(now)
        if self.runner is None:
            return
        for result in self.runner.finished_rounds():
            self.running_round = None
            self.follow_at_once = not result.wait_for_batches
        if self.rounds_stopped or self.running_round is not None:
            return
        if not self.follow_at_once and self.trained_batches < self.batches_per_round:
            return
        if self.next_round >= self.round_limit:
            self.stop_rounds()
            return
        sources = tuple(self.waiting_sources[: self.sources_per_round])
        generation_round = TaskGenerationRound(round_number=self.next_round, sources=sources)
        if not self.runner.start_round(generation_round):
            self.stop_rounds()
            return
        self.waiting_sources = self.waiting_sources[self.sources_per_round :]
        self.running_round = generation_round
        self.next_round += 1
        self.follow_at_once = False
        # The batches that the round's plays make count toward the next one.
        self.trained_batches = 0

    def stop_rounds(self) -> None:
        """No round runs again, so no session waits for one."""
        self.rounds_stopped = True
        self.waiting_sources = []

    def consume(self, units: tuple[BatchUnit, ...], *, trained: bool) -> frozenset[str]:
        if trained:
            self.trained_batches += 1
        return super().consume(units, trained=trained)

    def releasable_record_ids(self) -> frozenset[str]:
        held = {
            record_id
            for context in self.waiting_sources
            for record_id in (*context.references, context.report.agent_record_id)
        }
        if self.running_round is not None:
            held |= self.running_round.record_ids()
        return super().releasable_record_ids() - held

    def background_jobs_running(self) -> bool:
        """Return True while a round runs: its end can start the next round without a new record."""
        return self.running_round is not None or super().background_jobs_running()

    def operational_metrics(self) -> dict[str, float | int]:
        return {**super().operational_metrics(), "waiting_source_sessions": len(self.waiting_sources)}
