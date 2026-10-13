"""Report sets: reports and the inference records that they reference, alone or in groups.

``ReportStage`` reads inference and report records. It validates each report,
filters the reports that are not training data, and forms one
``ReportCandidate`` per report for the processor to assemble. It then places
each assembled report in its group (``ReportGrouping``) and holds singleton
reports and ready groups in its pool. It owns deduplication, group slots,
consumption, and record release.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, validate_report_payload
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder
from reef.record2record.sets import RecordSet
from reef.record2record.stage import Record2RecordStage, SetAssembly

logger = logging.getLogger(__name__)


class GroupDecision(Enum):
    """The decision for a report group: ready, incomplete, or invalid."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    DISCARD = "discard"


@dataclass(frozen=True)
class ReportContext:
    """A valid report and its inference records, in reference order."""

    report: AgentRecord
    score: float | None
    inferences: tuple[AgentRecord, ...]
    parsed_report: ReportBase | None = None

    @property
    def references(self) -> tuple[str, ...]:
        return self.report.references

    def require_score(self) -> float:
        """Read a finite reward for a training method that needs one."""
        if self.score is None or not math.isfinite(self.score):
            raise ValueError("training data requires a finite report score")
        return self.score


def report_score(report: AgentRecord) -> float | None:
    """Return the report's numeric score, or ``None`` when absent or non-numeric.

    Bool is not a score. Callers layer their own selection policy (windows,
    finiteness, minimums) on top of this one shared reading of the payload.
    """
    score = report.payload.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    return float(score)


def accepted_by(report_type: type[ReportBase], payload: Mapping[str, Any]) -> bool:
    """Whether ``report_type`` parses ``payload``."""
    try:
        report_type.from_dict(payload)
    except ReportValidationError:
        return False
    return True


class ReportGrouping:
    """A recipe's report policy: which reports train, and how reports form groups.

    The default takes every report and trains each one by itself. A grouped
    method overrides ``placement`` and ``decide``. These methods run on the
    trainer thread and read only records.
    """

    def is_training_report(self, report: AgentRecord) -> bool:
        """Whether a valid report is this method's training data.

        A report that is not, such as another role's signal sharing the
        scenario, is released and never assembled; nothing about it can fail
        ingestion. Its sources are released with it only under ``exclusive_sources``
        or when it references more than one inference; a single referenced
        inference stays retained for a report that trains on it.
        """
        return True

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        """Return the batching group and retry slot of an assembled report. The default is an independent report.

        These coordinates belong to ingestion, separately from an item's
        comparison group. For example, TTTD batches a whole step containing
        several comparison groups. A None slot uses the report ID.
        """
        return None, None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        """Decide whether the group's reports, in arrival order, are ready, incomplete, or invalid."""
        raise NotImplementedError(f"{type(self).__name__} placed a report in a group without a decide override")

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe group outcomes, such as groups that can never train."""
        return {}


@dataclass(frozen=True, kw_only=True)
class ReportCandidate(RecordSet):
    """One valid report and its inference records, before the processor assembles it. ``set_id`` is the report ID."""

    context: ReportContext

    def record_ids(self) -> tuple[str, ...]:
        return (*self.context.references, self.context.report.agent_record_id)


@dataclass(frozen=True)
class ReportEntry:
    """One assembled report in a group slot, or by itself."""

    order: int
    context: ReportContext
    group_key: Hashable | None
    slot: Hashable

    @property
    def report_id(self) -> str:
        return self.context.report.agent_record_id


@dataclass(frozen=True, kw_only=True)
class ReportSet(RecordSet):
    """A singleton report or a ready group, with its reports in arrival order."""

    entries: tuple[ReportEntry, ...]

    def record_ids(self) -> tuple[str, ...]:
        return tuple(record_id for entry in self.entries for record_id in (*entry.context.references, entry.report_id))


def report_sets(pool_sets: tuple[RecordSet, ...]) -> tuple[ReportSet, ...]:
    """Return pool sets as report sets. A report stage puts only report sets in its pool."""
    selected = tuple(record_set for record_set in pool_sets if isinstance(record_set, ReportSet))
    if len(selected) != len(pool_sets):
        raise TypeError("the report stage's pool must hold only report sets")
    return selected


class ReportStage(Record2RecordStage):
    """Form report sets, and hold singleton reports and ready groups until a batch uses them.

    Invalid references raise immediately. Deduplication, consumed-source
    tracking, and group slots preserve retry behavior. Release keeps every
    live report and the inference records that it references.

    * ``exclusive_sources``: a terminal report owns its referenced inference records outright.
    * ``ordered_groups``: batch ready groups in group-key order instead of arrival order.
    """

    def __init__(
        self,
        scenario: str,
        report_type: type[ReportBase] | None,
        grouping: ReportGrouping,
        *,
        exclusive_sources: bool = False,
        ordered_groups: bool = False,
    ) -> None:
        super().__init__(GroupKeyOrder() if ordered_groups else ArrivalOrder())
        self.scenario = scenario
        self.report_type = report_type
        self.admitted_report_type: type[ReportBase] | None = None
        self.grouping = grouping
        self.exclusive_sources = exclusive_sources
        self.formed: list[ReportCandidate] = []
        self.candidate_count = 0

        # --- inference store ---
        self.inferences: dict[str, AgentRecord] = {}

        # --- report lifecycle ---
        self.reports: dict[str, AgentRecord] = {}  # live: assembled or failed, not consumed
        self.seen_reports: set[str] = set()  # dedup
        self.consumed: set[str] = set()  # acknowledged batch members
        self.terminal: set[str] = set()  # terminal reports

        # --- source ownership ---
        self.trained_sources: set[str] = set()  # consumed by an acknowledged batch
        self.terminal_owned_sources: set[str] = set()  # owned by terminal reports

        # --- groups ---
        self.groups: dict[Hashable, dict[Hashable, ReportEntry]] = {}  # group key → slot → report
        self.discarded_groups: set[Hashable] = set()
        self.available_limit: int | None = None
        self.limit_warned = False

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        self.admitted_report_type = report_type

    # ------------------------------------------------------------- formation

    def ingest(self, record: AgentRecord) -> None:
        if record.scenario != self.scenario:
            raise ReportValidationError("records must belong to the processor's scenario")
        if record.request_type is RequestType.INFERENCE:
            self.inferences[record.agent_record_id] = record
            return
        if record.request_type is not RequestType.REPORT or record.agent_record_id in self.seen_reports:
            return
        validate_report_payload(record.payload)
        if not record.references or len(set(record.references)) != len(record.references):
            raise ReportValidationError("report references must be non-empty and unique")
        if any(ref in self.trained_sources for ref in record.references) or not self.grouping.is_training_report(
            record
        ):
            self.seen_reports.add(record.agent_record_id)
            self.terminate(record)
            return
        parsed_report: ReportBase | None = None
        if self.report_type is not None:
            try:
                parsed_report = self.report_type.from_dict(record.payload)
            except ReportValidationError as refusal:
                # A scenario of several components admits what any of them accepts: a report the ingress
                # contract takes and this one refuses is another component's, not this method's training
                # data, and this trainer releases it. A report every component refuses still raises.
                admitted = self.admitted_report_type
                if admitted is None or admitted is self.report_type or not accepted_by(admitted, record.payload):
                    raise
                # Named in the log: a report meant for this trainer with a broken field also lands here.
                logger.warning(
                    "scenario %r releases report %s to the other components: %s refused it: %s",
                    self.scenario,
                    record.agent_record_id,
                    self.report_type.__name__,
                    refusal,
                )
                self.seen_reports.add(record.agent_record_id)
                self.terminate(record)
                return
        context = self.report_context(record, parsed_report)
        # Retain the report before assembly: a contract failure must not let
        # buffer release drop its inputs or turn a retry into a successful no-op.
        self.reports[record.agent_record_id] = record
        self.candidate_count += 1
        self.formed.append(
            ReportCandidate(set_id=record.agent_record_id, arrival_index=self.candidate_count, context=context)
        )

    def take_formed(self) -> tuple[RecordSet, ...]:
        formed: tuple[RecordSet, ...] = tuple(self.formed)
        self.formed = []
        return formed

    def report_context(self, report: AgentRecord, parsed_report: ReportBase | None) -> ReportContext:
        missing = [ref for ref in report.references if ref not in self.inferences]
        if missing:
            raise ReportValidationError(f"report references unavailable inference records: {missing!r}")
        inferences = tuple(self.inferences[ref] for ref in report.references)
        return ReportContext(report, report_score(report), inferences, parsed_report)

    def accept(self, record_set: RecordSet, assembly: SetAssembly) -> None:
        context = candidate_context(record_set)
        report = context.report
        key, slot = self.grouping.placement(context)
        slot = report.agent_record_id if slot is None else slot
        hash(key)
        hash(slot)
        self.seen_reports.add(report.agent_record_id)
        if key is not None and (key in self.discarded_groups or slot in self.groups.get(key, {})):
            self.terminate(report)
            self.ended_set_ids.add(report.agent_record_id)
            return
        entry = ReportEntry(order=self.pool.next_arrival_index(), context=context, group_key=key, slot=slot)
        if key is None:
            self.pool.put(
                ReportSet(set_id=("report", report.agent_record_id), arrival_index=entry.order, entries=(entry,))
            )
        else:
            self.groups.setdefault(key, {})[slot] = entry
            self.refresh_group(key)
        self.apply_limit()

    def reject(self, record_set: RecordSet) -> None:
        report = candidate_context(record_set).report
        self.seen_reports.add(report.agent_record_id)
        self.terminate(report)
        self.ended_set_ids.add(report.agent_record_id)

    def terminate(self, report: AgentRecord) -> None:
        """Drop a report from the live set and mark it terminal.

        Terminal reports release their own record, and the sources they own
        outright: a report that claims more than one inference, or any report
        under ``exclusive_sources``.
        """
        report_id = report.agent_record_id
        self.reports.pop(report_id, None)
        self.terminal.add(report_id)
        if not report.references:
            return
        if self.exclusive_sources or len(report.references) > 1:
            self.terminal_owned_sources.update(report.references)

    # ---------------------------------------------------------------- groups

    def group_entries(self, key: Hashable) -> tuple[ReportEntry, ...]:
        return tuple(sorted(self.groups[key].values(), key=lambda entry: entry.order))

    def refresh_group(self, key: Hashable) -> None:
        entries = self.group_entries(key)
        decision = self.grouping.decide(key, tuple(entry.context for entry in entries))
        if decision is GroupDecision.READY:
            self.pool.put(
                ReportSet(set_id=("group", key), arrival_index=entries[0].order, group_key=key, entries=entries)
            )
        elif decision is GroupDecision.INCOMPLETE:
            self.pool.remove(("group", key))
        elif decision is GroupDecision.DISCARD:
            self.discard_group(key)
        else:
            raise TypeError("decide must return GroupDecision")

    def discard_group(self, key: Hashable) -> None:
        self.pool.remove(("group", key))
        self.discarded_groups.add(key)
        group = self.groups.pop(key)
        # Remove every member from the live set first, so the wholesale
        # release is not blocked by siblings of the same discarded group.
        members = [
            report for entry in group.values() if (report := self.reports.pop(entry.report_id, None)) is not None
        ]
        for report in members:
            self.terminate(report)
        self.ended_set_ids.update(entry.report_id for entry in group.values())

    # ------------------------------------------------------- available limit

    def limit_available(self, max_set_count: int | None) -> None:
        self.available_limit = max_set_count
        self.apply_limit()

    def apply_limit(self) -> None:
        """Release the oldest unreserved sets beyond the limit. The reserved batch stays."""
        limit = self.available_limit
        if limit is None or len(self.pool) <= limit:
            return
        reserved = self.reserved_report_ids()
        for report_set in report_sets(self.ordered_sets()):
            if len(self.pool) <= limit:
                return
            if report_set.entries[0].report_id in reserved:
                continue
            self.release_set(report_set)
            if not self.limit_warned:
                logger.warning(
                    "scenario %r released report %s beyond the manual limit %d (further releases are not logged)",
                    self.scenario,
                    report_set.entries[0].report_id,
                    limit,
                )
                self.limit_warned = True

    def ordered_sets(self) -> tuple[RecordSet, ...]:
        return self.pool.ordered_sets()

    def release_set(self, report_set: ReportSet) -> None:
        """Release a whole held group or singleton and its sources."""
        if report_set.group_key is not None:
            self.discard_group(report_set.group_key)
        else:
            self.pool.remove(report_set.set_id)
        for entry in report_set.entries:
            report = self.reports.pop(entry.report_id, None)
            if report is not None:
                self.terminate(report)
            self.terminal_owned_sources.update(entry.context.references)
            self.ended_set_ids.add(entry.report_id)

    # ------------------------------------------------------------ consumption

    def reserved_report_ids(self) -> set[str]:
        """Return the IDs of the reports in the reserved batch."""
        return {entry.report_id for report_set in report_sets(self.reserved_sets()) for entry in report_set.entries}

    def mark_trained(self, trained_sets: tuple[RecordSet, ...]) -> frozenset[str]:
        consumed_reports: set[str] = set()
        trained_sources: set[str] = set()
        changed_groups: set[Hashable] = set()
        for entry in (entry for report_set in report_sets(trained_sets) for entry in report_set.entries):
            report_id = entry.report_id
            self.consumed.add(report_id)
            consumed_reports.add(report_id)
            self.reports.pop(report_id, None)
            trained_sources.update(entry.context.references)
            self.ended_set_ids.add(report_id)
            if entry.group_key is not None:
                group = self.groups.get(entry.group_key)
                if group is not None:
                    group.pop(entry.slot, None)
                    changed_groups.add(entry.group_key)
        for key in changed_groups:
            if self.groups[key]:
                self.refresh_group(key)
            else:
                self.groups.pop(key)
                self.pool.remove(("group", key))
        self.trained_sources.update(trained_sources)
        return frozenset(consumed_reports | trained_sources)

    # ---------------------------------------------------------------- release

    def live_references(self) -> set[str]:
        return {ref for report in self.reports.values() for ref in report.references}

    def releasable_record_ids(self) -> frozenset[str]:
        """Find completed records with no remaining live dependents.

        The releasable-source set is recomputed here every time: a source is
        releasable while a terminal report owns it (or a batch consumed it)
        and no live report references it. Live claims and terminal ownership
        both move between reads, so the answer is never latched at event time.
        """
        releasable_sources = (self.terminal_owned_sources | self.trained_sources) - self.live_references()
        return frozenset(self.consumed | self.terminal | releasable_sources)

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        self.consumed -= agent_record_ids
        self.terminal -= agent_record_ids
        self.trained_sources -= agent_record_ids
        self.terminal_owned_sources -= agent_record_ids
        self.seen_reports -= agent_record_ids
        # Only completed inferences without live report references are released.
        for agent_record_id in agent_record_ids:
            self.inferences.pop(agent_record_id, None)

    def status(self) -> Mapping[str, Any]:
        return self.grouping.status()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Unconsumed reports, excluding the reserved batch."""
        reserved = self.reserved_report_ids()
        waiting = [report for record_id, report in self.reports.items() if record_id not in reserved]
        return {
            "unreserved_reports": len(waiting),
            "reserved_reports": len(reserved),
            "oldest_report_wait_seconds": (
                max(0.0, time.time() - min(report.created_at for report in waiting)) if waiting else 0.0
            ),
        }


def candidate_context(record_set: RecordSet) -> ReportContext:
    """Return the report of a formed candidate. A report stage forms only report candidates."""
    if not isinstance(record_set, ReportCandidate):
        raise TypeError(f"the report stage forms report candidates, not {type(record_set).__name__}")
    return record_set.context
