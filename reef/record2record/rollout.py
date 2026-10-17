"""Rollout groups: the record stage's half of rolling out saved tasks and training on their episodes.

A record stage that rolls out tasks holds two pools. Its task pool holds one ``TaskSet`` per saved task
that waits to run. Its ordinary pool holds the groups of rollout episodes: the reports of one task use in
one rollout batch, keyed by ``(batch, task_use)`` from the episode labels, form one group.

The pipeline begins a rollout batch with its task uses. When the plays end, it hands over the reports that
each use's training episodes produced. A group settles when each of those reports joined it or ended, or
when the pipeline closes the batch's groups after its settle deadline. A report whose batch is not active,
or whose arm is not the training arm, never trains.
"""

from __future__ import annotations

from collections.abc import Hashable

from reef.core.batches import TaskItem, TrainDataItem
from reef.core.records_types import AgentRecord
from reef.core.tasks.metadata import (
    BATCH_LABEL,
    TASK_USE_LABEL,
    TRAINING_ARM,
    reported_arm,
    reported_episode_id,
    reported_label,
)
from reef.pipeline_context import PipelineContext
from reef.record2record.pool import Pool
from reef.record2record.reports import GroupDecision, ReportContext, ReportGrouping, ReportRecord2Record
from reef.record2record.sets import PoolItem, TaskSet


def rollout_key(report: AgentRecord) -> tuple[str, str] | None:
    """The ``(batch, task_use)`` labels of a rollout episode's report, or None for another report."""
    batch = reported_label(report, BATCH_LABEL)
    task_use = reported_label(report, TASK_USE_LABEL)
    return (batch, task_use) if batch is not None and task_use is not None else None


def is_batch_group(item: PoolItem, batch: str) -> bool:
    """Whether a pool item is a group of the rollout batch ``batch``."""
    key = item.group_key
    return isinstance(key, tuple) and len(key) == 2 and key[0] == batch


class RolloutGrouping(ReportGrouping):
    """The training episodes of one task use in one rollout batch form one group.

    A group settles when every report that its plays returned joined it or ended without training, or when
    its batch's groups are closed. A group of fewer than two training episodes has no relative reward and is
    discarded; several reports of one episode count as one episode. A report without an episode ID never
    trains, because its episode cannot be counted; it ends, and so still counts toward its group.
    """

    def __init__(self) -> None:
        #: The rollout batches whose episodes may train.
        self.active_batches: set[str] = set()
        #: The reports that each group's plays returned, by group key, once its plays ended.
        self.expected_reports: dict[Hashable, frozenset[str]] = {}
        #: The reports of each group that ended without training, such as a skipped sample.
        self.ended_reports: dict[Hashable, set[str]] = {}
        #: The groups that settle with the reports they hold, because their batch's settle deadline passed.
        self.closed: set[Hashable] = set()

    def is_training_report(self, report: AgentRecord) -> bool:
        key = rollout_key(report)
        return (
            key is not None
            and key[0] in self.active_batches
            and reported_arm(report) == TRAINING_ARM
            and reported_episode_id(report) is not None
        )

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return rollout_key(context.report), None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if not self.is_complete(key, {member.report.agent_record_id for member in members}):
            return GroupDecision.INCOMPLETE
        episodes = {reported_episode_id(member.report) for member in members}
        return GroupDecision.READY if len(episodes) >= 2 else GroupDecision.DISCARD

    def is_complete(self, key: Hashable, member_report_ids: set[str]) -> bool:
        """Whether a group with these member reports can settle."""
        if key in self.closed:
            return True
        expected = self.expected_reports.get(key)
        return expected is not None and expected <= member_report_ids | self.ended_reports.get(key, set())


class RolloutReportRecord2Record(ReportRecord2Record):
    """A report stage that groups rollout episodes by ``(batch, task_use)``, with a task pool."""

    def __init__(
        self, context: PipelineContext, grouping: RolloutGrouping, *, exclusive_sources: bool = False
    ) -> None:
        super().__init__(context, grouping, exclusive_sources=exclusive_sources)
        self.rollout_grouping = grouping
        self.task_pool: Pool[TaskSet] = Pool()
        #: The task uses of each active rollout batch.
        self.rollout_uses: dict[str, tuple[str, ...]] = {}

    def add_tasks(self, tasks: tuple[TrainDataItem, ...], *, group_key: Hashable) -> None:
        """Put saved tasks in the task pool."""
        for task in tasks:
            if not isinstance(task, TaskItem):
                raise TypeError("the task pool holds TaskItem values")
            self.task_pool.put(
                TaskSet(
                    set_id=task.task_path.name,
                    arrival_index=self.task_pool.next_arrival_index(),
                    group_key=group_key,
                    item=task,
                )
            )

    def begin_rollout(self, batch: str, task_uses: tuple[str, ...]) -> None:
        """A rollout batch starts with these task uses; their episodes may train in it."""
        self.rollout_uses[batch] = task_uses
        self.rollout_grouping.active_batches.add(batch)

    def plays_ended(self, key: Hashable, report_ids: tuple[str, ...]) -> None:
        """The plays of the group ``key`` ended; these are the reports of its training episodes."""
        self.rollout_grouping.expected_reports[key] = frozenset(report_ids)
        # The plays can end before the group's first report arrives.
        if key in self.groups:
            self.refresh_group(key)

    def close_groups(self, batch: str) -> None:
        """Settle the batch's groups with the reports they hold: their settle deadline passed."""
        for task_use in self.rollout_uses.get(batch, ()):
            key = (batch, task_use)
            self.rollout_grouping.closed.add(key)
            if key in self.groups:
                self.refresh_group(key)

    def batch_groups_settled(self, batch: str) -> bool:
        """Whether every group of the batch is ready, discarded, or settled without a report."""
        for task_use in self.rollout_uses.get(batch, ()):
            key = (batch, task_use)
            group = self.groups.get(key)
            if group is None:
                if not self.rollout_grouping.is_complete(key, set()):
                    return False
            elif not group.discarded and ("group", key) not in self.pool.sets_by_id:
                return False
        return True

    def end_report(self, report: AgentRecord) -> None:
        """End a report without training; a training report of the active batch still counts toward its group."""
        super().end_report(report)
        key = rollout_key(report)
        if key is None or key[0] not in self.rollout_grouping.active_batches or reported_arm(report) != TRAINING_ARM:
            return
        self.rollout_grouping.ended_reports.setdefault(key, set()).add(report.agent_record_id)
        group = self.groups.get(key)
        if group is not None and not group.discarded:
            self.refresh_group(key)

    def end_rollout(self, batch: str, *, trained: bool) -> None:
        """The batch trained, or was given up when ``trained`` is False; later reports of it never train."""
        self.rollout_grouping.active_batches.discard(batch)
        for task_use in self.rollout_uses.pop(batch, ()):
            key = (batch, task_use)
            self.rollout_grouping.expected_reports.pop(key, None)
            self.rollout_grouping.ended_reports.pop(key, None)
            self.rollout_grouping.closed.discard(key)
            # A report that arrived after the batch was reserved ends with the batch.
            if key in self.groups:
                self.discard_group(key)
                self.groups.pop(key)
