"""CORAL attempts -> grouped policy batches (the training-semantics piece).

CORAL's attempt tree already has the shape reef's grouped relative-reward
training wants: every attempt names its ``parent_hash``, so the siblings of
one parent form one comparison group — attempts that started from the same
code and diverged. This processor groups reports by parent commit and
releases a group as one training unit once ``group_size`` scored siblings
accrued.

The reports it consumes are exactly what :mod:`recipes.beta.coral.reporter` emits:
``score``, ordered inference ``references``, and ``metadata.coral`` with
``agent_id``/``commit_hash``/``parent_hash``. Root attempts (no parent)
compare against each other under a sentinel group — they all diverged from
the task seed.

Token/loss-mask/logprob materialization is reef's ``SampleAssembly`` over the
referenced INFERENCE records; a CORAL attempt is a multi-call trajectory, so
multi-turn assembly is on by default.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.record2record import GroupDecision, ReportContext, ReportGrouping
from reef.train.processors.reported import ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

logger = logging.getLogger(__name__)

#: Group key for attempts without a parent commit: first-generation attempts
#: all diverged from the task seed, so they are each other's siblings.
ROOT_GROUP = "__coral_root__"


def coral_metadata(context: ReportContext) -> Mapping[str, Any] | None:
    """The report's ``metadata.coral``, or None when it has no commit hash."""
    metadata = context.report.payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    coral = metadata.get("coral")
    if not isinstance(coral, Mapping):
        return None
    if not isinstance(coral.get("commit_hash"), str) or not coral["commit_hash"]:
        return None
    return coral


def sibling_group(context: ReportContext) -> tuple[str, str]:
    """The attempt's sibling group (its parent commit, or ``ROOT_GROUP``) and its own commit hash."""
    coral = coral_metadata(context)
    if coral is None:
        raise ValueError("CoralProcessor requires metadata.coral with a commit_hash")
    parent = coral.get("parent_hash")
    return (parent if isinstance(parent, str) and parent else ROOT_GROUP), coral["commit_hash"]


def attempt_release_ids(context: ReportContext) -> set[str]:
    """The releases that served the attempt's inference records."""
    return {
        inference.artifact_ref.release_id for inference in context.inferences if inference.artifact_ref is not None
    }


class CoralGrouping(ReportGrouping):
    """The siblings of one parent commit form one group, and each attempt's commit hash is its slot.

    A group is ready once ``group_size`` siblings are present. A group whose
    attempts span releases is discarded and listed in ``discarded_groups``.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size
        self.mixed_release_groups: dict[str, tuple[str, ...]] = {}

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return sibling_group(context)

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < self.group_size:
            return GroupDecision.INCOMPLETE
        versions = {release_id for member in members for release_id in attempt_release_ids(member)}
        if len(versions) <= 1:
            return GroupDecision.READY
        ordered = tuple(sorted(versions))
        self.mixed_release_groups[str(key)] = ordered
        logger.error(
            "CORAL sibling group %s discarded: %d attempts span releases %s",
            key,
            len(members),
            list(ordered),
        )
        return GroupDecision.DISCARD

    def status(self) -> Mapping[str, Any]:
        return {
            "discarded_groups": [
                {"parent": parent, "reason": "mixed_release_ids", "release_ids": list(versions)}
                for parent, versions in sorted(self.mixed_release_groups.items())
            ],
        }


class CoralProcessor(ReportedFeedbackProcessor):
    """One CORAL sibling group = one grouped relative-reward training unit.

    Config:

    - ``group_size`` (default 4, min 2): scored siblings required before a
      parent's group trains. CORAL sibling counts are dynamic, so this is a
      recipe-level barrier, not a CORAL invariant; parents that never accrue
      enough scored children simply never train.
    """

    output_schema = TrainingBatch
    exclusive_sources = True
    ordered_groups = False  # sibling groups close in whatever order grading lands

    def __init__(self, context: ProcessorContext) -> None:
        config = dict(context.config)
        self.group_size = int(config.get("group_size", 4))
        if self.group_size < 2:
            raise ValueError("group_size must be at least two (relative rewards need contrast)")
        config.setdefault("accept_multi_turn_policy_samples", True)
        assembly_config = context.with_config(config)
        self._assembly = SampleAssembly.from_config(assembly_config)
        super().__init__(context.with_config({**config, "batch_size": 1}))

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        coral = coral_metadata(context)
        if coral is None:
            raise ValueError("CoralProcessor requires metadata.coral with a commit_hash")
        sample = self._assembly.build(context, context.require_score())
        release_ids = attempt_release_ids(context)
        if len(release_ids) > 1:
            raise ValueError(f"attempt {coral['commit_hash'][:12]} spans releases {sorted(release_ids)}")
        return sample.with_metadata(
            coral={**coral, "group": sibling_group(context)[0], "release_id": next(iter(release_ids), None)}
        )

    def report_grouping(self) -> CoralGrouping:
        return CoralGrouping(self.group_size)

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        parents = {item.metadata["coral"]["group"] for item in items}
        if len(parents) != 1:
            raise RuntimeError("CoralProcessor batches exactly one sibling group per step")
        parent = next(iter(parents))
        batch = TrainingBatch(f"{self.scenario}:coral:{parent}:{batch_number}", items)
        group = trajectories(batch)
        rewards = tuple(trajectory_reward(sample) for sample in group)
        if all(reward == rewards[0] for reward in rewards[1:]):
            # Constant-reward group: keep it for a well-defined zero-gradient
            # step rather than starving the barrier (mirrors TTTD's fallback).
            logger.info("CORAL group %s has constant reward %s", parent, rewards[0])
        self.experiment_logger.log(
            {
                "parent": parent,
                "siblings": len(group),
                "reward_min": min(rewards),
                "reward_max": max(rewards),
            },
            namespace="coral",
        )
        return replace(batch, items=tuple(replace(item, group_id=str(parent)) for item in group))
