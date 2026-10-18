"""Storage contract for the tasks that a record-to-task stage generates for one scenario.

A stage saves each generation job here before any rollout. A generation holds
its source record ids, then its candidate tasks, then one outcome for each
candidate: published to the train or eval split, or rejected with a reason.
A published task keeps its split for the life of the store. The stage chooses
the split; the store keeps it and refuses writes that conflict with it.

Task stores belong to the scenario's storage session. Archiving the scenario
retires them with its records and commits. Record retention does not delete
them, because a task holds its own content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from reef.core.errors import ReefError
from reef.core.tasks import GeneratedTask

SplitName = Literal["train", "eval"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "eval")


class TaskStoreError(ReefError):
    """A task store write conflicts with the stored state, or names state that the store does not hold."""


class TaskConflict(TaskStoreError):
    """A task name already identifies different task content, or the task is already in the other split."""


@dataclass(frozen=True)
class SplitSettings:
    """The split configuration of a task store. The store keeps it unchanged for its whole life."""

    eval_fraction: float
    seed: int

    def __post_init__(self) -> None:
        fraction = self.eval_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
            raise ValueError("eval_fraction must be a number between 0 and 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "eval_fraction", float(fraction))


@dataclass(frozen=True)
class CandidateOutcome:
    """What became of one candidate: published to ``split``, or rejected with ``error``."""

    index: int
    split: SplitName | None = None
    error: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("a candidate index must be a non-negative integer")
        if self.split is None:
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("a rejected candidate needs a reason")
        elif self.split not in SPLIT_NAMES or self.error:
            raise ValueError("a published candidate has a split, 'train' or 'eval', and no error")


@dataclass(frozen=True)
class GenerationState:
    """The checkpoint of one generation job.

    ``candidates`` is ``None`` until the stage saves them; an empty tuple means
    the job produced no candidate. ``outcomes`` holds at most one outcome for
    each candidate index.
    """

    generation: int
    source_record_ids: tuple[str, ...]
    candidates: tuple[GeneratedTask, ...] | None = None
    outcomes: tuple[CandidateOutcome, ...] = ()
    is_complete: bool = False

    def __post_init__(self) -> None:
        indices = [outcome.index for outcome in self.outcomes]
        if len(indices) != len(set(indices)) or any(index >= len(self.candidates or ()) for index in indices):
            raise ValueError("generation outcomes must name distinct saved candidates")
        if self.is_complete and (self.candidates is None or self.pending_indices):
            raise ValueError("a complete generation has an outcome for every candidate")

    @property
    def pending_indices(self) -> tuple[int, ...]:
        """Candidate indices that have no outcome yet."""
        completed = {outcome.index for outcome in self.outcomes}
        return tuple(index for index in range(len(self.candidates or ())) if index not in completed)


@dataclass(frozen=True)
class StoredTask:
    """A published task with its split, and the generation candidate that first published it."""

    generated: GeneratedTask
    split: SplitName
    generation: int
    index: int


class TaskStore(ABC):
    """The generation checkpoints and published tasks of one record-to-task stage.

    A store is part of a scenario session and closes with it. Every write is
    durable when it returns, and an identical retry of a write is a no-op.
    The caller serializes the writers of one store.
    """

    @abstractmethod
    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create the checkpoint of a generation, or return it when the job is retried.

        Raises ``TaskStoreError`` when the generation exists with other source record ids.
        """

    @abstractmethod
    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        """Save the candidates of a begun generation before any of them is validated or played.

        Raises ``TaskStoreError`` when the generation has not begun or already holds other candidates.
        """

    @abstractmethod
    def generation(self, generation: int) -> GenerationState | None:
        """The checkpoint of one generation, or ``None`` when it has not begun."""

    @abstractmethod
    def generations(self) -> tuple[GenerationState, ...]:
        """Every generation checkpoint, in generation order."""

    @abstractmethod
    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        """Publish one saved candidate to ``split`` and return the published task.

        A task name is published once. Another candidate with the same task
        content must name the same split, and gets the task that is already
        published. Raises ``TaskConflict`` when the name is published with
        other content or in the other split. Raises ``TaskStoreError`` when the
        candidate does not exist or already has another outcome.
        """

    @abstractmethod
    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        """Save the rejection of one saved candidate.

        Raises ``TaskStoreError`` when the candidate does not exist or already has another outcome.
        """

    @abstractmethod
    def complete_generation(self, *, generation: int) -> GenerationState:
        """Mark a generation complete. Raises ``TaskStoreError`` while a candidate has no outcome."""

    @abstractmethod
    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        """Published tasks in publication order, optionally of one split."""


class TaskStorage(ABC):
    """Opens the task stores of one scenario. Each store has its own name and split settings."""

    @abstractmethod
    def task_store(self, name: str, *, split: SplitSettings) -> TaskStore:
        """Open the task store ``name``.

        The first open saves ``split``. Raises ``TaskStoreError`` when a later
        open names different split settings.
        """


__all__ = [
    "CandidateOutcome",
    "GenerationState",
    "SplitName",
    "SplitSettings",
    "StoredTask",
    "TaskConflict",
    "TaskStorage",
    "TaskStore",
    "TaskStoreError",
]
