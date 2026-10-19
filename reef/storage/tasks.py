"""Storage contract for the tasks that a record-to-task stage generates.

A stage saves each generation job in a task store before any rollout. A
generation holds its source record ids and its candidate tasks. Each candidate
then gets one outcome: the stage publishes it to the train or eval split, or
rejects it with a reason. The stage chooses the split. The store keeps it and
refuses writes that conflict with it.

A published task is a Harbor task directory. The database keeps the state of
the store and the digest of each task directory. A ``TaskFileStore``
(``reef.storage.task_files``) keeps the files of the directory.

A task store belongs to a scenario session. Archiving the scenario retires its
task stores and task files with its records and commits. Record retention
does not delete tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reef.core.errors import ReefError
from reef.core.tasks import GeneratedTask

SplitName = Literal["train", "eval"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "eval")


class TaskStoreError(ReefError):
    """A write conflicts with the stored state, or it names state that the store does not hold."""


class TaskConflict(TaskStoreError):
    """A task name is already published with different files or in the other split."""


@dataclass(frozen=True)
class SplitSettings:
    """The split settings of a task store. They do not change for the life of the store."""

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
class TaskCandidate:
    """One candidate of a generation and its outcome.

    A published candidate has a ``split``. A rejected candidate has an
    ``error``. A pending candidate has neither.
    """

    generated: GeneratedTask
    split: SplitName | None = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.split is not None and self.split not in SPLIT_NAMES:
            raise ValueError("split must be 'train', 'eval' or None")
        if self.split is not None and self.error:
            raise ValueError("a published candidate has no error")

    @property
    def is_pending(self) -> bool:
        return self.split is None and not self.error


@dataclass(frozen=True)
class GenerationState:
    """The checkpoint of one generation job.

    ``candidates`` is ``None`` until the stage saves them. An empty tuple
    means that the job produced no candidate.
    """

    generation: int
    source_record_ids: tuple[str, ...]
    candidates: tuple[TaskCandidate, ...] | None = None
    is_complete: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        record_ids = self.source_record_ids
        if (
            not isinstance(record_ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError("source_record_ids must be a tuple of distinct non-empty strings")
        if self.is_complete and (self.candidates is None or self.pending_indices):
            raise ValueError("a complete generation has an outcome for each candidate")

    @property
    def pending_indices(self) -> tuple[int, ...]:
        """The indices of the candidates that have no outcome."""
        return tuple(index for index, candidate in enumerate(self.candidates or ()) if candidate.is_pending)


@dataclass(frozen=True)
class StoredTask:
    """A published task and its split.

    ``generation`` and ``index`` identify the candidate that published the task
    first. ``files_digest`` identifies its directory in the task file store.
    """

    generated: GeneratedTask
    split: SplitName
    generation: int
    index: int
    files_digest: str


class TaskStore(ABC):
    """The generations and published tasks of one record-to-task stage.

    A store is part of a scenario session and closes with it. Each write is
    durable when it returns. An identical retry of a write changes nothing.
    The caller serializes the writers of one store.
    """

    @abstractmethod
    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        """Create the checkpoint of a generation, or return the checkpoint on a retry.

        Raises ``TaskStoreError`` if the generation exists with other source record ids.
        """

    @abstractmethod
    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        """Save the candidates of a generation before validation or play.

        Raises ``TaskStoreError`` if the generation does not exist or has other candidates.
        """

    @abstractmethod
    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        """Save the task directory of one candidate, publish the candidate to ``split``, and return the task.

        The task files are durable before the publication. Each task name is
        published once. A later candidate with the same files must name the
        same split, and it returns the published task.
        Raises ``TaskConflict`` if the name is published with other files or
        in the other split. Raises ``TaskStoreError`` if the candidate does not
        exist or is rejected.
        """

    @abstractmethod
    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        """Save the rejection of one candidate.

        Raises ``TaskStoreError`` if the candidate does not exist or has another outcome.
        """

    @abstractmethod
    def complete_generation(self, *, generation: int) -> GenerationState:
        """Mark a generation complete. Raises ``TaskStoreError`` if a candidate has no outcome."""

    @abstractmethod
    def generation(self, generation: int) -> GenerationState | None:
        """Return the checkpoint of one generation, or ``None`` if the generation does not exist."""

    @abstractmethod
    def generations(self) -> tuple[GenerationState, ...]:
        """Return all generation checkpoints in generation order."""

    @abstractmethod
    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        """Return the published tasks in generation and candidate order, optionally of one split."""

    @abstractmethod
    def path(self, task: StoredTask) -> Path:
        """Return the local Harbor task directory of a published task. The caller must not change it."""


class TaskStorage(ABC):
    """Opens the task stores of one scenario. Each store has a name and its own split settings."""

    @abstractmethod
    def open(self, name: str, *, split: SplitSettings) -> TaskStore:
        """Open the task store ``name``.

        The first open saves ``split``. Raises ``TaskStoreError`` if a later
        open gives different split settings.
        """


__all__ = [
    "GenerationState",
    "SplitName",
    "SplitSettings",
    "StoredTask",
    "TaskCandidate",
    "TaskConflict",
    "TaskStorage",
    "TaskStore",
    "TaskStoreError",
]
