"""Storage contract for the training data items that clients upload to a scenario.

A client can upload a ``TrainDataItem``: an ATIF trajectory or a Harbor task.
A data item is not a record. The pipeline gives it to the task-to-batch stage
directly. The data item store keeps the data items of one scenario in upload
order. A trainer reads them after its cursor. A commit lists the data items
that its batch consumed, by :func:`consumed_id`, so a rebuilt trainer skips them.

The files of a task can be text or binary, so they are not in the database.
The scenario's ``TaskFileStore`` keeps them under their digest, and the stored
item holds the digest. Archiving the scenario retires its data items and their
files with its records. Record retention does not delete data items.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.batches import TrainDataItem
from reef.core.errors import ReefError

#: Marks the ID of a consumed data item in a commit, so that it cannot be a record ID.
CONSUMED_ITEM_PREFIX = "item:"


class DataItemConflict(ReefError):
    """A data item ID is already stored with different content."""


def consumed_id(item_id: str) -> str:
    """Return the ID that a commit lists for a consumed data item."""
    return f"{CONSUMED_ITEM_PREFIX}{item_id}"


@dataclass(frozen=True)
class StoredDataItem:
    """One stored data item, with its position in upload order and the ID that the client gave it.

    The ``task_path`` of a stored ``TaskItem`` is its directory in the task file store.
    """

    sequence: int
    item_id: str
    item: TrainDataItem


class DataItemStore(ABC):
    """Keeps the data items of one scenario in upload order, and the files of its tasks."""

    @abstractmethod
    def add(self, items: tuple[tuple[str, TrainDataItem], ...]) -> bool:
        """Save each ``(item_id, item)`` pair in one transaction, in order. Return True if one or more is new.

        The store copies the directory of each ``TaskItem`` to the task file
        store. An item whose ID is stored with the same content changes
        nothing. Raises ``DataItemConflict`` if an ID is stored with different
        content. Then the store saves no item of the call.
        """

    @abstractmethod
    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredDataItem, ...]:
        """Read at most ``limit`` data items after ``after_sequence``, in upload order."""

    @abstractmethod
    def count(self, *, after_sequence: int = 0) -> int:
        """Return the number of data items after ``after_sequence``."""
