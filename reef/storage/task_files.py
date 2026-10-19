"""Storage of the task directories that the task stores publish.

A published task is a Harbor task directory. Its files can be text or binary,
so they are not in the database. The database keeps the digest of the files;
a ``TaskFileStore`` keeps the files under that digest. The same files always
give the same digest, so a repeated save changes nothing.

A deployment selects the store in ``storage.task_file_backend``: ``local``
selects :class:`LocalTaskFileStore`, and a plugin reference selects another
subclass of :class:`TaskFileStore`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from reef.core.plugins import NoSettings, Plugin
from reef.storage.tasks import TaskStoreError


class TaskFileStore(Plugin, ABC):
    """Keeps the directory of each published task, addressed by the digest of its files.

    The store keeps the files of each scenario apart, so that ``archive``
    retires them with the scenario. The scenario storage owns the store and
    closes it.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        """Build the store that a deployment names.

        ``config`` is the configuration of the plugin. ``directory`` is
        ``storage.task_file_dir``. A local store keeps the files there. A
        remote store can use it as a local cache.
        """

    @abstractmethod
    def put(self, scenario: str, source: Path) -> str:
        """Save a copy of the task directory ``source`` and return the digest of its files.

        The copy is durable when the method returns. Raises ``TaskStoreError``
        if ``source`` holds an entry that is not a file or a directory.
        """

    @abstractmethod
    def path(self, scenario: str, digest: str) -> Path:
        """Return a local task directory with the files of ``digest``.

        The directory has the name of the saved source directory. The caller
        must not change it. Raises ``TaskStoreError`` if the store does not
        hold the digest.
        """

    @abstractmethod
    def archive(self, scenario: str) -> tuple[str, ...]:
        """Retire the files of a scenario and return the archive locations."""

    @abstractmethod
    def close(self) -> None:
        """Release the resources of the store. A second call does nothing."""


def files_digest(source: Path) -> str:
    """Return the sha256 digest of a task directory: its name, and the path and bytes of each file.

    Raises ``TaskStoreError`` if the directory holds a symbolic link or a special file.
    """
    digest = hashlib.sha256(source.name.encode("utf-8"))
    for path in sorted(source.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise TaskStoreError(f"task directory {source} holds {path.relative_to(source)}, which is not a file")
        name = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name + len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


class LocalTaskFileStore(TaskFileStore):
    """Keeps the task directories in a local directory.

    The layout is ``<directory>/<scenario key>/<digest>/<task name>/``. A copy
    is first written under ``<directory>/.staging`` and then renamed into
    place. The store never reads ``.staging``; an interrupted copy can stay
    there, and an operator can delete it while no Reef service runs. Without
    a directory, the store uses a temporary directory and deletes it on
    ``close``. The store creates its directories on the first ``put``.
    """

    def __init__(self, directory: Path | None = None) -> None:
        self.temporary: tempfile.TemporaryDirectory[str] | None
        if directory is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="reef-task-files-")
            self.directory = Path(self.temporary.name)
        else:
            self.temporary = None
            self.directory = Path(directory)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, directory: Path) -> Self:
        NoSettings.parse(config, plugin_class=cls)
        return cls(directory)

    def put(self, scenario: str, source: Path) -> str:
        digest = files_digest(source)
        target = self.scenario_directory(scenario) / digest
        if target.exists():
            return digest
        staging = self.directory / ".staging" / uuid.uuid4().hex
        try:
            shutil.copytree(source, staging / source.name)
            sync_tree(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
            sync_directory(target.parent)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return digest

    def path(self, scenario: str, digest: str) -> Path:
        target = self.scenario_directory(scenario) / digest
        entries = list(target.iterdir()) if target.is_dir() else []
        if len(entries) != 1:
            raise TaskStoreError(f"the task file store holds no files with digest {digest}")
        return entries[0]

    def archive(self, scenario: str) -> tuple[str, ...]:
        source = self.scenario_directory(scenario)
        if not source.exists():
            return ()
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        destination = self.directory / "archived" / f"{source.name}-{stamp}-{uuid.uuid4().hex}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return (str(destination),)

    def close(self) -> None:
        if self.temporary is not None:
            self.temporary.cleanup()

    def scenario_directory(self, scenario: str) -> Path:
        """Return the directory that holds the task files of one scenario."""
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return self.directory / hashlib.sha256(scenario.encode("utf-8")).hexdigest()


def sync_tree(root: Path) -> None:
    """Flush each file and directory under ``root``, the deepest entries first."""
    for path in sorted(root.rglob("*"), key=lambda entry: len(entry.parts), reverse=True):
        if path.is_dir():
            sync_directory(path)
        else:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    sync_directory(root)


def sync_directory(path: Path) -> None:
    """Flush the entries of one directory."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["LocalTaskFileStore", "TaskFileStore", "files_digest"]
