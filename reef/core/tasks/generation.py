"""The task generation contract: the request, the generated tasks, and their play results.

A deployment selects one :class:`TaskGenerator` in its
``pipeline.record2task.config.generator`` setting. A recipe that trains on
generated tasks asks the generator for tasks, validates each one, plays the
valid ones, and sends the results back to the generator. This module holds the values, their field validation, and the
generator interface. Consumers own file reads, transport limits, and execution.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from reef.core.plugins import Plugin
from reef.core.records_types import AgentRecord
from reef.core.tasks.harbor import HarborTask


@dataclass(frozen=True)
class TaskGenerationRequest:
    """Source records, requirements, and materials for one call to a task generator.

    Assets name files or directories on the caller's machine. Alternatively,
    asset_files maps relative file names to UTF-8 contents. Supply paths or
    contents, not both. Construction validates fields but performs no I/O.
    Consumers own file reads, transport limits, and generation settings.
    Description-only generation uses an empty source_records tuple.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()
    asset_files: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_records, tuple):
            raise ValueError("source_records must be a tuple of AgentRecord values")
        if any(not isinstance(record, AgentRecord) for record in self.source_records):
            raise TypeError("source_records must contain AgentRecord values")
        if len({record.scenario for record in self.source_records}) > 1:
            raise ValueError("source_records must belong to one scenario")
        record_ids = [record.agent_record_id for record in self.source_records]
        if any(not isinstance(record_id, str) or not record_id for record_id in record_ids) or len(
            set(record_ids)
        ) != len(record_ids):
            raise ValueError("source_records must have distinct non-empty record ids")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(self.assets, tuple) or any(not isinstance(path, Path) for path in self.assets):
            raise TypeError("assets must be a tuple of pathlib.Path values")
        if not isinstance(self.asset_files, Mapping) or any(
            not isinstance(name, str) or not isinstance(text, str) for name, text in self.asset_files.items()
        ):
            raise ValueError("asset_files must map relative file names to UTF-8 text")
        if self.assets and self.asset_files:
            raise ValueError("supply assets or asset_files, not both")
        for name, text in self.asset_files.items():
            path = PurePosixPath(name)
            if not name or name == "." or path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
                raise ValueError("asset_files must use relative file names without parent traversal")
            if "\x00" in text:
                raise ValueError("asset_files must contain UTF-8 text without NUL bytes")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("asset_files must contain UTF-8 text") from exc


@dataclass(frozen=True)
class TaskValidationResult:
    """A completed validation: no errors means the task passed all required checks.

    Errors describe task defects. Failures to execute the checks (for example,
    an unavailable container runtime) raise exceptions instead of rejecting the
    task as invalid.
    """

    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple of non-empty strings")
        if any(not isinstance(error, str) or not error.strip() for error in self.errors):
            raise ValueError("errors must contain non-empty strings")

    @property
    def is_valid(self) -> bool:
        """Whether every required check passed."""
        return not self.errors


@dataclass(frozen=True)
class TaskPlayResult:
    """What became of one generated task in a round.

    ``training_rewards`` holds the rewards of the plays the recipe trains on,
    only of the episodes that ran. An episode that ran and returned no reward
    counts as 0. ``error`` is not
    empty when the task was not played: it failed validation, the task player
    refused it, or the agent could not start on it.
    An eval task has no rewards and is not played, even with an empty error.
    """

    task: HarborTask
    training_rewards: tuple[float, ...] = ()
    error: str = ""
    split: Literal["train", "eval"] = "train"

    def __post_init__(self) -> None:
        if not isinstance(self.task, HarborTask):
            raise TypeError("task must be a HarborTask")
        if not isinstance(self.training_rewards, tuple):
            raise TypeError("training_rewards must be a tuple of numbers")
        if any(
            isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward)
            for reward in self.training_rewards
        ):
            raise ValueError("rewards must be finite numbers")
        if not isinstance(self.error, str):
            raise TypeError("error must be text")
        if self.error and self.training_rewards:
            raise ValueError("a task that was not played has no rewards")
        if self.split not in ("train", "eval"):
            raise ValueError("task result split must be train or eval")
        if self.split == "eval" and self.training_rewards:
            raise ValueError("an eval task result cannot contain rollout rewards")

    @property
    def is_played(self) -> bool:
        return self.split == "train" and not self.error


class TaskGenerator(Plugin, ABC):
    """Makes tasks for one scenario, and adapts to how the agent played them.

    A deployment names the class in ``pipeline.record2task.config.generator``. Reef calls
    :meth:`from_config` once for each scenario. A generator gets what it needs
    from its config and from these calls only; it does not read the recipe's
    settings.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        """Build the generator of ``scenario`` from the section's ``config``.

        ``environ`` holds the environment the deployment was read with, for
        credentials that the config names by environment variable.
        """

    @abstractmethod
    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        """Return zero or more new tasks.

        Preserve the request's source record ids in each task's
        ``source_agent_record_ids``. An exception stops the round.
        """

    @abstractmethod
    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """Check one generated task before the recipe plays it.

        Return task defects as validation errors. Raise when the checks cannot
        run; the exception stops the round. The task is a value: a validation
        that needs a directory writes its own copy.
        """

    async def record_results(self, results: tuple[TaskPlayResult, ...]) -> None:
        """Receive the result of each task that the last :meth:`generate` call returned.

        The recipe calls this once for each round, after the round's plays,
        also when the round stopped early. A task that the round did not reach
        has no result. The default keeps nothing.
        """
