"""Whether a service registers a new scenario, and which loaded scenarios it unloads to load another.

The registry asks a :class:`ScenarioAdmission` before it creates a scenario and
before it loads one. The plugin owns the policy: how many scenarios a service
holds and which loaded ones go first. Reef owns the safety: it unloads only a
scenario that is idle, and it refuses the load when too few are idle.
:class:`ScenarioLimits` is the built-in plugin; it applies
``reef.max_loaded_scenarios`` and ``reef.max_scenarios``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.errors import ScenarioLimitReached
from reef.core.plugins import ConfiguredPlugin


@dataclass(frozen=True)
class LoadedScenario:
    """One scenario that holds a load slot, as admission sees it."""

    name: str
    #: ``time.monotonic()`` of the scenario's last request, record or local cycle.
    last_used_at: float
    #: True while the scenario is still loading; Reef cannot unload it.
    loading: bool = False


@dataclass(frozen=True)
class LoadAdmission:
    """What Reef does before it loads a scenario."""

    #: How many loaded scenarios Reef must unload first.
    unload_count: int = 0
    #: The scenarios Reef tries to unload, in this order; it skips one that is not idle.
    candidates: tuple[str, ...] = ()


class ScenarioAdmission(ConfiguredPlugin, ABC):
    """Decides whether a service creates a scenario and what it unloads to load one.

    Reef calls the methods for one scenario under that scenario's lock, and it
    serializes the calls for different scenarios. A method must return
    quickly and must not call the dispatcher.
    """

    @abstractmethod
    def check_creation(self, scenario: str, *, registered_count: int) -> None:
        """Refuse a new scenario with ``ScenarioLimitReached``; ``registered_count`` excludes it."""

    @abstractmethod
    def admit_load(self, scenario: str, *, loaded: tuple[LoadedScenario, ...]) -> LoadAdmission:
        """What Reef unloads before it loads ``scenario``; ``ScenarioCapacityFull`` refuses the load.

        ``loaded`` holds every scenario with a load slot, including those
        still loading. When fewer of the candidates are idle than
        ``unload_count``, Reef refuses the load with ``ScenarioCapacityFull``.
        """


class ScenarioLimits(ScenarioAdmission):
    """The built-in admission: at most ``max_loaded_scenarios`` loaded and ``max_scenarios`` registered.

    ``None`` means no limit. At the load limit, the least recently used idle
    scenario is unloaded first.
    """

    def __init__(self, *, max_loaded_scenarios: int | None = None, max_scenarios: int | None = None) -> None:
        for setting, value in (("max_loaded_scenarios", max_loaded_scenarios), ("max_scenarios", max_scenarios)):
            if value is not None and value < 1:
                raise ValueError(f"reef.{setting} must be positive, got {value}")
        self.max_loaded_scenarios = max_loaded_scenarios
        self.max_scenarios = max_scenarios

    def check_creation(self, scenario: str, *, registered_count: int) -> None:
        if self.max_scenarios is not None and registered_count >= self.max_scenarios:
            raise ScenarioLimitReached(
                f"cannot create scenario {scenario!r}: the service holds its limit of {self.max_scenarios} "
                "scenarios (reef.max_scenarios); delete a scenario first"
            )

    def admit_load(self, scenario: str, *, loaded: tuple[LoadedScenario, ...]) -> LoadAdmission:
        if self.max_loaded_scenarios is None or len(loaded) < self.max_loaded_scenarios:
            return LoadAdmission()
        least_recent_first = sorted(
            (entry for entry in loaded if not entry.loading), key=lambda entry: entry.last_used_at
        )
        return LoadAdmission(
            unload_count=len(loaded) - self.max_loaded_scenarios + 1,
            candidates=tuple(entry.name for entry in least_recent_first),
        )


__all__ = ["LoadAdmission", "LoadedScenario", "ScenarioAdmission", "ScenarioLimits"]
