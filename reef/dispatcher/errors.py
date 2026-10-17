"""Training failures that the training status and operational metrics report.

Each error belongs to one scenario and to the thread that recorded it: the
training thread, or the local worker of one component. Only that thread
clears its own error, so a local cycle that succeeds after a failed
training turn does not hide that failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

#: The training thread's name in the error record; local workers are named by their component.
TRAINING_THREAD_SOURCE = "training"


def local_error_source(component: str | None) -> str:
    return "local" if component is None else f"local:{component}"


@dataclass
class TrainingErrors:
    """Recorded errors, failure counts and stale-result refusals; every method takes ``lock``.

    No method calls out while it holds ``lock``, so a caller may hold another
    lock, such as the worker coordination lock, when it calls a method.
    """

    lock: Lock = field(default_factory=Lock)
    #: The last error per (scenario, thread that recorded it).
    errors: dict[tuple[str, str], str] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    #: A failure to build the training status; it is not specific to one scenario.
    status_build_error: str | None = None
    #: Results refused because another trainer's commit replaced their base, per (scenario, component).
    stale_refusals_in_a_row: dict[tuple[str, str | None], int] = field(default_factory=dict)
    stale_refusals_total: dict[tuple[str, str | None], int] = field(default_factory=dict)

    def record(self, scenario: str, error: str, source: str = TRAINING_THREAD_SOURCE) -> None:
        """Store ``error`` for ``scenario`` under ``source`` and count the failure."""
        with self.lock:
            self.errors[(scenario, source)] = error
            self.failure_counts[scenario] = self.failure_counts.get(scenario, 0) + 1

    def clear(self, scenario: str, source: str | None = TRAINING_THREAD_SOURCE) -> None:
        """Clear the error that ``source`` recorded for ``scenario``; ``None`` clears every source."""
        with self.lock:
            for key in [key for key in self.errors if key[0] == scenario]:
                if source is None or key[1] == source:
                    self.errors.pop(key)

    def has_error(self, scenario: str) -> bool:
        with self.lock:
            return any(key[0] == scenario for key in self.errors)

    def failure_count(self, scenario: str) -> int:
        with self.lock:
            return self.failure_counts.get(scenario, 0)

    def forget(self, scenario: str) -> None:
        """Drop the errors and the failure count of a deleted scenario."""
        with self.lock:
            for key in [key for key in self.errors if key[0] == scenario]:
                self.errors.pop(key)
            self.failure_counts.pop(scenario, None)

    def set_status_build_error(self, error: str | None) -> bool:
        """Record a failure to build the training status, or clear it; return whether it changed."""
        with self.lock:
            changed = error != self.status_build_error
            self.status_build_error = error
            return changed

    def snapshot(self) -> tuple[dict[tuple[str, str], str], str | None]:
        """A copy of the recorded errors and the status build error."""
        with self.lock:
            return dict(self.errors), self.status_build_error

    def count_stale_refusal(self, scenario: str, component: str | None) -> int:
        """Count one refused result and return how many were refused in a row."""
        key = (scenario, component)
        with self.lock:
            in_a_row = self.stale_refusals_in_a_row.get(key, 0) + 1
            self.stale_refusals_in_a_row[key] = in_a_row
            self.stale_refusals_total[key] = self.stale_refusals_total.get(key, 0) + 1
            return in_a_row

    def end_stale_refusals(self, scenario: str, component: str | None) -> None:
        """Reset the refusals in a row after a commit lands."""
        with self.lock:
            self.stale_refusals_in_a_row.pop((scenario, component), None)

    def stale_refusal_total(self, scenario: str, component: str | None) -> int:
        with self.lock:
            return self.stale_refusals_total.get((scenario, component), 0)

    def forget_stale_refusals(self, scenario: str) -> None:
        """Drop the refusal counts of a scenario whose local workers stop."""
        with self.lock:
            for counts in (self.stale_refusals_in_a_row, self.stale_refusals_total):
                for key in [key for key in counts if key[0] == scenario]:
                    counts.pop(key)
