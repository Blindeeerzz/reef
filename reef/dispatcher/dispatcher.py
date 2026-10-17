"""The dispatcher: the process-wide owner of every scenario and its training threads."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackendFactory
from reef.core.errors import ScenarioBusy, ScenarioCapacityFull, UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import ingest
from reef.dispatcher.errors import TrainingErrors, error_text
from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.dispatcher.status import TrainingStatus
from reef.dispatcher.training_loop import DispatchedTrainingLoop, in_flight_job_marker
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.recipe.base import Recipe
from reef.recipe.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.admission import ScenarioAdmission
from reef.scenario.registry import InstanceState, ScenarioLifecycle, ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.storage.records import RecordRetention
from reef.storage.scenario import ScenarioStorage

logger = logging.getLogger(__name__)


@dataclass
class _LifecycleState:
    preload_thread: Thread | None = None
    metrics_thread: Thread | None = None
    idle_unload_thread: Thread | None = None


class Dispatcher(ScenarioLifecycle):
    """Coordinate scenario creation, inference state, training, and model commits.

    Invariant: at most one weight-training scenario per process. Its serial
    thread is bound to the first scenario using the deployment's
    ``TrainingRuntime``; resolving a second raises (enforced in
    :class:`ScenarioRegistry`).
    Local training backends are unlimited and drain on per-component threads
    (:class:`LocalTrainingWorkers`); weight training runs on the training
    thread (:class:`DispatchedTrainingLoop`).
    The dispatcher owns its recipe's runtime and closes it after all scenarios.
    It unloads an idle scenario (see :meth:`scenario_is_idle`) after
    ``scenario_idle_unload_seconds``, and when scenario admission needs a
    load slot.
    """

    operational_metrics_interval_seconds: float = 10.0
    # The longest time between two checks for scenarios that are idle for long enough to unload.
    idle_unload_check_seconds: float = 30.0

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        allow_implicit_creation: bool = True,
        experiment_tracker: ExperimentTracker | None = None,
        scenario_storage: ScenarioStorage,
        hold_local_cycles: bool = False,
        scenario_idle_unload_seconds: float | None = None,
        scenario_admission: ScenarioAdmission | None = None,
    ) -> None:
        """``hold_local_cycles``: run no local cycle until :meth:`open_local_cycles`, for a service whose recipe
        calls this same service, which does not answer yet while the dispatcher starts.

        ``scenario_idle_unload_seconds``: unload a scenario that is idle for
        this time. With ``None``, every scenario stays loaded.

        ``scenario_admission``: decides whether a scenario is created or
        loaded. The default admits all scenarios.
        """
        if scenario_idle_unload_seconds is not None and scenario_idle_unload_seconds <= 0:
            raise ValueError("scenario_idle_unload_seconds must be positive")
        self._recipe = recipe
        self._storage = scenario_storage
        self._record_retention_lock = Lock()
        self._experiment_tracker = experiment_tracker if experiment_tracker is not None else NullExperimentTracker()
        self._registry = ScenarioRegistry(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            scenario_storage=scenario_storage,
            allow_implicit_creation=allow_implicit_creation,
            experiment_tracker=self._experiment_tracker,
            admission=scenario_admission,
            lifecycle=self,
        )
        self.errors = TrainingErrors()
        self._lifecycle = _LifecycleState()
        self.local_workers = LocalTrainingWorkers(self._registry, self.errors, Event(), held=hold_local_cycles)
        self.training_loop = DispatchedTrainingLoop(self._registry, self.errors, self.local_workers)
        self.training_status = TrainingStatus(recipe, self._registry, self.errors, self.training_loop)
        if isinstance(backend_factory, EnumerableRepositoryBackendFactory):
            self._lifecycle.preload_thread = Thread(
                target=self._preload_scenarios,
                args=(backend_factory.list_registrations(),),
                name="reef-preload",
                daemon=True,
            )
            self._lifecycle.preload_thread.start()
        if self._experiment_tracker.operational_metrics_enabled:
            self._lifecycle.metrics_thread = Thread(
                target=self.run_operational_metrics,
                name="reef-operational-metrics",
                daemon=True,
            )
            self._lifecycle.metrics_thread.start()
        if scenario_idle_unload_seconds is not None:
            self._lifecycle.idle_unload_thread = Thread(
                target=self.run_idle_unload,
                args=(scenario_idle_unload_seconds,),
                name="reef-idle-unload",
                daemon=True,
            )
            self._lifecycle.idle_unload_thread.start()

    @property
    def recipe(self) -> Recipe:
        """The deployment's recipe, before any scenario's own model settings."""
        return self._recipe

    # -- Scenario API (delegates to registry) ----------------------------

    def has_scenario(self, scenario: str) -> bool:
        return self._registry.has(scenario)

    def get_or_create_scenario(
        self,
        scenario: str,
        *,
        release_id: str | None = None,
        allow_implicit_creation: bool | None = None,
    ) -> Scenario | None:
        return self._registry.get_or_create(
            scenario,
            release_id,
            allow_implicit_creation=allow_implicit_creation,
        )

    def loaded_scenario(self, scenario: str, *, release_id: str | None = None) -> Scenario | None:
        """The loaded instance of ``scenario``, or None; see :meth:`ScenarioRegistry.get_loaded`."""
        return self._registry.get_loaded(scenario, release_id)

    def configure_scenario_model(
        self, scenario: str, model: object, *, create: bool = False, release_id: str | None = None
    ) -> Scenario:
        return self._registry.configure_model(scenario, model, create=create, release_id=release_id)

    def set_training_mode(self, scenario: str, training_mode: str) -> dict[str, Any]:
        with self._registry.lock_for(scenario):
            current = self._registry.set_training_mode(scenario, training_mode)
            self.local_workers.wake(current)
            return {"scenario": scenario, "training_mode": current.training_mode}

    def list_scenarios(self) -> tuple[dict[str, Any], ...]:
        return self._registry.list()

    def prune_record_archives(self, retention: RecordRetention) -> int:
        """Apply deployment-wide retention without racing scenario file moves."""
        with self._record_retention_lock:
            return self._storage.prune(max_bytes=retention.max_bytes)

    def delete_scenario(self, scenario: str) -> dict[str, Any]:
        """Remove a scenario from this deployment and move its own state aside.

        Under the scenario's lock, so no accept or commit interleaves: the
        training thread and the local worker lose the name, the loaded
        instance closes, the storage archives the records, commits and
        settings, the recipe's own directories move
        beside themselves, and the repository registration is archived so
        the name is free. Artifacts other scenarios share stay.
        """
        if "/" in scenario or scenario in ("", ".", ".."):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        with self._record_retention_lock, self._registry.lock_for(scenario):
            if not self._registry.has(scenario):
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            loaded = self._registry.get_optional(scenario)
            if (loaded is not None and loaded.is_job_reserved) or self.training_job_in_flight(scenario):
                # The backend's job would outlive its scenario: nothing could commit or acknowledge it, and
                # its marker keeps inference admission closed for every scenario on the runtime.
                raise ScenarioBusy(
                    f"cannot delete scenario {scenario!r} while its training job is out; "
                    "retry once the job has committed or been rejected"
                )
            dropped = self._registry.remove(scenario)
            self.local_workers.remove(scenario)
            self.errors.forget(scenario)
            if dropped is not None:
                for bound in dropped.trainers:
                    backend = bound.trainer.candidate_backend
                    if backend is not None:
                        backend.retire_scenario(scenario)
                dropped.close()
            self.local_workers.close_replaced_instances(scenario)
            archived = [*self._archive_scenario_state(scenario), *self._registry.archive_registration(scenario)]
        self._registry.forget_lock(scenario)
        return {"scenario": scenario, "archived": archived}

    def scenario_loaded(self, instance: Scenario) -> None:
        """Start the training thread when the registry loads a scenario that trains weights.

        The registry calls this method on every resolution path (get_or_create,
        require, preload), so no entry point has to start the thread itself.
        """
        if instance.training_runtime is not None:
            self.training_loop.start()

    def close_replaced(self, instance: Scenario) -> None:
        """Close a reloaded scenario's old instance once no local cycle or dispatched turn runs on it."""
        self.local_workers.close_replaced(instance)

    def scenario_is_idle(self, current: Scenario) -> bool:
        """Whether Reef can unload ``current`` with no loss of data and no stop of work.

        A scenario is idle when all of these are true:

        - Its storage is durable.
        - No request runs on it.
        - No trainer has records that it would still train on.
        - No release exists only in memory.
        - No reload or close of an instance waits.

        A weight-training scenario is never idle, because only a delete
        releases its runtime binding. The caller holds the lock and the
        cycle lock of the scenario, so no record or cycle starts meanwhile.
        """
        if current.training_runtime is not None or not current.store.durable:
            return False
        if current.operations.active_count() > 0 or current.repository.holds_local_releases:
            return False
        if self.local_workers.instance_change_pending(current.name):
            return False
        return not any(bound.trainer.holds_untrained_records() for bound in current.trainers)

    def unload_if_idle(self, scenario: str) -> bool:
        """Unload ``scenario`` if it is idle, and return whether it was unloaded.

        Unload closes the loaded instance and stops its local workers. It is
        a delete without the archive: the storage, the repository
        registration and the settings stay, and the next request loads the
        scenario again. Counters in memory start again from zero. If another
        thread holds the lock or the cycle lock of the scenario, the scenario
        is busy, so this method never waits for them.
        """
        with self._registry.lock_for(scenario, blocking=False) as acquired:
            if not acquired:
                return False
            cycle_lock = self.local_workers.cycle_lock(scenario)
            if not cycle_lock.acquire(blocking=False):
                return False
            try:
                current = self._registry.get_optional(scenario)
                if current is None or not self.scenario_is_idle(current):
                    return False
                self._registry.unload(scenario)
                self.local_workers.remove(scenario)
                current.close()
            finally:
                cycle_lock.release()
            self.local_workers.close_replaced_instances(scenario)
        logger.info("unloaded idle scenario %r; its next request loads it again", scenario)
        return True

    def run_idle_unload(self, idle_seconds: float) -> None:
        """Until the dispatcher closes, unload each scenario that has no use for ``idle_seconds`` and is idle."""
        while not self.local_workers.closed.wait(min(idle_seconds, self.idle_unload_check_seconds)):
            for name in self._registry.unused_since(time.monotonic() - idle_seconds):
                self.unload_if_idle(name)

    def training_job_in_flight(self, scenario: str) -> bool:
        """Whether the training runtime's durable job marker names a job of ``scenario`` still out.

        A loaded instance knows its reserved batch; the marker also covers a
        scenario whose turn failed and was rebuilt, or one not loaded yet,
        whose job the backend still holds. Every job's marker names the
        scenario that owns it.
        """
        try:
            marker = in_flight_job_marker(self._recipe.training_runtime)
        except Exception as exc:
            # A runtime that does not answer cannot say whether a job is out; a blind delete could orphan one.
            raise ScenarioBusy(
                f"cannot delete scenario {scenario!r}: the training runtime does not say whether its job is out "
                f"({error_text(exc)}); retry once it answers"
            ) from exc
        return marker is not None and marker.get("scenario") == scenario

    def _archive_scenario_state(self, scenario: str) -> list[str]:
        """Move the scenario's own files and directories under an ``archived`` sibling, stamped so a name can be deleted twice."""
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        moved = list(self._registry.archive_store(scenario))
        for directory in self._recipe.scenario_state_dirs(scenario):
            if directory.exists():
                destination = directory.parent / "archived" / f"{directory.name}-{stamp}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(directory), str(destination))
                moved.append(str(destination))
        return moved

    def recipe_has_files(self) -> bool:
        """Whether the served recipe creates a file-serving surface."""
        # Capability probe only: file serving does not depend on the scenario.
        return self._recipe.build_surface("").files is not None

    def list_releases(self, scenario: str) -> tuple[dict[str, Any], ...]:
        with self._registry.lock_for(scenario):
            return self._registry.require(scenario).releases()

    def read_records(
        self, scenario: str, *, after_sequence: int = 0, limit: int = 50, request_type: RequestType | None = None
    ) -> dict[str, Any]:
        """Read retained summaries, optionally of one request type, without changing the training queue."""
        from reef.scenario.history import read_records

        with self._registry.lock_for(scenario):
            return read_records(
                self._registry.require(scenario),
                after_sequence=after_sequence,
                limit=limit,
                request_type=request_type,
            )

    def read_record(self, scenario: str, record_id: str) -> dict[str, Any] | None:
        """Read a retained trace within its scenario, including consumed records."""
        from reef.scenario.history import read_record

        with self._registry.lock_for(scenario):
            return read_record(self._registry.require(scenario), record_id)

    def read_commits(
        self, scenario: str, *, after_step: int = 0, limit: int = 50, record_ids: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        from reef.scenario.history import read_commits

        with self._registry.lock_for(scenario):
            return read_commits(
                self._registry.require(scenario), after_step=after_step, limit=limit, record_ids=record_ids
            )

    def scenario_contract(self, scenario: str) -> dict[str, Any]:
        with self._registry.lock_for(scenario):
            current = self._registry.require(scenario)
            pipeline = current.trainer.pipeline
            return {
                "scenario": scenario,
                # The key keeps its name for callers; the value names the recipe's task-to-batch stage.
                "processor": type(pipeline.task2batch).__name__,
                "training_mode": current.trainer.pipeline.training_mode,
                "status": dict(current.trainer.pipeline_status()),
                "required_request_types": sorted(rt.value for rt in pipeline.required_request_types),
            }

    def promote(self, scenario: str, release_id: str) -> ArtifactRef:
        """Serve a pending release: the rollback path under its own record operation."""
        return self.rollback(scenario, release_id, operation="promote")

    def rollback(self, scenario: str, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        while True:
            with self._registry.lock_for(scenario):
                current = self._registry.require(scenario)
            # Wait for earlier reports without blocking scenario resolution.
            with current.experiment_reporting(), self._registry.lock_instance(current) as state:
                if state is not InstanceState.CURRENT:
                    # Reload or deletion won the gap before the reporting scope.
                    continue
                return current.rollback(release_id, operation=operation)

    def current_artifact(
        self,
        scenario: str,
        *,
        release_id: str | None = None,
    ) -> Artifact:
        current = self.get_or_create_scenario(scenario, release_id=release_id)
        if current is None:
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        return Artifact(current.repository.require_current_artifact(), current.repository)

    # -- Record acceptance -----------------------------------------------

    def accept_record(
        self,
        item: AgentRecord,
        *,
        release_id: str | None = None,
    ) -> AgentRecord:
        with self._registry.lock_for(item.scenario):
            # An instruction names a scenario that exists; inference and reports keep implicit creation.
            current = self.get_or_create_scenario(
                item.scenario,
                release_id=release_id,
                allow_implicit_creation=False if item.request_type is RequestType.TRAIN else None,
            )
            if current is None:
                raise UnknownScenario(f"unknown scenario {item.scenario!r}")
            stored, inserted = ingest.append_record(current, item)
            if inserted:
                self.process_accepted_records(current)
            return stored

    def accept_records(
        self, items: Sequence[AgentRecord], *, release_id: str | None = None
    ) -> tuple[AgentRecord, ...]:
        """Validate an ordered import batch, commit it, then wake its consumer."""
        if not items:
            raise ValueError("a record batch must not be empty")
        scenario = items[0].scenario
        if any(item.scenario != scenario or item.request_type is RequestType.TRAIN for item in items):
            raise ValueError("a record batch must contain inferences or reports from one scenario")
        with self._registry.lock_for(scenario):
            current = self.get_or_create_scenario(scenario, release_id=release_id)
            if current is None:
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            stored, inserted = ingest.append_records(current, items)
            if inserted:
                self.process_accepted_records(current)
            return stored

    def process_accepted_records(self, current: Scenario) -> None:
        """Start consumption only after the accepted records are durable."""
        # Every component's trainer sees the record: a dispatched backend wakes
        # the training thread, a local backend its own worker, and a trainer
        # without a backend consumes inline.
        self.local_workers.wake(current)
        if current.training_runtime is not None:
            return
        for bound in current.trainers:
            if bound.trainer.candidate_backend is not None:
                continue
            result = current.prepare_training_step(bound.component)
            if result is not None:
                # Commit consumption durably with the model update so recovery
                # never repeats a batch whose update was already published.
                current.commit(result, component=bound.component)

    # -- Preload ---------------------------------------------------------

    def _preload_scenarios(self, scenarios: tuple[str, ...]) -> None:
        # Preload failures land in their own training_status field: a scenario
        # that cannot be re-loaded at boot is a different failure domain from
        # a training step that cannot commit.
        for scenario in scenarios:
            try:
                current = self.get_or_create_scenario(scenario)
                if current is not None:
                    # Rows left unread before the restart train now, not once the next record arrives.
                    self.local_workers.wake(current)
            except ScenarioCapacityFull:
                # This is not a preload failure. The scenario loads on its first request, when a load slot is free.
                logger.info("scenario %r is not preloaded: the service holds its limit of loaded scenarios", scenario)
            except Exception as exc:
                logger.exception("failed to preload scenario %r", scenario)
                self._registry.record_preload_error(scenario, f"{type(exc).__name__}: {exc}")

    # -- Background workers ---------------------------------------------

    def run_operational_metrics(self) -> None:
        """Sample while training is idle, blocked, or failing, without status polling."""
        while not self.local_workers.closed.wait(self.operational_metrics_interval_seconds):
            self.record_operational_metrics()

    def record_operational_metrics(self) -> None:
        """Publish numeric state only; one unavailable scenario must not hide others."""
        for current in self._registry.loaded_scenarios():
            self.record_scenario_operational_metrics(current)

    def record_scenario_operational_metrics(self, current: Scenario) -> None:
        """Keep recipe, storage, and provider sampling failures local to one scenario."""
        try:
            metrics = current.trainer.operational_metrics()
            metrics.update(current.operations.snapshot())
            metrics["training/failed_attempts_total"] = self.errors.failure_count(current.name)
            metrics["training/error"] = int(self.errors.has_error(current.name))
            if current.training_runtime is not None:
                with self.training_loop.lock:
                    metrics["training/checkpoint_storage_blocked"] = int(self.training_loop.storage_status is not None)
            current.trainer.pipeline.experiment_logger.log(metrics, namespace="operations")
        except Exception as exc:
            logger.warning("operational metrics unavailable for scenario %r (%s)", current.name, type(exc).__name__)

    def build_training_status(self) -> Mapping[str, Any]:
        """The training status block; reading it also checks training health (see :class:`TrainingStatus`)."""
        return self.training_status.build()

    # -- Lifecycle -------------------------------------------------------

    def open_local_cycles(self) -> None:
        """Let local cycles run, and wake every loaded scenario's workers for the rows they hold."""
        self.local_workers.open()
        for current in self._registry.loaded_scenarios():
            self.local_workers.wake(current)

    def stop_local_cycles(self) -> None:
        """Local cycles start no more, and one that ends from now on commits nothing; its rows wait for the next start.

        A stopping service takes its routes away before it closes this
        dispatcher, and a harness cycle's model calls go through them: a
        call that fails comes back as a skip, which would consume the batch.
        """
        self.local_workers.stop()

    def close(self) -> None:
        """Stop all scenario workers, then close this dispatcher's runtime."""
        if self.local_workers.closed.is_set():
            return
        local_workers = self.local_workers.begin_close()
        self.training_loop.ready.set()
        if self._lifecycle.preload_thread is not None:
            self._lifecycle.preload_thread.join()
        if self._lifecycle.idle_unload_thread is not None:
            self._lifecycle.idle_unload_thread.join()
        self.training_loop.join()
        for worker in local_workers:
            worker.thread.join()
        if self._lifecycle.metrics_thread is not None:
            self._lifecycle.metrics_thread.join()
            self.record_operational_metrics()
        errors: list[BaseException] = []
        replaced = self.local_workers.take_replaced()
        for scenario in (*replaced, *self._registry.loaded_scenarios()):
            # scenario.close(), not records.close(): processor teardown has to
            # precede the store closing, or a processor worker still in flight
            # observes a closed store.
            try:
                scenario.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self._storage.close()
        except BaseException as exc:
            errors.append(exc)
        if self._recipe.runtime is not None:
            self._recipe.runtime.pause_admission()
        for component in (self._recipe.training_runtime, self._recipe.runtime):
            if component is not None:
                try:
                    component.shutdown()
                except BaseException as exc:
                    errors.append(exc)
        try:
            self._experiment_tracker.close()
        except Exception:
            logger.exception("experiment tracker failed to close")
        if errors:
            raise errors[0]


def build_default_dispatcher(
    *,
    backend_factory: RepositoryBackendFactory | None = None,
    checkpoint_strategy: CheckpointStrategy | None = None,
    local_artifact_dir: Path | None = None,
    scenario_storage: ScenarioStorage,
) -> Dispatcher:
    """Build a Dispatcher serving the core record-only ``recipe``.

    Convenience for tests and service entrypoints that want a working
    dispatcher. The recipe is the same base ``Recipe`` a deployment gets
    from ``reef.recipe: recipe``.
    Uses an in-memory artifact backend when ``backend_factory`` is not
    provided. Record and commit storage must be supplied explicitly through
    ``scenario_storage``; this helper does not select a record backend.
    """
    if backend_factory is None:
        root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        initial = root / "initial"
        initial.mkdir()
        backend_factory = InMemoryRepositoryBackend.factory(initial, root=root / "repository")
    recipe = Recipe(
        checkpoint_strategy=checkpoint_strategy if checkpoint_strategy is not None else EveryNVersions(1),
    )
    return Dispatcher(
        recipe,
        backend_factory,
        local_artifact_dir=local_artifact_dir,
        scenario_storage=scenario_storage,
    )
