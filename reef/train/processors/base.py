"""The task-to-batch contract: tasks become batches.

A record set from the record-to-record stage becomes tasks
(:class:`~reef.record2task.Record2Task`), and ``DataProcessor`` shapes the
tasks into batches. A task is a ``TrainDataItem``. The two engines that
implement it for recipes live beside this module: ``reported`` (feedback
received in a report) and ``computed`` (feedback computed from traffic).
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.core.training_request import TrainingRequest
from reef.observability import ExperimentLogger
from reef.record2record import AssemblyStatus, PassThroughStage, Record2RecordStage
from reef.record2task import NoTasks, Record2Task
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch


@dataclass(frozen=True)
class InstructionFailure:
    """A failed instruction and the attempt metadata carried through an in-process trainer reload."""

    error: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


class DataProcessor:
    """The processor contract: turn records into typed training batches.

    Which base a recipe builds on is one question — how does its feedback
    arrive?

    * As **reports** referencing inference records → subclass
      :class:`~reef.train.processors.reported.ReportedFeedbackProcessor` and write
      ``make_sample`` (assemble feedback) and ``make_batch`` (what a batch looks like).
      A grouped method's recipe gives the stage its ``ReportGrouping``. The
      record-to-record stage owns everything between them.
    * **Computed from the traffic itself** — correlated across records,
      judged by a model, landing asynchronously → subclass
      :class:`~reef.train.processors.computed.ComputedFeedbackProcessor`, write
      ``make_sample`` and ``make_batch``. The recipe gives the stage a
      ``RecordJudging`` (the method's correlation in ``observe``, and its
      ``async def judge``). The stage hands each completed record's job to a
      worker thread, so ``judge`` may call models and take minutes without
      blocking serving.

    Every processor sends records through its record-to-record stage, which
    decides which records train and in what order. The recipe builds the
    stage with its own policy (``WeightTrainingRecipe.build_record_stage``)
    and passes it as ``record_stage``. The processor assembles the stage's
    record sets into samples and batches.

    ``DataProcessor`` itself is never a recipe's processor. Instantiated
    bare it is the no-update default: it leaves records in storage
    but never becomes ready and never produces a
    batch. The tradeoff of folding that default into the base (rather than
    keeping it abstract with a separate ``NoUpdateProcessor``) is that a
    half-written subclass that forgets to override ``build_batch`` silently
    becomes a no-op instead of failing at construction. Recipes that go
    through :class:`WeightTrainingRecipe.build` are still guarded — it raises
    ``TypeError`` unless the recipe declares a concrete ``processor``.

    Whatever implements this contract, the trainer-facing surface stays
    synchronous: ``ingest``/``ready``/``build_batch`` run on the trainer's
    thread under its lock and must never block on network or model
    latency. Judging that calls models therefore runs on a background
    thread — the judging stage owns one, starts it on demand, exchanges
    results through thread-safe queues, and releases it in :meth:`close`,
    the teardown hook the owning scenario guarantees to call when the
    processor is dropped (dispatcher shutdown and durable reload both go
    through it). State that lives only in processor memory is rebuilt by
    replaying ingest after a restart: the trainer replays un-acknowledged
    records, so the recompute cost is bounded by the training backlog, not
    history.
    """

    required_request_types: frozenset[RequestType] = frozenset({RequestType.INFERENCE, RequestType.REPORT})
    supported_training_modes: frozenset[str] = frozenset({"auto"})

    def __init__(
        self,
        context: ProcessorContext,
        *,
        record_stage: Record2RecordStage | None = None,
        record2task: Record2Task | None = None,
    ) -> None:
        self._context = context
        # The two modes that take an instruction go together: a recipe that cannot run one cannot run it in either.
        if not context.config.get("manual_enabled", True):
            self.supported_training_modes = self.supported_training_modes - {"manual", "hybrid"}
        # Validate the initial mode before subclasses initialize their buffers.
        DataProcessor.set_training_mode(self, context.training_mode)
        self._training_requests: dict[str, TrainingRequest] = {}
        self._consumed_requests: set[str] = set()
        # The error of each buffered instruction whose step failed; its next batch is a skip row, not a run.
        self._request_failures: dict[str, InstructionFailure] = {}
        self._scenario = context.scenario
        self._batch_size = int(context.config.get("batch_size", 1))
        if self._batch_size <= 0:
            raise ValueError("batch_size must be positive")
        #: The batch handed out and not yet acknowledged. While it exists the
        #: processor is ready, hands out the same object, and must not
        #: reshuffle what it references.
        self._pending: TrainingBatch | None = None
        self._batch_number = 0
        #: The record-to-record stage that turns this processor's records into record sets.
        self.record_stage = self.default_record_stage() if record_stage is None else record_stage
        #: The record-to-task stage that turns each record set into tasks.
        self.record2task = self.default_record2task() if record2task is None else record2task
        #: Tasks by the ID of the record set that made them, until the set trains or ends.
        self.tasks: dict[Hashable, tuple[TrainDataItem, ...]] = {}

    def default_record_stage(self) -> Record2RecordStage:
        """Return the stage to use when the recipe gives none. It has no method policy: each record passes through."""
        return PassThroughStage()

    def default_record2task(self) -> Record2Task:
        """Return the record-to-task stage to use when the recipe gives none. The default makes no tasks."""
        return NoTasks()

    @property
    def context(self) -> ProcessorContext:
        return self._context

    @property
    def scenario(self) -> str:
        return self._scenario

    @property
    def training_mode(self) -> str:
        """The batching policy selected for this processor."""
        return self._context.training_mode

    def set_training_mode(self, training_mode: str) -> None:
        """Select future batches while preserving shared buffers and reservations."""
        if training_mode not in ("auto", "manual", "hybrid"):
            raise ValueError("training_mode must be 'auto', 'manual' or 'hybrid'")
        if training_mode not in self.supported_training_modes:
            raise NotImplementedError(f"{type(self).__name__} does not implement training_mode={training_mode!r}")
        self._context = replace(self._context, training_mode=training_mode)

    def admit_reports_of(self, report_type: type[ReportBase] | None) -> None:
        """Name the contract the scenario's ingress admits when it is wider than this processor's own."""
        self._context = replace(self._context, admitted_report_type=report_type)
        self.record_stage.admit_reports_of(report_type)

    def buffered_requests(self) -> int:
        """How many instructions are read into memory and not yet consumed."""
        return len(self._training_requests)

    def request_failure(self, request_id: str) -> str | None:
        """What the instruction's failed step said, when it failed at all."""
        failure = self._request_failures.get(request_id)
        return None if failure is None else failure.error

    def request_failure_metrics(self, request_id: str) -> Mapping[str, Any]:
        """The exact failed attempt's metadata, if the backend had produced any."""
        failure = self._request_failures.get(request_id)
        return {} if failure is None else failure.metrics

    def request_failures(self) -> Mapping[str, InstructionFailure]:
        """The failed instructions still buffered, by id, with what their step said."""
        return dict(self._request_failures)

    def mark_request_failed(self, request_id: str, error: str, metrics: Mapping[str, Any] | None = None) -> None:
        """Record that the instruction's step failed; its next batch is consumed with a skip row."""
        self._request_failures[request_id] = InstructionFailure(error, dict(metrics or {}))

    def set_request_failures(self, failures: Mapping[str, InstructionFailure]) -> None:
        """Carry the failed instructions of a replaced processor into this one."""
        self._request_failures = dict(failures)

    @property
    def experiment_logger(self) -> ExperimentLogger:
        """The scenario logger shared by its recipe, processor, and backend."""
        return self._context.experiment_logger

    #: The batch type ``build_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch

    def forget_sets(self, set_ids: frozenset[Hashable]) -> None:
        """Forget the tasks of sets that the stage ended."""
        for set_id in set_ids:
            self.tasks.pop(set_id, None)

    def assemble_formed(self) -> None:
        """Make tasks from each set that the stage formed, give each result back, and forget the sets that ended."""
        for record_set in self.record_stage.take_formed():
            result = self.record2task.make_tasks(record_set)
            if result.assembly.status is AssemblyStatus.READY:
                self.tasks[record_set.set_id] = result.tasks
            self.record_stage.settle(record_set, result.assembly)
        self.forget_sets(self.record_stage.take_ended())

    def ingest(self, item: AgentRecord) -> None:
        """Read one record. Training instructions stay here; every other record goes through the record stage."""
        if item.request_type is RequestType.TRAIN:
            if item.scenario != self.scenario:
                raise ValueError("training records must belong to the processor's scenario")
            request = replace(TrainingRequest.from_dict(item.payload), id=item.agent_record_id)
            if request.id not in self._consumed_requests:
                self._training_requests.setdefault(request.id, request)
            return
        self.record_stage.ingest(item)
        self.assemble_formed()

    # ------------------------------------------------------------ batch cycle
    #
    # The shape is the same for every processor: batch when enough units are
    # held (auto, hybrid) or an instruction is queued (manual, hybrid), hand the
    # same batch out until it is acknowledged, then release what it consumed.
    # An engine fills in the three things that differ: what a unit is, how
    # the selected ones become a batch, and what consuming them releases.

    def ready(self) -> bool:
        # Background work in the stage, such as judgments, forms sets without a new record.
        self.record_stage.poll(time.monotonic())
        self.assemble_formed()
        if self._pending is not None:
            return True
        if self.training_mode == "manual":
            return bool(self._training_requests)
        if self.training_mode == "hybrid" and self._training_requests:
            return True
        return self._ready_count() >= self._batch_size

    def build_batch(self) -> TrainingBatch:
        if self._pending is None:
            if not self.ready():
                raise RuntimeError(f"{type(self).__name__} batch is not ready")
            self._batch_number += 1
            # Manual and hybrid run the oldest queued instruction first; auto leaves the queue to a mode that takes it.
            request = (
                next(iter(self._training_requests.values()))
                if self.training_mode != "auto" and self._training_requests
                else None
            )
            self._pending = self.make_training_batch(self._batch_number, request)
            if request is not None:
                self._pending = replace(
                    self._pending, batch_id=f"{self.scenario}:instruction:{request.id}", request=request
                )
            self.record_stage.mark_batch_out(True)
        return self._pending

    def make_training_batch(self, batch_number: int, request: TrainingRequest | None) -> TrainingBatch:
        """Select inputs for one batch; in ``manual`` and ``hybrid`` a queued instruction arrives as ``request``.

        Override this single assembly hook to take instructions. Ingestion,
        acknowledgement and buffer release operate on the same state in every mode.
        With a request the hook's own batch id is replaced by
        ``<scenario>:instruction:<request id>`` and the request is attached.
        """
        if request is not None:
            raise NotImplementedError(f"{type(self).__name__} does not implement instruction batch assembly")
        return self._make_pending(batch_number)

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        if self._pending is None or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        consumed = self._consume_pending()
        if self._pending.request is not None:
            request_id = self._pending.request.id
            self._training_requests.pop(request_id)
            self._request_failures.pop(request_id, None)
            self._consumed_requests.add(request_id)
            consumed = consumed | {request_id}
        self._pending = None
        self.record_stage.mark_batch_out(False)
        return consumed

    def dropped(self, batch_id: str) -> None:
        """The trainer says the batch is acknowledged without training: the backend dropped it as stale."""
        self.record_stage.mark_batch_dropped()

    def release_batch(self, batch_id: str) -> None:
        """Forget the handed out batch without consuming anything; the next ``build_batch`` selects again."""
        if self._pending is None or self._pending.batch_id != batch_id:
            raise ValueError(f"unknown batch_id {batch_id!r}")
        self._pending = None
        self.record_stage.mark_batch_out(False)

    def discard_request(self, request_id: str) -> frozenset[str]:
        """Consume one queued instruction without a batch; the committed row that names it is what recovery skips."""
        if self._pending is not None and self._pending.request is not None and self._pending.request.id == request_id:
            raise ValueError(f"training request {request_id!r} is reserved; release its batch first")
        if request_id not in self._training_requests:
            raise ValueError(f"unknown training request {request_id!r}")
        self._training_requests.pop(request_id)
        self._request_failures.pop(request_id, None)
        self._consumed_requests.add(request_id)
        return frozenset({request_id})

    def _ready_count(self) -> int:
        """How many batch-ready units are held.

        Zero is the no-update default, and it is what makes a bare
        ``DataProcessor`` ingest for audit without ever becoming ready.
        """
        return 0

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        """Select this batch's units and shape them through ``make_batch``."""
        raise RuntimeError(f"{type(self).__name__} never produces a training batch")

    def _consume_pending(self) -> frozenset[str]:
        """Release what the acknowledged batch consumed and name its records.

        The returned ids ride the step's commit record so recovery can skip
        them when rebuilding processor memory: a record a committed batch
        consumed must never train twice, even when retention keeps it stored.
        The no-update default consumes nothing.
        """
        return frozenset()

    def releasable_record_ids(self) -> frozenset[str]:
        """Records whose buffered state can be released after a durable commit."""
        return frozenset(self._consumed_requests) | self.record_stage.releasable_record_ids()

    def release_records(self, agent_record_ids: frozenset[str]) -> None:
        """Release committed in-memory state without changing stored records."""
        self._consumed_requests -= agent_record_ids
        self.record_stage.release_records(agent_record_ids)

    def derivation_pending(self) -> bool:
        """Whether background derivation could flip ``ready`` without records.

        The training worker sleeps until the next accepted record; a
        processor whose judgments land asynchronously (or whose sessions
        flush on a TTL) returns ``True`` here so the worker polls on a
        bounded interval instead. Read on the training thread between
        drains — implementations must not block.
        """
        return self.record_stage.derivation_pending() or self.record2task.derivation_pending()

    def operational_metrics(self) -> Mapping[str, float | int]:
        """Numeric queue state, sampled under the trainer lock without advancing work."""
        return {
            "buffered_requests": self.buffered_requests(),
            **self.record_stage.operational_metrics(),
            **self.record2task.operational_metrics(),
        }

    def status(self) -> Mapping[str, Any]:
        """Return JSON-safe state that callers need while waiting.

        ``ready_units`` against ``batch_size`` says how close the next batch
        is. A caller waiting on a release needs it to tell a batch that is
        still filling from one that never will: a scenario whose producer
        stopped short, or whose record stream picked up rows the producer did
        not send, sits below the batch size with nothing else to report, and
        without these two the wait is indistinguishable from slow training.

        A processor may override this for a terminal outcome that cannot
        become a training batch, allowing a bounded external wait to fail
        explicitly.
        """
        state: dict[str, Any] = {"ready_units": self._ready_count(), "batch_size": self._batch_size}
        if self.supported_training_modes & {"manual", "hybrid"}:
            state["buffered_requests"] = self.buffered_requests()
        return {**state, **self.record_stage.status(), **self.record2task.status()}

    def close(self) -> None:
        """Release resources the processor owns; safe to call more than once.

        The default closes the record stage and the record-to-task stage.
        Processors with other background work override this to signal their
        workers and join them; after ``close`` returns, no thread of the
        processor may touch shared state or deliver further results.
        """
        try:
            self.record_stage.close()
        finally:
            self.record2task.close()
