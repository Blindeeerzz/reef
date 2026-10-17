"""The generated-task stages: tasks generated and saved here, rolled out from the task pool, and trained here.

When a round is due, the record stage (``GeneratedTaskRecord2Record``, a ``RoundRecord2Record``) forms a round
set. ``GeneratedTaskTasks`` runs it as one generation job on a private worker, off the trainer's thread, and
returns pending until the job finishes. A generation job:

1. Asks the generator for tasks, with the round's source sessions when the record stage gave any.
2. Saves the candidates and validates each task with the generator.
3. Publishes valid tasks with their fixed train/eval split. Eval tasks are never played.

The round's train tasks wait in the record stage's task pool. The pipeline takes up to ``tasks_per_step`` of
them as one rollout batch, and ``GeneratedTaskBatches`` plays them on its own worker: each task's training
arm ``plays_per_task`` times, then its measurement arms, with at most ``episode_capacity`` episodes queued or
running. Each episode carries the batch and its task use. The training episodes of one task use form one
group, and the batch trains on its groups once they settled. A task's first play saves its result; when every
train task of a generation has a result, the next job sends the generation's results to the generator.

The first round starts when the pipeline first looks for a batch. The next round starts after
``batches_per_round`` batches were acknowledged since the previous round started, and only while the task
pool holds fewer than ``refill_tasks`` tasks. A round that made no train task and a rollout that trained
nothing are followed at once. A restart continues from the task store: train tasks that did not train go
back to the task pool.

Only the training arm, labeled ``plain``, trains. The record stage releases reports of other arms, and
reports without an arm, such as a generator's own reports, without assembling them. A generated task
whose measurement arm is named ``plain`` is not played.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from reef.background_jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.core import AgentRecord
from reef.core.config import config_option
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.tasks import GeneratedTask, TaskGenerationRequest, TaskGenerator, TaskPlayResult
from reef.core.tasks.metadata import BATCH_LABEL, TASK_USE_LABEL, TRAINING_ARM, reported_task_name
from reef.core.tasks.split import TaskSplitConflict
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    Record2Record,
    RecordSet,
    ReportContext,
    RoundGrouping,
    RoundRecord2Record,
    RoundSet,
    TaskStatus,
)
from reef.record2task import Record2Task, ReportTasks, TaskResult
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.generator.client import DuplicateTask, Generator, GeneratorError, HttpGenerator, TaskNameConflict
from reef.record2task.generator.runner import Episode, EpisodeRunner
from reef.record2task.store import FileTaskStore, PlayedResult, StoredTask, TaskConflict, TaskStoreError
from reef.task2batch import RolloutResult, RolloutTask2Batch
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_PLAYS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
# A realigned drift spans the whole previous response, so the budget must exceed max_tokens plus the scaffold.
REALIGN_THRESHOLD = 32768
DEFAULT_BATCHES_PER_ROUND = 1
ERROR_EXCERPT_CHARS = 300
STORE_DIRECTORY = "store"
DEFAULT_EVAL_FRACTION = 0.0
DEFAULT_SPLIT_SEED = 0
DEFAULT_GENERATION_ATTEMPTS = 3
DEFAULT_GENERATION_DEADLINE_SECONDS = 1800.0
DELIVERY_RECEIPT = "delivery"


def rewards_of(episodes: tuple[Episode, ...]) -> tuple[float, ...]:
    """The rewards of the episodes that ran. An episode that ran without a reward counts as 0."""
    return tuple(
        episode.reward if episode.reward is not None else 0.0
        for episode in episodes
        if not episode.is_infrastructure_failure
    )


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation to run, with its source sessions' records; the receipt names it."""

    generation: int
    source_records: tuple[AgentRecord, ...] = ()


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """What the worker hands back for one generation: its train tasks, or none and the error after its attempts."""

    generation: int
    tasks: tuple[TaskItem, ...]
    error: str


@dataclass(frozen=True)
class DeliveryJob(SupportsReceipt):
    """Send the generator the results that it has not received; a restart submits it."""


@dataclass(frozen=True)
class DeliveryOutcome(SupportsReceipt):
    """The error of a delivery that failed, empty when every delivery went through."""

    error: str


def session_records(sessions: tuple[ReportContext, ...]) -> tuple[AgentRecord, ...]:
    """The records of the source sessions for the generator: each session's inference records, then its report.

    Sessions that share an inference record send it once.
    """
    records: dict[str, AgentRecord] = {}
    for session in sessions:
        for record in (*session.inferences, session.report):
            records.setdefault(record.agent_record_id, record)
    return tuple(records.values())


class EpisodeCapacity:
    """At most ``limit`` episodes queued or running at once, across the tasks of one rollout batch.

    A wave waits until all of its episodes fit, so a task with more plays than the limit plays in several
    waves. Create it on the event loop that uses it.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("the episode capacity must be positive")
        self.limit = limit
        self.admitted = 0
        self.changed = asyncio.Condition()

    async def admit(self, count: int) -> None:
        async with self.changed:
            await self.changed.wait_for(lambda: self.admitted + count <= self.limit)
            self.admitted += count

    async def release(self, count: int) -> None:
        async with self.changed:
            self.admitted -= count
            self.changed.notify_all()


def task_item(saved: StoredTask) -> TaskItem:
    """The task that the task pool holds for a saved train task, with its generation, split and arms."""
    task = saved.generated.task
    return TaskItem(
        saved.path,
        source_agent_record_ids=task.source_agent_record_ids,
        metadata={
            "task": {
                "name": task.name,
                "digest": task.digest,
                "round": saved.generation,
                "split": saved.split,
                "arms": [arm.name for arm in saved.generated.measurement_arms],
            }
        },
    )


def episode_assembly(scaffold_tolerance: int = DEFAULT_SCAFFOLD_TOLERANCE) -> SampleAssembly:
    """The sample assembly for an agent's episode: many model calls that extend one conversation."""
    return SampleAssembly(
        SampleAssemblySettings(
            accept_multi_turn_policy_samples=True,
            realign_threshold=REALIGN_THRESHOLD,
            scaffold_tolerance=scaffold_tolerance,
        )
    )


def source_tag_of(setting: str) -> tuple[str, str] | None:
    """Read ``source_tag`` as a tag name and value: ``"source=session"`` is the tag ``x-reef-tag-source: session``.

    An empty setting selects no source sessions.
    """
    if not setting:
        return None
    name, separator, value = setting.partition("=")
    if not separator or not name.strip() or not value:
        raise ValueError("source_tag must be name=value, the x-reef-tag-<name> header that marks a source session")
    # The service keeps tag names in lower case.
    return name.strip().lower(), value


def inference_tags(record: AgentRecord) -> Mapping[str, object]:
    """The ``x-reef-tag-*`` pairs that the service keeps under an inference record's ``metadata.tags``."""
    metadata = record.payload.get("metadata")
    tags = metadata.get("tags") if isinstance(metadata, Mapping) else None
    return tags if isinstance(tags, Mapping) else {}


class TaskGroupGrouping(RoundGrouping):
    """The training arm's episodes of one task use form one group; a source session carries ``source_tag``.

    Measurement arms and the generator's own reports share the scenario; only the training arm trains. A source
    session is a report whose inference records all carry ``source_tag``.
    """

    def __init__(self, source_tag: tuple[str, str] | None = None) -> None:
        super().__init__()
        self.source_tag = source_tag

    def is_source_report(self, context: ReportContext) -> bool:
        if self.source_tag is None or not context.inferences:
            return False
        name, value = self.source_tag
        return all(inference_tags(inference).get(name) == value for inference in context.inferences)


@dataclass(frozen=True)
class TaskGroupSettings(PluginSettings):
    """The config of ``GeneratedTaskRecord2Record``."""

    plays_per_task: int = config_option(DEFAULT_PLAYS_PER_TASK, help="The training plays of each task in a rollout.")
    batches_per_round: int = config_option(
        DEFAULT_BATCHES_PER_ROUND, help="The trained batches after which the next round starts."
    )
    source_records_per_round: int = config_option(0, help="The source sessions that each round takes, at most.")
    source_tag: str = config_option("", help="The name=value tag that marks a source session.")
    refill_tasks: int = config_option(
        1, help="The next round starts only while the task pool holds fewer saved tasks than this."
    )

    def __post_init__(self) -> None:
        if self.refill_tasks < 1:
            raise ValueError("refill_tasks must be positive")
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two: a group of one has no relative reward")
        if self.batches_per_round < 1:
            raise ValueError("batches_per_round must be positive")
        if self.source_records_per_round < 0:
            raise ValueError("source_records_per_round must be non-negative")
        if self.source_records_per_round > 0 and not self.source_tag:
            raise ValueError("source_records_per_round needs a source_tag that marks the source sessions")
        source_tag_of(self.source_tag)


@dataclass(frozen=True)
class RoundSettings(PluginSettings):
    """The config of ``GeneratedTaskTasks``."""

    rounds: int = config_option(0, help="The generations to finish; 0 generates nothing.")
    generator: Mapping[str, Any] | None = config_option(
        None, help="The TaskGenerator class (module) and its config. Required with rounds."
    )
    description: str = config_option("", help="What the tasks are about; every generation request carries it.")
    generator_url: str = config_option("", help="The generator service, ${endpoints.generator}.")
    state_dir: str = config_option("", help="Where each scenario keeps its task store.")
    served_model: str = config_option("", help="The model the task player asks for; the deployment's by default.")
    eval_fraction: float = config_option(
        DEFAULT_EVAL_FRACTION, help="The fraction of new source components held out before rollout."
    )
    split_seed: int = config_option(DEFAULT_SPLIT_SEED, help="The fixed seed for the task store's train/eval split.")
    scaffold_tolerance: int = config_option(
        DEFAULT_SCAFFOLD_TOLERANCE, help="Masked tokens the sample assembly may realign before a response."
    )
    generation_attempts: int = config_option(
        DEFAULT_GENERATION_ATTEMPTS, help="The attempts of a generation whose generator or checks fail to run."
    )
    generation_deadline_seconds: float = config_option(
        DEFAULT_GENERATION_DEADLINE_SECONDS, help="The seconds that all attempts of one generation may take."
    )

    def __post_init__(self) -> None:
        if self.rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.generation_attempts < 1 or self.generation_deadline_seconds <= 0:
            raise ValueError("generation_attempts and generation_deadline_seconds must be positive")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if isinstance(self.eval_fraction, bool) or not 0 <= self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be between 0 and 1")
        if isinstance(self.split_seed, bool) or not isinstance(self.split_seed, int):
            raise ValueError("split_seed must be an integer")
        if self.rounds > 0 and (not self.description.strip() or not self.state_dir):
            raise ValueError("rounds need a description (what the tasks are about) and a state_dir")
        if self.generator is not None:
            TaskGenerator.reference(self.generator, setting="pipeline.record2task.config.generator")


@dataclass(frozen=True)
class TaskBatchSettings(PluginSettings):
    """The config of ``GeneratedTaskBatches``."""

    tasks_per_step: int = config_option(DEFAULT_TASKS_PER_STEP, help="The saved tasks of one rollout batch.")
    episode_capacity: int = config_option(
        0, help="The episodes of a rollout batch queued or running at once; 0 is tasks_per_step x plays_per_task."
    )
    settle_timeout_seconds: float = config_option(
        600.0, help="How long a batch whose plays ended waits for its reports before its groups settle as they are."
    )

    def __post_init__(self) -> None:
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.episode_capacity < 0:
            raise ValueError("episode_capacity must be non-negative")
        if self.settle_timeout_seconds <= 0:
            raise ValueError("settle_timeout_seconds must be positive")


class GeneratedTaskRecord2Record(RoundRecord2Record):
    """The generated-task record stage: one group per task use, and a round every ``batches_per_round`` batches.

    Config: ``plays_per_task`` (the training plays of each task, at least 2), ``batches_per_round``,
    ``source_records_per_round`` (the source sessions that each request takes,
    at most), and ``source_tag`` (``name=value``; required when
    ``source_records_per_round`` is above 0). A terminal report owns its sources.
    """

    def __init__(self, context: PipelineContext, settings: TaskGroupSettings) -> None:
        self.task_grouping = TaskGroupGrouping(source_tag_of(settings.source_tag))
        #: The training plays of each task in a rollout.
        self.plays_per_task = settings.plays_per_task
        super().__init__(
            context,
            self.task_grouping,
            batches_per_round=settings.batches_per_round,
            sources_per_round=settings.source_records_per_round,
            exclusive_sources=True,
            refill_tasks=settings.refill_tasks,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(context, TaskGroupSettings.parse(config, plugin_class=cls))


class GeneratedTaskTasks(ReportTasks, BackgroundJobRunner):
    """The record-to-task stage of generated tasks: played episodes as tasks, and the generation jobs.

    The record stage forms a ``RoundSet`` when a round is due. ``make_tasks``
    submits it to the worker and returns ``TaskResult.pending()``.

    A generation job generates tasks from the round's description and source
    sessions, validates them and saves them in the task store; it plays none.
    A generator or check that fails to run is retried up to
    ``generation_attempts`` times within ``generation_deadline_seconds``. The
    round comes back ``READY`` with the saved train tasks, or ``SKIPPED`` when
    it made none or its attempts ran out.

    ``GeneratedTaskBatches`` plays saved tasks through ``play_and_record``,
    which saves a task's first result. When every train task of a generation
    has a result, the next job sends the generation's results to the generator.
    A restart puts the train tasks that did not train back in the record
    stage's task pool and sends the results that the generator did not receive.
    """

    def __init__(
        self,
        context: PipelineContext,
        settings: RoundSettings,
        round_stage: GeneratedTaskRecord2Record,
        *,
        task_generator: TaskGenerator | None = None,
        task_player: Generator | None = None,
        worker: BackgroundJobWorker | None = None,
    ) -> None:
        #: The record stage that forms the round sets and holds the task pool.
        self.round_stage = round_stage
        self.assembly = episode_assembly(settings.scaffold_tolerance)
        self.scenario = context.scenario
        self.plays_per_task = round_stage.plays_per_task
        self.rounds = settings.rounds
        self.description = settings.description
        self.served_model = settings.served_model or None
        self.generation_attempts = settings.generation_attempts
        self.generation_deadline_seconds = settings.generation_deadline_seconds
        # Every scenario keeps its own task store.
        self.state_dir = Path(settings.state_dir).expanduser() / self.scenario if settings.state_dir else None
        if self.rounds > 0:
            if task_generator is None:
                raise ValueError("rounds need a task generator: set pipeline.record2task.config.generator")
            if task_player is None:
                generator_url = settings.generator_url
                if not generator_url or "${" in generator_url:
                    raise ValueError(
                        "rounds need generator_url: the generator service's address, ${endpoints.generator} "
                        "when the deployment has a generator section"
                    )
                task_player = HttpGenerator(generator_url)
        self.task_generator = task_generator
        self.runner = EpisodeRunner(task_player) if task_player is not None else None
        self.worker = worker
        if self.worker is None and self.rounds > 0:
            self.worker = BackgroundJobWorker(self, thread_name="generated-task-rounds", concurrency=1)
        if self.task_generator is not None:
            logger.info("scenario %r generates tasks with %s", self.scenario, type(self.task_generator).__name__)
        self.task_store = (
            FileTaskStore(
                root=self.state_dir / STORE_DIRECTORY, eval_fraction=settings.eval_fraction, seed=settings.split_seed
            )
            if self.state_dir is not None and self.rounds > 0
            else None
        )
        #: Results that a play could not save, by generation; the generation worker saves them before it delivers.
        self.unsaved_results: queue.SimpleQueue[tuple[int, PlayedResult]] = queue.SimpleQueue()
        round_stage.resume_rounds(
            completed_rounds=self.task_store.complete_generation_count() if self.task_store is not None else 0,
            round_limit=self.rounds,
        )
        if self.task_store is not None:
            for saved in self.task_store.available_tasks():
                round_stage.add_tasks((task_item(saved),), group_key=saved.generation)
            # A task that trained before its result was saved does not play again; the generator learns it was lost.
            for saved in self.task_store.tasks(split="train"):
                name = saved.generated.task.name
                trained = self.task_store.generation(generation=saved.generation).trained_task_names
                if name in trained and not self.task_store.has_result(task_name=name):
                    self.task_store.record_result(
                        generation=saved.generation,
                        result=PlayedResult(
                            task_name=name,
                            error="the result was lost before it was saved",
                            infrastructure_failure=True,
                        ),
                    )
        #: The set ID of each submitted job, by the job's receipt.
        self.job_set_ids: dict[str, Hashable] = {}
        self.generation_in_flight: int | None = None
        self.last_error = ""
        # Results that a stop kept from the generator go out now, even when no other job is due.
        if self.task_store is not None and self.worker is not None and self.task_store.deliverable_generations():
            self.worker.submit(DeliveryJob(receipt=DELIVERY_RECEIPT))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> Self:
        """Build the rounds of one scenario, with the ``TaskGenerator`` that ``generator`` names."""
        settings = RoundSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, GeneratedTaskRecord2Record):
            raise PluginConfigError(
                f"{cls.__name__} needs GeneratedTaskRecord2Record as record2record, not {type(record2record).__name__}"
            )
        task_generator = None
        if settings.generator is not None:
            generator = TaskGenerator.reference(settings.generator, setting="pipeline.record2task.config.generator")
            task_generator = TaskGenerator.referenced_class(generator).from_config(
                generator.config, scenario=context.scenario, environ=environ
            )
        return cls(context, settings, record2record, task_generator=task_generator)

    @classmethod
    def scenario_state_dirs(cls, config: Mapping[str, Any], scenario: str) -> tuple[Path, ...]:
        state_dir = RoundSettings.parse(config, plugin_class=cls).state_dir
        return (Path(state_dir).expanduser() / scenario,) if state_dir else ()

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("a generated task's report requires metadata.task.name, as the task player sends it")
        sample = self.assembly.build(context, context.require_score())
        return replace(sample, group_id=task_name)

    def make_tasks(self, record_set: RecordSet) -> TaskResult:
        """Submit a round set to the worker and return pending; make a sample from any other set."""
        if not isinstance(record_set, RoundSet):
            return super().make_tasks(record_set)
        if self.task_store is None:
            raise RuntimeError("a round needs a task store: set pipeline.record2task.config.state_dir")
        records = session_records(record_set.sources)
        completed = self.task_store.completed_source_record_ids()
        remaining = tuple(record for record in records if record.agent_record_id not in completed)
        # A restart replays sessions that a complete generation already used; their tasks are saved.
        if records and not remaining:
            return TaskResult(TaskStatus.INGESTED)
        existing = self.task_store.find_generation(
            source_record_ids=tuple(record.agent_record_id for record in remaining)
        )
        generation = self.task_store.next_generation() if existing is None else existing.generation
        job = GenerationJob(receipt=f"generation-{generation:05d}", generation=generation, source_records=remaining)
        self.generation_in_flight = generation
        if self.worker is None or not self.worker.submit(job):
            raise RuntimeError(f"the round worker is closed or broken; {job.receipt} cannot run")
        self.job_set_ids[job.receipt] = record_set.set_id
        return TaskResult.pending()

    def take_finished(self) -> tuple[tuple[Hashable, TaskResult], ...]:
        """Hand back each finished generation."""
        if self.worker is None:
            return ()
        finished: list[tuple[Hashable, TaskResult]] = []
        for outcome in self.worker.poll():
            if outcome.receipt == DELIVERY_RECEIPT:
                self.last_error = (
                    outcome.error if isinstance(outcome, DeliveryOutcome) else "the delivery did not finish"
                )
                continue
            set_id = self.job_set_ids.pop(outcome.receipt)
            if isinstance(outcome, Failed):
                self.last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("generated-task %s failed on scenario %r", outcome.receipt, self.scenario)
                self.generation_in_flight = None
                finished.append((set_id, TaskResult.skipped()))
                continue
            if isinstance(outcome, GenerationOutcome):
                self.generation_in_flight = None
                self.last_error = outcome.error
                logger.info(
                    "generated-task generation %d on scenario %r: %d train tasks",
                    outcome.generation,
                    self.scenario,
                    len(outcome.tasks),
                )
                finished.append((set_id, TaskResult.ready(outcome.tasks) if outcome.tasks else TaskResult.skipped()))
                continue
            raise TypeError("the round worker must hand back a GenerationOutcome")
        return tuple(finished)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()
        if self.task_store is not None:
            self.task_store.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {"generation_in_flight": 0 if self.generation_in_flight is None else 1}

    def status(self) -> Mapping[str, Any]:
        completed = self.task_store.complete_generation_count() if self.task_store is not None else 0
        if self.generation_in_flight is not None:
            next_generation = self.generation_in_flight + 1
        else:
            next_generation = self.task_store.next_generation() if self.task_store is not None else 0
        return {
            "task_store": self.task_store.counts() if self.task_store is not None else {},
            "task_generation": {
                "generator": None if self.task_generator is None else type(self.task_generator).__name__,
                "in_flight": self.generation_in_flight,
                "next": next_generation,
                "completed": completed,
                "of": self.rounds,
                "last_error": self.last_error,
            },
        }

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Run one job on the worker, after sending the generator any results that it has not received."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a job needs a task generator and a state_dir")
        self.save_unsaved_results()
        delivery_error = await self.deliver_results()
        if isinstance(job, DeliveryJob):
            return DeliveryOutcome(receipt=job.receipt, error=delivery_error)
        if isinstance(job, GenerationJob):
            generation = await self.run_generation(job)
            # A generation without train tasks has its complete results now.
            delivery_error = await self.deliver_results() or delivery_error
            return replace(generation, error=generation.error or delivery_error)
        raise TypeError(f"the round worker takes GenerationJob or DeliveryJob, not {type(job).__name__}")

    async def run_generation(self, job: GenerationJob) -> GenerationOutcome:
        """Generate, validate and save one generation; retry a failure to run within the attempts and deadline."""
        if self.task_store is None:
            raise RuntimeError("a generation needs a state_dir")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.generation_deadline_seconds
        error = ""
        for attempt in range(1, self.generation_attempts + 1):
            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                break
            # The generator is a deployment's plugin: whatever it raises is a failure to run.
            try:
                await asyncio.wait_for(self.generated(job), remaining_seconds)
            except Exception as exc:
                error = str(exc) or type(exc).__name__
                logger.exception(
                    "generated-task generation %d attempt %d failed on scenario %r",
                    job.generation,
                    attempt,
                    self.scenario,
                )
                continue
            state = self.task_store.generation(generation=job.generation)
            saved = {saved.generated.task.name: saved for saved in self.task_store.tasks(split="train")}
            tasks = tuple(task_item(saved[name]) for name in state.train_task_names)
            return GenerationOutcome(receipt=job.receipt, generation=job.generation, tasks=tasks, error="")
        return GenerationOutcome(
            receipt=job.receipt, generation=job.generation, tasks=(), error=error or "the generation deadline passed"
        )

    async def generated(self, job: GenerationJob) -> None:
        """One attempt: continue from the saved candidates and validation outcomes, then complete the generation."""
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("a generation needs a task generator and a state_dir")
        generation = self.task_store.begin_generation(
            generation=job.generation,
            source_record_ids=tuple(record.agent_record_id for record in job.source_records),
        )
        if generation.candidates is None:
            generated = await self.task_generator.generate(
                TaskGenerationRequest(job.source_records, self.description, generation=job.generation)
            )
            generation = self.task_store.save_candidates(generation=job.generation, candidates=generated)
        for index in generation.pending_indices:
            item = generation.candidate(index=index)
            if any(arm.name == TRAINING_ARM for arm in item.measurement_arms):
                refusal = f"the measurement arm {TRAINING_ARM!r} has the name of the training arm"
                self.task_store.reject_task(generation=job.generation, index=index, reason=refusal)
                continue
            validation = await self.task_generator.validate(item.task)
            if validation.is_valid:
                try:
                    self.task_store.publish_task(generation=job.generation, index=index)
                except (TaskConflict, TaskSplitConflict) as conflict:
                    self.task_store.reject_task(generation=job.generation, index=index, reason=str(conflict))
            else:
                self.task_store.reject_task(
                    generation=job.generation, index=index, reason="; ".join(validation.errors)
                )
        self.task_store.complete_generation(generation=job.generation)

    async def play_and_record(
        self, saved: StoredTask, *, labels: Mapping[str, str], capacity: EpisodeCapacity | None = None
    ) -> tuple[TaskPlayResult, tuple[str, ...]]:
        """Play one saved task with ``labels`` on its episodes; save its result in the task store at its first play.

        Returns the result and the reports of its training episodes.
        """
        if self.task_store is None:
            raise RuntimeError("a play needs a state_dir")
        name = saved.generated.task.name
        try:
            result, reports = await self.played(saved.generated, saved.generation, labels=labels, capacity=capacity)
        except Exception as exc:
            logger.exception("generated-task %s could not play on scenario %r", name, self.scenario)
            result = TaskPlayResult(saved.generated.task, error=str(exc) or type(exc).__name__)
            result = replace(result, infrastructure_failure=True)
            reports = ()
        # The generator learns from a task's first play; a rollout after a restart only trains.
        if not self.task_store.has_result(task_name=name):
            played_result = PlayedResult(
                task_name=name,
                training_rewards=result.training_rewards,
                measurement_rewards=dict(result.measurement_rewards),
                error=result.error,
                infrastructure_failure=result.infrastructure_failure,
                reported_plays=len(reports),
            )
            try:
                self.task_store.record_result(generation=saved.generation, result=played_result)
            except (OSError, TaskStoreError):
                # The episodes still train; the next job saves the result again before it delivers.
                logger.exception("the result of generated-task %s was not saved on scenario %r", name, self.scenario)
                self.unsaved_results.put((saved.generation, played_result))
        return result, reports

    async def played(
        self,
        item: GeneratedTask,
        generation: int,
        *,
        labels: Mapping[str, str],
        capacity: EpisodeCapacity | None = None,
    ) -> tuple[TaskPlayResult, tuple[str, ...]]:
        """Stage one saved task and play it through the episode runner: the training arm, then each measurement arm.

        Returns the task's result and the reports of its training episodes, none for a task that was not
        played. A task whose training episodes all could not run comes back with the first episode's error
        as an infrastructure failure, and its measurement arms are not played. With ``capacity``, each arm plays
        in waves that the capacity admits.
        """
        if self.runner is None:
            raise RuntimeError("playing a task needs the generator service")
        task = item.task
        try:
            task_path = await self.runner.stage(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskPlayResult(task, error=f"refused: {exc}"), ()
        labels = {"round": str(generation), **labels}
        training = await self.played_arm(task_path, TRAINING_ARM, self.plays_per_task, (), labels, capacity)
        if training and all(episode.is_infrastructure_failure for episode in training):
            error = f"the agent could not play the task: {training[0].error[:ERROR_EXCERPT_CHARS]}"
            return TaskPlayResult(task, error=error, infrastructure_failure=True), ()
        reports = tuple(report_id for episode in training for report_id in episode.report_record_ids)
        measurement_rewards: dict[str, tuple[float, ...]] = {}
        for arm in item.measurement_arms:
            try:
                episodes = await self.played_arm(
                    task_path, arm.name, arm.count, arm.extra_instruction_files, labels, capacity
                )
            except GeneratorError:
                # The training episodes still train; the arm measures nothing.
                logger.exception(
                    "generated-task %s arm %s could not play on scenario %r", task.name, arm.name, self.scenario
                )
                episodes = ()
            measurement_rewards[arm.name] = rewards_of(episodes)
        return TaskPlayResult(task, rewards_of(training), measurement_rewards), reports

    async def played_arm(
        self,
        task_path: Path,
        arm: str,
        plays: int,
        extra_instruction_files: tuple[str, ...],
        labels: Mapping[str, str],
        capacity: EpisodeCapacity | None,
    ) -> tuple[Episode, ...]:
        """Play one arm of a staged task, in waves of at most the capacity's limit when there is one."""
        if self.runner is None:
            raise RuntimeError("playing a task needs the generator service")
        wave_size = plays if capacity is None else capacity.limit
        episodes: list[Episode] = []
        for start in range(0, plays, wave_size):
            count = min(wave_size, plays - start)
            if capacity is not None:
                await capacity.admit(count)
            try:
                episodes.extend(
                    await self.runner.play(
                        task_path,
                        scenario=self.scenario,
                        arm=arm,
                        plays=count,
                        extra_instruction_files=extra_instruction_files,
                        labels=labels,
                        model=self.served_model,
                    )
                )
            finally:
                if capacity is not None:
                    await capacity.release(count)
        return tuple(episodes)

    def save_unsaved_results(self) -> None:
        """Save the results that a play could not save; one that still fails waits for the next job."""
        if self.task_store is None:
            return
        failed: list[tuple[int, PlayedResult]] = []
        while not self.unsaved_results.empty():
            generation, result = self.unsaved_results.get_nowait()
            if self.task_store.has_result(task_name=result.task_name):
                continue
            try:
                self.task_store.record_result(generation=generation, result=result)
            except (OSError, TaskStoreError):
                logger.exception("the result of generated-task %s is still not saved", result.task_name)
                failed.append((generation, result))
        for entry in failed:
            self.unsaved_results.put(entry)

    def deliver_later(self) -> None:
        """Send the generator the results that it has not received, on the worker, after the jobs before it."""
        if self.worker is not None:
            self.worker.submit(DeliveryJob(receipt=DELIVERY_RECEIPT))

    async def deliver_results(self) -> str:
        """Send each generation whose train tasks all have results to the generator, oldest first.

        Returns the error of the last delivery that failed; a failed generation waits for the next job, and later
        generations still go out.
        """
        if self.task_generator is None or self.task_store is None:
            raise RuntimeError("delivering results needs a task generator and a state_dir")
        error = ""
        for state in self.task_store.deliverable_generations():
            try:
                await self.task_generator.record_results(state.task_results(), generation=state.generation)
            except Exception as exc:
                logger.exception("the task generator did not take the results of generation %d", state.generation)
                error = str(exc) or type(exc).__name__
                continue
            self.task_store.mark_delivered(generation=state.generation)
        return error


@dataclass(frozen=True)
class RolloutJob(SupportsReceipt):
    """One rollout batch to play: each task use and its saved task's name. The receipt is the batch."""

    task_uses: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RolloutOutcome(SupportsReceipt):
    """The reports of each task use's training episodes in one rollout batch."""

    training_reports: Mapping[str, tuple[str, ...]]


class GeneratedTaskBatches(RolloutTask2Batch, BackgroundJobRunner):
    """Roll saved tasks out as batches, and batch the training groups of each rollout.

    The pipeline submits up to ``tasks_per_step`` saved tasks from the task pool as one rollout batch. This
    stage plays them on its own worker through the record-to-task stage's episode runner. Each episode
    carries the batch and the task use, and at most ``episode_capacity`` episodes are queued or running at
    once. A task's first result goes to the task store, and the record-to-task stage sends finished
    generations' results to the generator. The tasks of groups that trained are marked trained in the store.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        settings = TaskBatchSettings.parse(config, plugin_class=cls)
        if not isinstance(record2record, GeneratedTaskRecord2Record) or not isinstance(
            record2task, GeneratedTaskTasks
        ):
            raise PluginConfigError(
                f"{cls.__name__} needs GeneratedTaskRecord2Record and GeneratedTaskTasks, "
                f"not {type(record2record).__name__} and {type(record2task).__name__}"
            )
        return cls(context, settings, record2record=record2record, record2task=record2task)

    def __init__(
        self,
        context: PipelineContext,
        settings: TaskBatchSettings,
        *,
        record2record: GeneratedTaskRecord2Record,
        record2task: GeneratedTaskTasks,
        worker: BackgroundJobWorker | None = None,
    ) -> None:
        super().__init__(context, settings.tasks_per_step)
        self.record2record = record2record
        self.record2task = record2task
        self.settle_timeout_seconds = settings.settle_timeout_seconds
        self.episode_capacity = settings.episode_capacity or settings.tasks_per_step * record2task.plays_per_task
        self.worker = worker or BackgroundJobWorker(self, thread_name="generated-task-rollouts", concurrency=1)
        self.running_batch: str | None = None
        #: The capacity of the rollout batch that plays, for status.
        self.capacity: EpisodeCapacity | None = None

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:generated-tasks:{batch_number}", tasks)

    def submit(self, batch: str, tasks: tuple[TaskItem, ...]) -> tuple[str, ...]:
        task_uses = tuple(f"{batch}/{task.task_path.name}" for task in tasks)
        job = RolloutJob(
            receipt=batch,
            task_uses=tuple((task_use, task.task_path.name) for task_use, task in zip(task_uses, tasks, strict=True)),
        )
        if not self.worker.submit(job):
            raise RuntimeError(f"the rollout worker is closed or broken; rollout {batch} cannot run")
        self.running_batch = batch
        return task_uses

    def take_finished(self) -> tuple[RolloutResult, ...]:
        finished: list[RolloutResult] = []
        for outcome in self.worker.poll():
            self.running_batch = None
            if isinstance(outcome, RolloutOutcome):
                finished.append(RolloutResult(outcome.receipt, outcome.training_reports))
            else:
                logger.error("generated-task rollout %s did not finish on scenario %r", outcome.receipt, self.scenario)
                finished.append(RolloutResult(outcome.receipt, {}))
            self.record2task.deliver_later()
        return tuple(finished)

    def running(self) -> bool:
        return self.running_batch is not None

    def rollout_trained(self, batch: str, task_uses: tuple[str, ...]) -> None:
        if self.record2task.task_store is not None:
            self.record2task.task_store.mark_trained(
                task_names=tuple(task_use.rsplit("/", 1)[1] for task_use in task_uses)
            )

    def status(self) -> Mapping[str, Any]:
        return {
            "rollout_capacity": {
                "episodes": self.episode_capacity,
                "admitted_episodes": 0 if self.capacity is None else self.capacity.admitted,
                "running_batch": self.running_batch,
            }
        }

    def close(self) -> None:
        self.worker.close()

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Play one rollout batch: its tasks at once, their episodes within the capacity."""
        if not isinstance(job, RolloutJob):
            raise TypeError(f"the rollout worker takes RolloutJob, not {type(job).__name__}")
        record2task = self.record2task
        if record2task.task_store is None:
            raise RuntimeError("a rollout needs the record-to-task stage's task store")
        saved_by_name = {saved.generated.task.name: saved for saved in record2task.task_store.tasks(split="train")}
        capacity = EpisodeCapacity(self.episode_capacity)
        self.capacity = capacity

        async def played(task_use: str, name: str) -> tuple[str, tuple[str, ...]]:
            labels = {BATCH_LABEL: job.receipt, TASK_USE_LABEL: task_use}
            try:
                _, reports = await record2task.play_and_record(saved_by_name[name], labels=labels, capacity=capacity)
            except Exception:
                # The other tasks of the batch play on; this task rolls out again after a restart.
                logger.exception(
                    "generated-task %s failed in rollout %s on scenario %r", name, job.receipt, self.scenario
                )
                return task_use, ()
            return task_use, reports

        reports = await asyncio.gather(*(played(task_use, name) for task_use, name in job.task_uses))
        return RolloutOutcome(receipt=job.receipt, training_reports=dict(reports))
