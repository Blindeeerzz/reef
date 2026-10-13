"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.core.plugins import PluginConfigError, PluginReference
from reef.core.records_types import AgentRecord
from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.core.tasks import TaskGenerator
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    DEFAULT_BATCHES_PER_ROUND,
    DEFAULT_PLAYS_PER_TASK,
    DEFAULT_SCAFFOLD_TOLERANCE,
    DEFAULT_TASKS_PER_STEP,
    TRAINING_ARM,
    GeneratedTaskBatches,
    GeneratedTaskTasks,
    reported_arm,
    reported_task_name,
)
from reef.record2record import GroupDecision, ReportContext, RoundGrouping, RoundStage
from reef.train.algos import StepScheduling
from reef.train.pipeline import TrainingPipeline


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
    """The training arm's episodes of one task form one group, ready at ``plays_per_task`` episodes.

    Measurement arms and the generator's own reports share the scenario; only the training arm trains.
    A source session is a report whose inference records all carry ``source_tag``.
    """

    def __init__(self, plays_per_task: int, source_tag: tuple[str, str] | None = None) -> None:
        self.plays_per_task = plays_per_task
        self.source_tag = source_tag

    def is_training_report(self, report: AgentRecord) -> bool:
        return reported_arm(report) == TRAINING_ARM

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        return reported_task_name(context.report), None

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY if len(members) >= self.plays_per_task else GroupDecision.INCOMPLETE

    def is_source_report(self, context: ReportContext) -> bool:
        if self.source_tag is None or not context.inferences:
            return False
        name, value = self.source_tag
        return all(inference_tags(inference).get(name) == value for inference in context.inferences)


def task_group_record_stage(context: PipelineContext) -> RoundStage:
    """The generated-task record stage: one group per task, and a round request every ``batches_per_round`` batches.

    Each request takes at most ``source_records_per_round`` source sessions. A terminal report owns its sources.
    """
    config = context.config
    return RoundStage(
        context.scenario,
        context.report_type,
        TaskGroupGrouping(
            int(config.get("plays_per_task", DEFAULT_PLAYS_PER_TASK)), source_tag_of(str(config.get("source_tag", "")))
        ),
        batches_per_round=int(config.get("batches_per_round", DEFAULT_BATCHES_PER_ROUND)),
        sources_per_round=int(config.get("source_records_per_round", 0)),
        exclusive_sources=True,
    )


@dataclass(frozen=True)
class TaskGenerationConfig:
    """The ``task-generation`` section: the checked generator class, its config, and the deployment's environment."""

    plugin: PluginReference[TaskGenerator]
    environ: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_setting(cls, value: object, *, environ: Mapping[str, str]) -> TaskGenerationConfig:
        return cls(TaskGenerator.reference(value, setting="task-generation"), dict(environ))

    def build(self, scenario: str) -> TaskGenerator:
        """Build the generator of one scenario."""
        return self.plugin.plugin_class.from_config(self.plugin.config, scenario=scenario, environ=self.environ)


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Rounds of generated tasks played by the served model; group relative advantages per task.

    The deployment's ``task-generation`` section selects the generator. With ``rounds`` above zero the
    pipeline runs that many rounds: ``generator_url`` names the generator service that ``reef serve``
    starts from the deployment's ``generator`` section, ``description`` says what the tasks are about,
    and ``state_dir`` keeps a summary of each round, in one directory for each scenario.

    With ``source_records_per_round`` above zero, each round also gives the generator at most that many
    recorded sessions: reports whose inference records carry ``source_tag`` (``name=value``, the
    ``x-reef-tag-<name>`` header).
    """

    name: str = "generated-tasks"
    tasks_per_step: int = config_field(DEFAULT_TASKS_PER_STEP)
    plays_per_task: int = config_field(DEFAULT_PLAYS_PER_TASK)
    # A thinking template's generation prompt ends with a think scaffold the history drops; the assembly may
    # realign that many masked tokens ahead of the previous response.
    scaffold_tolerance: int = config_field(DEFAULT_SCAFFOLD_TOLERANCE)
    generator_url: str = config_field("")
    rounds: int = config_field(0)
    batches_per_round: int = config_field(DEFAULT_BATCHES_PER_ROUND)
    description: str = config_field("")
    source_tag: str = config_field("")
    source_records_per_round: int = config_field(0)
    state_dir: str = config_field("")
    served_model: str = config_field("")
    task_generation: TaskGenerationConfig | None = field(default=None, repr=False, compare=False)

    @property
    def report_type(self) -> type[ReportBase]:
        # The task player's report: the verifier reward as the score; the task under metadata.task.
        return ScoredRolloutReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=TaskGroupObjective.name,
            task2batch=GeneratedTaskBatches,
            scheduling=StepScheduling(unit="sample", batch_size="actual"),
        )

    @classmethod
    def _recipe_kwargs(cls, settings: Mapping[str, Any], values: Mapping[str, str]) -> dict[str, Any]:
        kwargs = super()._recipe_kwargs(
            {key: value for key, value in settings.items() if key != "task_generation"}, values
        )
        setting = settings.get("task_generation")
        if setting is None:
            return kwargs
        try:
            return {**kwargs, "task_generation": TaskGenerationConfig.from_setting(setting, environ=values)}
        except PluginConfigError as exc:
            raise RecipeConfigError(str(exc)) from exc

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.plays_per_task < 2:
            raise ValueError("plays_per_task must be at least two")
        if self.scaffold_tolerance < 0:
            raise ValueError("scaffold_tolerance must be non-negative")
        if self.rounds < 0 or self.batches_per_round < 1:
            raise ValueError("rounds must be non-negative and batches_per_round positive")
        if self.rounds > 0:
            if self.task_generation is None:
                raise ValueError("rounds need a task generator: add a task-generation section")
            if not self.generator_url or "${" in self.generator_url:
                raise ValueError(
                    "rounds need generator_url: ${endpoints.generator} when the deployment has a generator section"
                )
            if not self.description.strip() or not self.state_dir:
                raise ValueError("rounds need a description and a state_dir")
        if self.source_records_per_round < 0:
            raise ValueError("source_records_per_round must be non-negative")
        if self.source_records_per_round > 0 and not self.source_tag:
            raise ValueError("source_records_per_round needs a source_tag that marks the source sessions")
        source_tag_of(self.source_tag)

    def scenario_state_dirs(self, scenario: str) -> tuple[Path, ...]:
        """The scenario's round summaries: what a delete archives beside the record store."""
        return (Path(self.state_dir).expanduser() / scenario,) if self.state_dir else ()

    def build_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        # The record-to-task stage tells the record stage when each round finished, so both share one stage.
        configured = context.with_config(self.pipeline_config())
        round_stage = task_group_record_stage(configured)
        task_generator = None if self.task_generation is None else self.task_generation.build(configured.scenario)
        return TrainingPipeline(
            configured,
            record_stage=round_stage,
            record2task=GeneratedTaskTasks(configured, round_stage, task_generator=task_generator),
            task2batch=GeneratedTaskBatches(configured),
        )
