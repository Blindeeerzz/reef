"""The generated-task recipe: train on the tasks that the deployment's task generator makes."""

from __future__ import annotations

from dataclasses import dataclass

from reef.core.reports import ReportBase, ScoredRolloutReport
from reef.pipeline_context import PipelineContext
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import GeneratedTaskBatches, GeneratedTaskRecord2Record, GeneratedTaskTasks
from reef.record2record import Record2Record
from reef.train.algos import StepScheduling


@dataclass(frozen=True, kw_only=True)
class GeneratedTaskRecipe(WeightTrainingRecipe):
    """Group relative advantages per generated task, on Tinker's importance sampling loss.

    The recipe binds the objective, the step schedule, and the report
    contract. Its stages are ``GeneratedTaskRecord2Record``,
    ``GeneratedTaskTasks``, and ``GeneratedTaskBatches``; the deployment's
    ``pipeline`` section names them with their config: the rounds, the task
    generator, and the generator service in ``record2task``, the task groups
    and the source sessions in ``record2record``, and the batch size in
    ``task2batch``. Without that section, no round runs and the recipe trains
    on the training arm's reports only.
    """

    name: str = "generated-tasks"

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

    def build_record2record(self, context: PipelineContext) -> GeneratedTaskRecord2Record:
        return GeneratedTaskRecord2Record(context)

    def build_record2task(self, context: PipelineContext, record2record: Record2Record) -> GeneratedTaskTasks:
        return GeneratedTaskTasks.from_config({}, context=context, record2record=record2record, environ={})
