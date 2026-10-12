"""The built-in task-training recipe: train the served model on the tasks that a task generator makes.

The deployment selects the generator in its ``task-generation`` section (see
:class:`reef.core.tasks.TaskGenerator`). The recipe does not depend on the
generator's method:

- ``processor``: the rounds (generate, validate, play through the generator
  service, send the results back) and the training arm's episodes grouped by task.
- ``objective``: group relative advantages per task group on Tinker's
  importance sampling loss.
- ``recipe``: the configuration that binds them.

The recipe lives beside the generator service because it plays tasks through it.
"""

from reef.record2dataset.task_training.objective import TaskGroupObjective
from reef.record2dataset.task_training.processor import TaskTrainingProcessor
from reef.record2dataset.task_training.recipe import TaskGenerationConfig, TaskTrainingRecipe

__all__ = [
    "TaskGenerationConfig",
    "TaskGroupObjective",
    "TaskTrainingProcessor",
    "TaskTrainingRecipe",
]
