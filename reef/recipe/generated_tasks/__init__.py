"""The built-in generated-task recipe: train the served model on the tasks that a task generator makes.

The deployment selects the generator in its ``task-generation`` section (see
:class:`reef.core.tasks.TaskGenerator`). The recipe does not depend on the
generator's method:

- ``processor``: ``GeneratedTaskTasks``, the record-to-task stage that runs the
  rounds (generate, validate, play through the generator service, send the
  results back) and turns the training arm's episodes into tasks; and
  ``GeneratedTaskProcessor``, which batches complete task groups.
- ``objective``: group relative advantages per task group on Tinker's
  importance sampling loss.
- ``recipe``: the configuration that binds them.

The recipe plays tasks through the generator service (``reef.record2task.generator``).
"""

from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import GeneratedTaskProcessor, GeneratedTaskTasks
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe, TaskGenerationConfig

__all__ = [
    "GeneratedTaskProcessor",
    "GeneratedTaskRecipe",
    "GeneratedTaskTasks",
    "TaskGenerationConfig",
    "TaskGroupObjective",
]
