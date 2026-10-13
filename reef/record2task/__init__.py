"""Record to task: the stage that turns one record set into tasks.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch

``reef.record2record`` decides which records Reef uses and forms record sets.
This package turns each set into tasks. A task is a ``TrainDataItem``: a
``TrajectoryItem`` that trains directly, or a ``TaskItem`` that the algorithm
rolls out. The processor (task-to-batch, ``reef.train.processors``) shapes the
tasks into batches.

Entry point: ``Record2Task`` in ``stage.py``. The recipe builds one in
``build_record2task(context)``.

Implementations:

* ``ReportTasks`` (``reports.py``): one valid report and its inference
  records into one task. A recipe writes ``make_sample``.
* ``JudgedTasks`` (``judged.py``): one judged record into one task. A recipe
  writes ``make_sample``.
* ``GeneratedTaskTasks`` (``generated_tasks``): new Harbor tasks from a
  description and, optionally, from recorded sessions. Each round generates
  tasks, checks them, and plays them; the training plays come back as
  reports and become tasks.

Subpackages, which this module does not import:

* ``generator``: the generator service. It runs the slow steps that need
  containers (the designer prompt, the authoring rules, the oracle check,
  and the task plays) in a process beside the HTTP service.
* ``generated_tasks``: the built-in generated-task recipe.

The modules in this directory depend only on ``reef.core`` and
``reef.record2record``. Thus the trainer can use them without the generator
service.
"""

from reef.record2task.judged import JudgedTasks
from reef.record2task.reports import ReportTasks, SkippedReport
from reef.record2task.stage import NoTasks, Record2Task, TaskResult

__all__ = [
    "JudgedTasks",
    "NoTasks",
    "Record2Task",
    "ReportTasks",
    "SkippedReport",
    "TaskResult",
]
