"""Task to batch: the stage that shapes the tasks of the selected record sets into one batch.

Reef reuses records in a pipeline of three stages::

    record ─► record2record ─► record set ─► record2task ─► task ─► task2batch ─► batch

The training pipeline (``reef.train.pipeline.TrainingPipeline``) reserves the
next batch's record sets from ``reef.record2record`` and gives their tasks,
which ``reef.record2task`` made, to this stage.

Entry point: ``Task2Batch`` in ``stage.py``. A recipe writes ``make_batch`` and
names the class in ``WeightTrainingSpec.task2batch``.

* ``NoBatches`` (``stage.py``): the no-update default; it never makes a batch.
* ``DistillBatches`` (``distill.py``): the distilling recipes' batches, named by
  ``batch_label``.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here. The package depends only
on ``reef.core``, ``reef.observability``, and ``reef.pipeline_context``.
"""

from reef.task2batch.distill import DistillBatches
from reef.task2batch.stage import NoBatches, Task2Batch

__all__ = [
    "DistillBatches",
    "NoBatches",
    "Task2Batch",
]
