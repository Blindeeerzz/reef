"""Task to batch: the tasks of the selected record sets become one typed training batch.

A recipe writes a ``Task2Batch`` with ``make_batch``. The training pipeline
(:class:`reef.train.pipeline.TrainingPipeline`) selects the record sets of
each batch and gives their tasks to it. ``DistillBatches`` batches the
distilling recipes' tasks; ``distill`` also holds their record-to-task stage.
``reported`` holds ``SampleAssembly``, which turns a report and its inference
records into a policy sample.

The design (the three stages, what a recipe writes for each, and a record's
path to a batch) is at https://reefinfra.ai/docs/developer-guide/processors/.

Everything numeric about a method, such as advantages and the loss family,
lives in its backend training objective, not here.
"""

from reef.train.processors.base import NoBatches, Task2Batch
from reef.train.processors.distill import DistillBatches

__all__ = [
    "DistillBatches",
    "NoBatches",
    "Task2Batch",
]
