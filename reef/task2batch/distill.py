"""Batches of distillation tasks; ``reef.record2task.distill`` makes the tasks."""

from __future__ import annotations

from reef.core.batches import TrainDataItem, TrainingBatch
from reef.task2batch.base import Task2Batch


class DistillBatches(Task2Batch):
    """Batch distillation tasks as they come. A recipe's subclass sets ``batch_label``, its batches' name."""

    batch_label = "teacher"

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:{self.batch_label}:{batch_number}", tasks)
