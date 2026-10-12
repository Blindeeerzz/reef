"""Record to record: the shared stage that decides which recorded experience is used, and in what order.

Reef reuses recorded experience in two ways. It trains on records directly,
and it turns records into runnable tasks (``reef.record2dataset``). Both need
the same decisions about each record. This package owns those decisions:

* ``ExperienceUnit``: one sample or one ready group. A processor subclasses it
  to add its own data, such as the sample or the reports of a group.
* ``ExperienceBuffer``: the units that are held, and the reserved batch.
* ``EligibilityCheck``: finds held units that can no longer train.
  ``StalenessCheck`` drops units that are more than ``max_staleness``
  versions behind the newest held unit.
* ``SelectionPolicy``: the order in which units go into a batch.
  ``ArrivalOrder`` takes the oldest first; ``GroupKeyOrder`` takes ungrouped
  units, then groups by key.
* ``JudgingWorker``: runs a ``JobRunner``'s asynchronous jobs off the trainer
  thread; a job that never finishes resolves as ``Failed``.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``, and
``reef.observability``, so record-to-task and the trainer can both use it.
RFC #765 describes the stage; the processor engines still own grouping and
judging until it is complete.
"""

from reef.record2record.buffer import ExperienceBuffer
from reef.record2record.eligibility import EligibilityCheck, IneligibleUnit, StalenessCheck
from reef.record2record.selection import ArrivalOrder, ExperienceUnit, GroupKeyOrder, SelectionPolicy
from reef.record2record.worker import Failed, JobRunner, JudgingWorker, SupportsReceipt

__all__ = [
    "ArrivalOrder",
    "EligibilityCheck",
    "ExperienceBuffer",
    "ExperienceUnit",
    "Failed",
    "GroupKeyOrder",
    "IneligibleUnit",
    "JobRunner",
    "JudgingWorker",
    "SelectionPolicy",
    "StalenessCheck",
    "SupportsReceipt",
]
