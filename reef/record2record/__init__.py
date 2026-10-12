"""Record to record: the shared stage that decides which records Reef uses, and in what order.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.record2dataset``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2RecordStage`` in ``stage.py``.

* Each trainer has one stage. ``Trainer.build(record_stage=...)`` sets it, and
  ``TrainingFeed`` calls it.
* The stage reads records and returns ``RecordSet`` values: records that a
  processor uses together. ``SingleRecordSet`` holds one record.
* ``PassThroughStage`` is the default. It returns one set per record, in
  arrival order.

Components. Until RFC #765 is complete, the processor engines in
``reef.train`` use them directly:

* ``BatchUnit`` (``units.py``): one sample or one ready group. A processor
  subclasses it to add its own data, such as the sample or the reports of a
  group. ``RecordSet`` is also a ``BatchUnit``.
* ``UnitBuffer`` (``buffer.py``): the units that are held, and the reserved
  batch.
* ``SelectionPolicy`` (``selection.py``): the order in which units go into a
  batch. ``ArrivalOrder`` takes the oldest first. ``GroupKeyOrder`` takes
  ungrouped units, then groups by key.
* ``EligibilityCheck`` (``eligibility.py``): finds held units that can no
  longer train. ``StalenessCheck`` drops units that are more than
  ``max_staleness`` versions behind the newest held unit.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``, and
``reef.observability``. Thus record-to-task and the trainer can both use it.
"""

from reef.record2record.buffer import UnitBuffer
from reef.record2record.eligibility import EligibilityCheck, IneligibleUnit, StalenessCheck
from reef.record2record.reports import (
    GroupDecision,
    ReportCandidate,
    ReportContext,
    ReportEntry,
    ReportGrouping,
    ReportSet,
    ReportStage,
    accepted_by,
    report_score,
    report_sets,
)
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder, SelectionPolicy
from reef.record2record.stage import (
    AssemblyStatus,
    PassThroughStage,
    Record2RecordStage,
    RecordSet,
    SetAssembler,
    SetAssembly,
    SingleRecordSet,
    record_sets,
)
from reef.record2record.units import BatchUnit

__all__ = [
    "ArrivalOrder",
    "AssemblyStatus",
    "BatchUnit",
    "EligibilityCheck",
    "GroupDecision",
    "GroupKeyOrder",
    "IneligibleUnit",
    "PassThroughStage",
    "Record2RecordStage",
    "RecordSet",
    "ReportCandidate",
    "ReportContext",
    "ReportEntry",
    "ReportGrouping",
    "ReportSet",
    "ReportStage",
    "SelectionPolicy",
    "SetAssembler",
    "SetAssembly",
    "SingleRecordSet",
    "StalenessCheck",
    "UnitBuffer",
    "accepted_by",
    "record_sets",
    "report_score",
    "report_sets",
]
