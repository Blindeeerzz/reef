"""Record to record: the shared stage that decides which records Reef uses, and in what order.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.recipe.generated_tasks``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2RecordStage`` in ``stage.py``.

* Each processor has one stage, from ``DataProcessor.build_record_stage()``.
  The processor gives the stage every record except training instructions.
* The stage forms ``RecordSet`` values: records that a processor uses
  together. The processor assembles each set and gives the result back
  (``settle``). The stage holds the assembled sets in its pool, drops the sets
  that can no longer train, and gives the processor the next batch's sets in
  order. The stage never calls the processor.

Stages:

* ``PassThroughStage`` (``stage.py``): the default. It forms one set per
  record, and the processor keeps the records itself.
* ``ReportStage`` (``reports.py``): reports and the inference records that
  they reference, alone or in groups. A recipe's ``ReportGrouping`` decides
  which reports train and how they form groups.
* ``JudgingStage`` (``judging.py``): records that a model judges on a
  background worker after later traffic completes them. A recipe's
  ``RecordJudging`` holds the correlation and the judge.
* ``InferenceStage`` (``stage.py``): one set per inference record.
* ``RoundStage`` (``rounds.py``): a ``ReportStage`` that also forms an empty
  ``RoundRequest`` set each time a task-generation round is due.

Components of the pool:

* ``RecordSet`` (``sets.py``): the base of every set, with the fields that the
  pool, the selection policies, and the eligibility checks read.
* ``RecordSetPool`` (``pool.py``): the held sets, and the reserved batch.
* ``SelectionPolicy`` (``selection.py``): the order in which sets go into a
  batch. ``ArrivalOrder`` takes the oldest first. ``GroupKeyOrder`` takes
  ungrouped sets, then groups by key.
* ``EligibilityCheck`` (``eligibility.py``): finds held sets that can no
  longer train. ``StalenessCheck`` drops sets that are more than
  ``max_staleness`` versions behind the newest held set.

The package depends only on ``reef.core``, ``reef.runtime.interfaces``,
``reef.observability``, and ``reef.background_jobs``. Thus record-to-task and
the trainer can both use it.
"""

from reef.record2record.eligibility import EligibilityCheck, IneligibleSet, StalenessCheck
from reef.record2record.judging import JudgedSet, JudgingStage, RecordJudging, judged_sets
from reef.record2record.pool import RecordSetPool
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
from reef.record2record.rounds import RoundRequest, RoundStage
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder, SelectionPolicy
from reef.record2record.sets import RecordSet, SingleRecordSet
from reef.record2record.stage import AssemblyStatus, InferenceStage, PassThroughStage, Record2RecordStage, SetAssembly

__all__ = [
    "ArrivalOrder",
    "AssemblyStatus",
    "EligibilityCheck",
    "GroupDecision",
    "GroupKeyOrder",
    "IneligibleSet",
    "InferenceStage",
    "JudgedSet",
    "JudgingStage",
    "PassThroughStage",
    "Record2RecordStage",
    "RecordJudging",
    "RecordSet",
    "RecordSetPool",
    "ReportCandidate",
    "ReportContext",
    "ReportEntry",
    "ReportGrouping",
    "ReportSet",
    "ReportStage",
    "RoundRequest",
    "RoundStage",
    "SelectionPolicy",
    "SetAssembly",
    "SingleRecordSet",
    "StalenessCheck",
    "accepted_by",
    "judged_sets",
    "report_score",
    "report_sets",
]
