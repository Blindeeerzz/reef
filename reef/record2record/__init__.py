"""Record to record: the shared stage that decides which records Reef uses, and in what order.

Reef uses records in two ways. It trains on records directly, and it turns
records into runnable tasks (``reef.recipe.generated_tasks``). Both need the same
decisions about each record. This package owns those decisions.

Entry point: ``Record2Record`` in ``base.py``.

* Each training pipeline (``reef.train.pipeline.TrainingPipeline``) has one
  stage, from the recipe's ``build_record_stage(context)``. The pipeline gives
  the stage every record except training instructions.
* The stage forms ``RecordSet`` values: records that are used together. The
  record-to-task stage makes tasks from each set, and the pipeline gives the
  result back (``settle``). The stage holds the sets that train in its pool,
  drops the sets that can no longer train, and gives the pipeline the next
  batch's sets in order. The stage never calls the other stages.

Stages:

* ``PassThroughRecord2Record`` (``base.py``): the default. It forms one set per
  record and decides nothing.
* ``ReportStage`` (``reports.py``): reports and the inference records that
  they reference, alone or in groups. A recipe's ``ReportGrouping`` decides
  which reports train and how they form groups.
* ``JudgingStage`` (``judging.py``): records that a model judges on a
  background worker after later traffic completes them. A recipe's
  ``RecordJudging`` holds the correlation and the judge.
* ``InferenceRecord2Record`` (``base.py``): one set per inference record.
* ``RoundStage`` (``rounds.py``): a ``ReportStage`` that also forms a
  ``RoundRequest`` set each time a task-generation round is due. A recipe's
  ``RoundGrouping`` decides which reports are source sessions for the rounds.

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

from reef.record2record.base import (
    AssemblyStatus,
    InferenceRecord2Record,
    PassThroughRecord2Record,
    Record2Record,
    SetAssembly,
)
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
from reef.record2record.rounds import RoundGrouping, RoundRequest, RoundStage
from reef.record2record.selection import ArrivalOrder, GroupKeyOrder, SelectionPolicy
from reef.record2record.sets import RecordSet, SingleRecordSet

__all__ = [
    "ArrivalOrder",
    "AssemblyStatus",
    "EligibilityCheck",
    "GroupDecision",
    "GroupKeyOrder",
    "IneligibleSet",
    "InferenceRecord2Record",
    "JudgedSet",
    "JudgingStage",
    "PassThroughRecord2Record",
    "Record2Record",
    "RecordJudging",
    "RecordSet",
    "RecordSetPool",
    "ReportCandidate",
    "ReportContext",
    "ReportEntry",
    "ReportGrouping",
    "ReportSet",
    "ReportStage",
    "RoundGrouping",
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
