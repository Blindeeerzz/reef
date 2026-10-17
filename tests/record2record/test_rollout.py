"""The rollout loop: saved tasks become a rollout batch, its episodes group by task use, and only that batch trains."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory
from tests.train.test_record_stage_batch_parity import inference, report

from reef.core import AgentRecord
from reef.core.reports import ScoredRolloutReport
from reef.core.reports.base import ReportBase
from reef.recipe.composite import any_component_report
from reef.record2record import PassThroughRecord2Record, ReportContext
from reef.record2record.rollout import RolloutGrouping, RolloutReportRecord2Record
from reef.record2task import ReportTasks
from reef.record2task.reports import SkippedReport
from reef.task2batch import RolloutResult, RolloutTask2Batch
from reef.train import PipelineContext
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch


class EpisodeTasks(ReportTasks):
    """Each training report becomes one sample, produced by the version its inference record names."""

    def make_sample(self, context: ReportContext) -> TrainDataItem:
        training = context.inferences[0].payload["response"]["training"]
        return policy_trajectory(
            context.report.agent_record_id,
            [1, 2, 3],
            [0, 1, 1],
            [-0.1, -0.2],
            context.require_score(),
            runtime_load_id=training.get("runtime_load_id"),
        )


class FakeRollout(RolloutTask2Batch):
    """Records each submitted rollout batch; ``finish`` hands back each use's training reports on the next look."""

    def __init__(self, context: PipelineContext, batch_size: int) -> None:
        super().__init__(context, batch_size)
        self.submitted: list[tuple[str, tuple[str, ...]]] = []
        self.finished: list[RolloutResult] = []
        self.trained: list[tuple[str, tuple[str, ...]]] = []

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"rollout:{batch_number}", tasks)

    def submit(self, batch: str, tasks: tuple[TaskItem, ...]) -> tuple[str, ...]:
        task_uses = tuple(f"{batch}/{task.task_path.name}" for task in tasks)
        self.submitted.append((batch, task_uses))
        return task_uses

    def finish(self, training_reports: Mapping[str, tuple[str, ...]]) -> None:
        batch, _ = self.submitted[-1]
        self.finished.append(RolloutResult(batch, training_reports))

    def take_finished(self) -> tuple[RolloutResult, ...]:
        finished, self.finished = tuple(self.finished), []
        return finished

    def running(self) -> bool:
        return False

    def rollout_trained(self, batch: str, task_uses: tuple[str, ...]) -> None:
        self.trained.append((batch, task_uses))


def rollout_pipeline(
    tasks: int = 2, batch_size: int = 2
) -> tuple[TrainingPipeline, RolloutReportRecord2Record, FakeRollout]:
    context = PipelineContext("s")
    stage = RolloutReportRecord2Record(context, RolloutGrouping(), exclusive_sources=True)
    stage.add_tasks(tuple(TaskItem(Path(f"task-{index}")) for index in range(tasks)), group_key=0)
    rollout = FakeRollout(context, batch_size)
    return (
        TrainingPipeline(context, record2record=stage, record2task=EpisodeTasks(), task2batch=rollout),
        stage,
        rollout,
    )


def episode(
    pipeline: TrainingPipeline,
    batch: str,
    task_use: str,
    episode_id: str,
    *,
    arm: str = "plain",
    version: str | None = None,
    report_id: str | None = None,
) -> str:
    """Ingest one rollout episode's inference record and report; return the report ID."""
    report_id = report_id or f"rep-{episode_id}"
    pipeline.ingest(inference(f"rec-{report_id}", version=version))
    labels = {"arm": arm, "batch": batch, "task_use": task_use}
    record: AgentRecord = report(
        report_id, f"rec-{report_id}", score=1.0, episode={"id": episode_id, "labels": labels}
    )
    pipeline.ingest(record)
    return report_id


@pytest.mark.unit
def test_saved_tasks_roll_out_and_only_their_settled_groups_form_the_batch() -> None:
    pipeline, stage, rollout = rollout_pipeline()
    assert not pipeline.ready() and len(stage.task_pool) == 0
    ((batch, (first, second)),) = rollout.submitted
    assert pipeline.background_jobs_running() and pipeline.status()["rollout"]["batch"] == batch
    reports = {first: tuple(episode(pipeline, batch, first, f"{first}-{name}") for name in "ab")}
    assert not pipeline.ready(), "the plays have not ended, so no group knows its reports"
    reports[second] = tuple(episode(pipeline, batch, second, f"{second}-{name}") for name in "ab")
    rollout.finish(reports)
    assert pipeline.ready() and pipeline.pending is None, "a ready batch waits for build_batch"
    built = pipeline.build_batch()
    assert len(built.items) == 4
    pipeline.acknowledge(built.batch_id)
    assert rollout.trained == [(batch, (first, second))]
    late = episode(pipeline, batch, first, f"{first}-late")
    assert late in pipeline.releasable_record_ids(), "a report of a batch that trained never trains"
    assert pipeline.status()["rollout"] is None


@pytest.mark.unit
def test_a_group_counts_episodes_not_reports_and_a_group_of_one_is_discarded() -> None:
    pipeline, _, rollout = rollout_pipeline()
    pipeline.ready()
    ((batch, (first, second)),) = rollout.submitted
    episode(pipeline, batch, first, "episode-1", report_id="segment-1")
    episode(pipeline, batch, first, "episode-1", report_id="segment-2")
    episode(pipeline, batch, second, "episode-2")
    episode(pipeline, batch, second, "episode-3")
    episode(pipeline, batch, second, "episode-4", arm="hint")
    rollout.finish({first: ("segment-1", "segment-2"), second: ("rep-episode-2", "rep-episode-3")})
    assert pipeline.ready()
    built = pipeline.build_batch()
    assert {item.source_agent_record_ids[-1] for item in built.items} == {"rep-episode-2", "rep-episode-3"}
    assert {"segment-1", "segment-2", "rep-episode-4"} <= pipeline.releasable_record_ids()


@pytest.mark.unit
def test_a_batch_whose_groups_all_went_stale_is_given_up_and_the_next_rollout_starts() -> None:
    pipeline, stage, rollout = rollout_pipeline(tasks=3, batch_size=2)
    pipeline.observe_serving_runtime_load_id("v1")
    pipeline.ready()
    ((batch, (first, second)),) = rollout.submitted
    reports = {
        task_use: tuple(episode(pipeline, batch, task_use, f"{task_use}-{name}", version="v1") for name in "ab")
        for task_use in (first, second)
    }
    pipeline.observe_serving_runtime_load_id("v2")
    rollout.finish(reports)
    assert not pipeline.ready(), "every group lags a version behind, so none trains"
    assert len(rollout.submitted) == 2 and len(stage.task_pool) == 0, "the third task starts the next rollout"


@pytest.mark.unit
def test_a_rollout_stage_needs_a_rollout_record_stage() -> None:
    context = PipelineContext("s")
    with pytest.raises(TypeError, match="needs a RolloutReportRecord2Record"):
        TrainingPipeline(context, record2record=PassThroughRecord2Record(), task2batch=FakeRollout(context, 1))


@pytest.mark.unit
def test_a_rollout_that_reported_no_episode_settles_and_the_next_rollout_starts() -> None:
    pipeline, stage, rollout = rollout_pipeline(tasks=4, batch_size=2)
    pipeline.ready()
    rollout.finish({})
    assert not pipeline.ready(), "no group waits for a report"
    assert len(rollout.submitted) == 2 and len(stage.task_pool) == 0


class SkippingTasks(EpisodeTasks):
    """Skip the report named ``rep-skip``; train on the others."""

    def make_sample(self, context: ReportContext) -> TrainDataItem | SkippedReport:
        if context.report.agent_record_id == "rep-skip":
            return SkippedReport("no sample")
        return super().make_sample(context)


@pytest.mark.unit
def test_a_skipped_episode_counts_toward_its_group() -> None:
    context = PipelineContext("s")
    stage = RolloutReportRecord2Record(context, RolloutGrouping(), exclusive_sources=True)
    stage.add_tasks((TaskItem(Path("task-0")),), group_key=0)
    rollout = FakeRollout(context, 1)
    pipeline = TrainingPipeline(context, record2record=stage, record2task=SkippingTasks(), task2batch=rollout)
    pipeline.ready()
    ((batch, (task_use,)),) = rollout.submitted
    for name in ("a", "b"):
        episode(pipeline, batch, task_use, name)
    episode(pipeline, batch, task_use, "c", report_id="rep-skip")
    rollout.finish({task_use: ("rep-a", "rep-b", "rep-skip")})
    assert pipeline.ready()
    assert len(pipeline.build_batch().items) == 2


@pytest.mark.unit
def test_a_report_that_arrives_while_its_batch_waits_is_released_when_the_batch_ends() -> None:
    pipeline, stage, rollout = rollout_pipeline(tasks=1, batch_size=1)
    pipeline.ready()
    ((batch, (task_use,)),) = rollout.submitted
    episode(pipeline, batch, task_use, "a")
    episode(pipeline, batch, task_use, "b")
    rollout.finish({task_use: ("rep-a", "rep-b")})
    built = pipeline.build_batch()
    late = episode(pipeline, batch, task_use, "c")
    pipeline.acknowledge(built.batch_id)
    assert late in pipeline.releasable_record_ids() and (batch, task_use) not in stage.groups


@pytest.mark.unit
def test_a_group_waits_for_the_reports_its_plays_returned_until_the_settle_deadline() -> None:
    pipeline, _, rollout = rollout_pipeline(tasks=1, batch_size=1)
    pipeline.ready()
    ((batch, (task_use,)),) = rollout.submitted
    episode(pipeline, batch, task_use, "a")
    episode(pipeline, batch, task_use, "b")
    rollout.finish({task_use: ("rep-a", "rep-b", "rep-lost")})
    assert not pipeline.ready(), "the third report has not arrived"
    rollout.settle_timeout_seconds = 0.0
    assert pipeline.ready(), "past the deadline the group settles with the two episodes it holds"
    assert len(pipeline.build_batch().items) == 2


@pytest.mark.unit
def test_a_failed_submit_keeps_the_tasks_in_the_pool() -> None:
    class FailingRollout(FakeRollout):
        def submit(self, batch: str, tasks: tuple[TaskItem, ...]) -> tuple[str, ...]:
            raise RuntimeError("worker down")

    context = PipelineContext("s")
    stage = RolloutReportRecord2Record(context, RolloutGrouping(), exclusive_sources=True)
    stage.add_tasks((TaskItem(Path("task-0")),), group_key=0)
    pipeline = TrainingPipeline(
        context, record2record=stage, record2task=EpisodeTasks(), task2batch=FailingRollout(context, 1)
    )
    with pytest.raises(RuntimeError, match="worker down"):
        pipeline.ready()
    assert len(stage.task_pool) == 1 and not stage.task_pool.reserved_sets()


@dataclass(frozen=True)
class AnyReport(ReportBase):
    """Another component's contract, which accepts any valid payload."""


@pytest.mark.unit
def test_a_training_report_that_the_contract_refuses_counts_toward_its_group() -> None:
    context = PipelineContext(
        "s",
        report_type=ScoredRolloutReport,
        admitted_report_type=any_component_report((ScoredRolloutReport, AnyReport)),
    )
    stage = RolloutReportRecord2Record(context, RolloutGrouping(), exclusive_sources=True)
    stage.add_tasks((TaskItem(Path("task-0")),), group_key=0)
    rollout = FakeRollout(context, 1)
    pipeline = TrainingPipeline(context, record2record=stage, record2task=EpisodeTasks(), task2batch=rollout)
    pipeline.ready()
    ((batch, (task_use,)),) = rollout.submitted
    episode(pipeline, batch, task_use, "a")
    episode(pipeline, batch, task_use, "b")
    pipeline.ingest(inference("rec-c"))
    labels = {"arm": "plain", "batch": batch, "task_use": task_use}
    unscored = report("rep-c", "rec-c", score=1.0, episode={"id": "c", "labels": labels})
    unscored.payload["score"] = None
    pipeline.ingest(unscored)
    rollout.finish({task_use: ("rep-a", "rep-b", "rep-c")})
    assert pipeline.ready() and "rep-c" in pipeline.releasable_record_ids()
    assert len(pipeline.build_batch().items) == 2


@pytest.mark.unit
def test_reports_without_an_episode_id_never_train_but_still_settle_their_group() -> None:
    pipeline, _, rollout = rollout_pipeline(tasks=1, batch_size=1)
    pipeline.ready()
    ((batch, (task_use,)),) = rollout.submitted
    labels = {"arm": "plain", "batch": batch, "task_use": task_use}
    for report_id in ("segment-1", "segment-2"):
        pipeline.ingest(inference(f"rec-{report_id}"))
        pipeline.ingest(report(report_id, f"rec-{report_id}", score=1.0, episode={"labels": labels}))
    rollout.finish({task_use: ("segment-1", "segment-2")})
    assert not pipeline.ready() and pipeline.rollout is None, "the batch settled with nothing to train"
    assert {"segment-1", "segment-2"} <= pipeline.releasable_record_ids()
