"""Imported examples and subsequent chat traffic use the same ingestion path."""

from __future__ import annotations

import asyncio
import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.train.test_reef_trainer_contracts import ExampleBackend

from reef.artifact import InMemoryRepositoryBackend
from reef.core.records_types import RequestType
from reef.core.reports import ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.record2record import ReportGrouping, ReportRecord2Record
from reef.record2task import ReportTasks
from reef.runtime.interfaces import InferenceHandler
from reef.service.app import create_app
from reef.service.routes.records import MAX_RECORD_IMPORT_BYTES
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.task2batch import Task2Batch
from reef.train import PreparedStep, Trainer, TrainingBatch
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TaskItem, TrajectoryItem

HEADERS = {"x-reef-scenario": "s", "Authorization": "Bearer import-test-token"}


def inference_payload(text):
    return {
        "model": "example-model",
        "messages": [{"role": "user", "content": text}],
        "response": {"choices": [{"message": {"role": "assistant", "content": f"answer:{text}"}}]},
        "metadata": {"source": "fixture"},
    }


def imported(record_id, payload=None, request_type="inference"):
    return {
        "agent_record_id": record_id,
        "request_type": request_type,
        "payload": inference_payload(record_id) if payload is None else payload,
    }


class FeedbackTasks(ReportTasks):
    def make_sample(self, context):
        return TaskItem(Path(context.report.agent_record_id), metadata=context.inferences[0].payload)


class FeedbackBatches(Task2Batch):
    def make_batch(self, tasks, batch_id):
        return TrainingBatch(batch_id, tasks)


def feedback_pipeline(context, batch_size):
    return TrainingPipeline(
        context,
        record2record=ReportRecord2Record(context, ReportGrouping()),
        record2task=FeedbackTasks(),
        task2batch=FeedbackBatches(context, batch_size),
    )


class LearningBackend(ExampleBackend):
    def __init__(self, scenario, batches):
        super().__init__(scenario, [])
        self.batches = batches

    def prepare_step(self, batch, state, scenario_step):
        self.batches.append(batch)
        return PreparedStep.skipped(state={"trained": state.get("trained", 0) + len(batch.items)})


@dataclass(frozen=True, kw_only=True)
class LearningRecipe(Recipe):
    batches: list[TrainingBatch] = field(default_factory=list)

    @property
    def report_type(self):
        return ScoredRolloutReport

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: feedback_pipeline(context, batch_size=2),
            candidate_backend=LearningBackend(scenario, self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            report_type=self.report_type,
            training_mode=self.training_mode,
        )


class EchoHandler(InferenceHandler):
    def __init__(self):
        self.calls = 0

    async def inference(self, artifact, path, payload):
        self.calls += 1
        return inference_payload(payload["messages"][0]["content"])["response"]


def dispatcher_for(tmp_path, recipe):
    initial = tmp_path / "initial"
    initial.mkdir()
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


async def wait_for_training(scenario, count):
    async with asyncio.timeout(5):
        while scenario.trainer.state.get("trained", 0) < count:
            await asyncio.sleep(0.01)
    assert scenario.trainer.state["trained"] == count


@pytest.mark.parametrize("endpoint", ("/v1/chat/completions", "/v1/messages"))
def test_large_inference_context_reaches_the_handler(tmp_path, endpoint):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    handler = EchoHandler()
    text = "x" * (3 * 1024 * 1024)

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            response = await client.post(
                endpoint, headers=HEADERS, json={"messages": [{"role": "user", "content": text}]}
            )
            assert response.status == 200, await response.text()
            assert (await response.json())["choices"][0]["message"]["content"] == f"answer:{text}"
            assert handler.calls == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("bulk", (False, True))
def test_record_import_takes_a_long_trajectory_and_keeps_its_body_limit(tmp_path, bulk):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    long_record = imported("long", {"text": "x" * (3 * 1024 * 1024)})
    too_large = imported("too-large", {"text": "x" * MAX_RECORD_IMPORT_BYTES})

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            endpoint = "/reef/records/batch" if bulk else "/reef/records"
            for record, status in ((long_record, 200), (too_large, 413)):
                body = {"records": [record]} if bulk else record
                response = await client.post(endpoint, headers=HEADERS, json=body)
                assert response.status == status, await response.text()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


@pytest.mark.parametrize("bulk", (False, True))
def test_import_then_chat_continue_with_the_same_reported_processor(tmp_path, bulk):
    recipe = LearningRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)
    handler = EchoHandler()

    async def run():
        async with TestClient(
            TestServer(create_app(dispatcher, tokens="import-test-token", inference_handler=handler))
        ) as client:
            # Interleave records and their feedback just as an online producer
            # does, so the existing processor can consume each ready batch.
            records = []
            for index in range(6):
                name = f"offline-{index}"
                records.extend(
                    [
                        imported(name),
                        imported(f"score-{name}", {"score": 1, "references": [name]}, "report"),
                    ]
                )
            if bulk:
                response = await client.post("/reef/records/batch", headers=HEADERS, json={"records": records})
                assert response.status == 200, await response.text()
                receipts = (await response.json())["records"]
                assert [item["agent_record_id"] for item in receipts] == [item["agent_record_id"] for item in records]
            else:
                for record in records:
                    response = await client.post("/reef/records", headers=HEADERS, json=record)
                    assert response.status == 200, await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            processor = scenario.trainer.pipeline
            await wait_for_training(scenario, 6)
            if bulk:
                retry = await client.post("/reef/records/batch", headers=HEADERS, json={"records": records})
                assert retry.status == 200
                assert len(recipe.batches) == 3
            assert handler.calls == 0
            assert len(recipe.batches) == 3

            # Retrying an imported record after training/compaction must not
            # reactivate it. Changed content still conflicts.
            retry = await client.post("/reef/records", headers=HEADERS, json=imported("offline-0"))
            assert retry.status == 200
            conflict = await client.post(
                "/reef/records", headers=HEADERS, json=imported("offline-0", inference_payload("changed"))
            )
            assert conflict.status == 409
            for index in range(2):
                text = f"online-{index}"
                response = await client.post(
                    "/v1/chat/completions",
                    headers=HEADERS,
                    json={"model": "example-model", "messages": [{"role": "user", "content": text}]},
                )
                assert response.status == 200, await response.text()
                receipt = response.headers["x-reef-agent-record-id"]
                feedback = await client.post(
                    "/reef/report", headers=HEADERS, json={"score": 1, "references": [receipt]}
                )
                assert feedback.status == 200, await feedback.text()
            await wait_for_training(scenario, 8)
            assert handler.calls == 2
            assert scenario.trainer.pipeline is processor
            assert len(scenario.store.history()) == len(recipe.batches) == 4
            samples = [sample for batch in recipe.batches for sample in batch.items]
            assert [sample.metadata["messages"][0]["content"] for sample in samples] == [
                *(f"offline-{index}" for index in range(6)),
                "online-0",
                "online-1",
            ]
            assert all(sample.metadata["response"]["choices"] for sample in samples)
            for index in range(6):
                stored = scenario.records.get_for_audit("s", f"offline-{index}")
                assert stored.item.payload == inference_payload(f"offline-{index}")

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_import_admission_authentication_and_scenario_isolation(tmp_path):
    dispatcher = dispatcher_for(tmp_path, Recipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            body = imported("first")
            response = await client.post("/reef/records", json=body, headers={"x-reef-scenario": "s"})
            assert response.status == 401
            response = await client.post(
                "/reef/records", json=body, headers={"Authorization": "Bearer import-test-token"}
            )
            assert response.status == 400
            invalid = [
                [],
                {},
                {**body, "agent_record_id": " "},
                {**body, "agent_record_id": 1},
                {**body, "request_type": "train"},
                {**body, "request_type": "other"},
                {**body, "payload": []},
                {**body, "scenario": "other"},
            ]
            for envelope in invalid:
                response = await client.post("/reef/records", headers=HEADERS, json=envelope)
                assert response.status == 400, await response.text()
            for _ in range(2):
                response = await client.post("/reef/records", headers=HEADERS, json=body)
                assert response.status == 200
                assert await response.json() == {
                    "agent_record_id": "first",
                    "scenario": "s",
                    "request_type": "inference",
                }
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.records.count("s") == 1
            assert scenario.records.get("s", "first").payload == body["payload"]
            other_headers = {**HEADERS, "x-reef-scenario": "other"}
            response = await client.post("/reef/records", headers=other_headers, json=imported("second"))
            assert response.status == 200
            other = dispatcher.get_or_create_scenario("other")
            assert other.records.get("other", "first") is None
            assert scenario.records.get("s", "second") is None
            response = await client.post(
                "/reef/records",
                headers=other_headers,
                json=imported("foreign-report", {"score": 1, "references": ["first"]}, "report"),
            )
            assert response.status == 400
            response = await client.post(
                "/reef/records",
                headers=HEADERS,
                json=imported("bad-report", {"references": ["missing"]}, "report"),
            )
            assert response.status == 400
            assert scenario.records.count("s") == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_imported_reports_follow_the_recipes_report_schema(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            response = await client.post("/reef/records", headers=HEADERS, json=imported("source"))
            assert response.status == 200
            response = await client.post(
                "/reef/records",
                headers=HEADERS,
                json=imported("missing-score", {"references": ["source"]}, "report"),
            )
            assert response.status == 400
            assert dispatcher.get_or_create_scenario("s").records.get("s", "missing-score") is None

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_batch_admission_is_atomic_and_ordered(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            endpoint = "/reef/records/batch"
            response = await client.post(endpoint, json={"records": [imported("first")]})
            assert response.status == 401
            invalid = [
                [],
                {},
                {"records": []},
                {"records": [1]},
                {"records": [{}]},
                {"records": [imported("first")], "scenario": "other"},
                {"records": [imported(str(index)) for index in range(1001)]},
                {"records": [imported("first"), imported("bad", {"references": ["first"]}, "report")]},
                {"records": [imported("first"), imported("bad", {"score": 1, "references": ["missing"]}, "report")]},
                {"records": [imported("bad", {"score": 1, "references": ["first"]}, "report"), imported("first")]},
                {"records": [imported("first"), imported("instruction", {}, "train")]},
            ]
            for body in invalid:
                response = await client.post(endpoint, headers=HEADERS, json=body)
                assert response.status == 400, await response.text()
            scenario = dispatcher.get_or_create_scenario("s")
            assert scenario.records.count("s") == 0

            body = {"records": [imported("first"), imported("first", inference_payload("conflict"))]}
            response = await client.post(endpoint, headers=HEADERS, json=body)
            assert response.status == 409
            assert scenario.records.count("s") == 0
            response = await client.post(endpoint, headers=HEADERS, json={"records": [imported("first")] * 2})
            assert response.status == 200
            assert scenario.records.count("s") == 1
            response = await client.post(
                endpoint,
                headers=HEADERS,
                json={
                    "records": [
                        imported("new"),
                        imported("first", inference_payload("conflict")),
                    ]
                },
            )
            assert response.status == 409
            assert scenario.records.get("s", "new") is None
            response = await client.post(
                endpoint,
                headers={**HEADERS, "x-reef-scenario": "other"},
                json={
                    "records": [imported("bad", {"score": 1, "references": ["first"]}, "report")],
                },
            )
            assert response.status == 400
            response = await client.post(
                endpoint,
                headers=HEADERS,
                json={"records": [imported("large", {"text": "x" * MAX_RECORD_IMPORT_BYTES})]},
            )
            assert response.status == 413
            assert scenario.records.get("s", "large") is None
            assert not dispatcher.recipe.batches

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def atif(text):
    return {
        "schema_version": "ATIF-v1.4",
        "session_id": text,
        "agent": {"name": "fixture", "version": "1"},
        "steps": [{"step_id": 1, "source": "user", "message": text}],
        "extra": {"reef": {"reward": 1.0}},
    }


def write_task(root, name="demo", instruction="Print hello."):
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text(instruction)
    (task / "environment" / "Dockerfile").write_text("FROM alpine\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")
    return task


def task_archive(*entries):
    """A gzip tar of ``(name, path)`` pairs; a path of None adds an empty directory entry."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, path in entries:
            tar.add(path, arcname=name)
    return buffer.getvalue()


class UploadBatches(Task2Batch):
    required_request_types = frozenset(RequestType) - {RequestType.TRAIN}

    def make_batch(self, tasks, batch_id):
        return TrainingBatch(batch_id, tasks)


@dataclass(frozen=True, kw_only=True)
class UploadRecipe(Recipe):
    batches: list[TrainingBatch] = field(default_factory=list)

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None, task_storage=None):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=lambda context: TrainingPipeline(context, task2batch=UploadBatches(context, 2)),
            candidate_backend=LearningBackend(scenario, self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
            task_storage=task_storage,
        )


def test_uploaded_trajectories_and_tasks_train_as_items(tmp_path):
    recipe = UploadRecipe()
    dispatcher = dispatcher_for(tmp_path, recipe)
    task = write_task(tmp_path / "source")
    changed = write_task(tmp_path / "changed", instruction="Print goodbye.")
    task_headers = {**HEADERS, "x-reef-agent-record-id": "task-1"}

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            trajectory = imported("trajectory-1", atif("trajectory-1"), "trajectory")
            response = await client.post("/reef/records/batch", headers=HEADERS, json={"records": [trajectory]})
            assert response.status == 200, await response.text()
            assert (await response.json())["records"][0]["request_type"] == "trajectory"
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", task)))
            assert response.status == 200, await response.text()
            receipt = await response.json()
            assert receipt["request_type"] == "task"
            assert receipt["task"]["name"] == "demo"
            scenario = dispatcher.get_or_create_scenario("s")
            await wait_for_training(scenario, 2)

            # A retry is accepted after training and does not train again; changed content conflicts.
            response = await client.post("/reef/records", headers=HEADERS, json=trajectory)
            assert response.status == 200
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", task)))
            assert response.status == 200
            response = await client.post(
                "/reef/records", headers=HEADERS, json=imported("trajectory-1", atif("other"), "trajectory")
            )
            assert response.status == 409
            response = await client.post("/reef/tasks", headers=task_headers, data=task_archive(("demo", changed)))
            assert response.status == 409
            for invalid in (
                imported("bad", {**atif("bad"), "steps": []}, "trajectory"),
                imported("bad", {**atif("bad"), "schema_version": "v1"}, "trajectory"),
                imported("bad", {"task": {"name": "demo", "digest": receipt["task"]["digest"]}}, "task"),
            ):
                response = await client.post("/reef/records", headers=HEADERS, json=invalid)
                assert response.status == 400, await response.text()
            assert scenario.records.get_for_audit("s", "bad") is None

            (batch,) = recipe.batches
            uploaded_trajectory, uploaded_task = batch.items
            assert isinstance(uploaded_trajectory, TrajectoryItem)
            assert uploaded_trajectory.trajectory == atif("trajectory-1")
            assert uploaded_trajectory.source_agent_record_ids == ("trajectory-1",)
            assert isinstance(uploaded_task, TaskItem)
            assert uploaded_task.source_agent_record_ids == ("task-1",)
            assert uploaded_task.metadata == {"task": receipt["task"]}
            assert uploaded_task.task_path.name == "demo"
            assert (uploaded_task.task_path / "instruction.md").read_text() == "Print hello."
            assert len(scenario.store.history()) == 1

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_uploads_need_a_recipe_that_takes_them(tmp_path):
    dispatcher = dispatcher_for(tmp_path, LearningRecipe())
    task = write_task(tmp_path / "source")

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            response = await client.post(
                "/reef/records", headers=HEADERS, json=imported("trajectory-1", atif("trajectory-1"), "trajectory")
            )
            assert response.status == 400
            assert "does not take trajectory uploads" in await response.text()
            response = await client.post(
                "/reef/tasks",
                headers={**HEADERS, "x-reef-agent-record-id": "task-1"},
                data=task_archive(("demo", task)),
            )
            assert response.status == 400
            assert "does not take task uploads" in await response.text()
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_a_task_upload_must_be_one_harbor_task_directory(tmp_path):
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    task = write_task(tmp_path / "source")
    no_verifier = write_task(tmp_path / "no-verifier")
    (no_verifier / "tests" / "test.sh").unlink()
    (tmp_path / "link").symlink_to(task / "instruction.md")
    headers = {**HEADERS, "x-reef-agent-record-id": "task-1"}

    async def run():
        async with TestClient(TestServer(create_app(dispatcher, tokens="import-test-token"))) as client:
            response = await client.post("/reef/tasks", headers=HEADERS, data=task_archive(("demo", task)))
            assert response.status == 400
            assert "x-reef-agent-record-id" in await response.text()
            for archive in (
                b"not a gzip tar",
                task_archive(("demo", task), ("other", task)),
                task_archive(("demo", task), ("demo/link", tmp_path / "link")),
                task_archive(("demo", task), ("demo/../escape", task / "instruction.md")),
                task_archive(("demo", task / "instruction.md")),
                task_archive(("demo", no_verifier)),
            ):
                response = await client.post("/reef/tasks", headers=headers, data=archive)
                assert response.status == 400, await response.text()
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 0

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()
