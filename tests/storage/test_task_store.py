"""Generated task store contract on every bundled scenario storage."""

from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.tasks import GeneratedTask, HarborTask, PlayArm
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import (
    CandidateOutcome,
    GenerationState,
    SplitSettings,
    StoredTask,
    TaskConflict,
    TaskStoreError,
)

SPLIT = SplitSettings(eval_fraction=0.25, seed=7)


def generated(name: str, *, instruction: str = "", sources: tuple[str, ...] = ("record-1",)) -> GeneratedTask:
    return GeneratedTask(
        task=HarborTask(
            name=name,
            instruction=instruction or f"Solve {name}.",
            tests={"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
            environment={"Dockerfile": "FROM python:3.12-slim\n"},
            config={
                "verifier": {"timeout_sec": 30},
                "environment": {"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
            },
            metadata={"designer_record_id": "designer-1"},
            solution={"hint.txt": "look closer"},
            source_agent_record_ids=sources,
        ),
        measurement_arms=(PlayArm("hinted", 2, ("solution/hint.txt",)),),
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresScenarioStorage(url, tmp_path, schema=schema)
    else:
        opened = SQLiteScenarioStorage(None if request.param == "memory" else tmp_path)
    with closing(opened):
        yield opened


@pytest.fixture
def session(storage):
    with closing(storage.open("math")) as session:
        yield session


def test_a_generation_records_candidates_and_outcomes(session):
    store = session.task_store("tasks", split=SPLIT)
    candidates = (generated("alpha"), generated("beta", sources=("record-2",)), generated("gamma"))

    assert store.begin_generation(generation=0, source_record_ids=("record-1", "record-2")) == GenerationState(
        generation=0, source_record_ids=("record-1", "record-2")
    )
    assert store.save_candidates(generation=0, candidates=candidates).candidates == candidates
    alpha = store.publish_task(generation=0, index=0, split="train")
    store.publish_task(generation=0, index=1, split="eval")
    store.reject_task(generation=0, index=2, reason="the oracle failed")
    completed = store.complete_generation(generation=0)

    assert alpha == StoredTask(generated=candidates[0], split="train", generation=0, index=0)
    assert completed == GenerationState(
        generation=0,
        source_record_ids=("record-1", "record-2"),
        candidates=candidates,
        outcomes=(
            CandidateOutcome(0, split="train"),
            CandidateOutcome(1, split="eval"),
            CandidateOutcome(2, error="the oracle failed"),
        ),
        is_complete=True,
    )
    assert store.generation(0) == completed
    assert store.generation(1) is None
    assert store.generations() == (completed,)
    assert [task.generated.task.name for task in store.tasks()] == ["alpha", "beta"]
    assert [task.generated for task in store.tasks(split="eval")] == [candidates[1]]


def test_identical_retries_change_nothing(session):
    store = session.task_store("tasks", split=SPLIT)
    candidates = (generated("alpha"), generated("beta"))
    for _ in range(2):
        store.begin_generation(generation=3, source_record_ids=("record-1",))
        store.save_candidates(generation=3, candidates=candidates)
        store.publish_task(generation=3, index=0, split="train")
        store.reject_task(generation=3, index=1, reason="invalid")
        store.complete_generation(generation=3)

    state = store.generation(3)
    assert state is not None and state.is_complete
    assert state.outcomes == (CandidateOutcome(0, split="train"), CandidateOutcome(1, error="invalid"))
    assert len(store.tasks()) == 1


def test_conflicting_generation_writes_are_refused(session):
    store = session.task_store("tasks", split=SPLIT)
    with pytest.raises(TaskStoreError, match="has not begun"):
        store.save_candidates(generation=0, candidates=(generated("alpha"),))
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    with pytest.raises(TaskStoreError, match="other source records"):
        store.begin_generation(generation=0, source_record_ids=("record-2",))
    store.save_candidates(generation=0, candidates=(generated("alpha"), generated("beta")))
    with pytest.raises(TaskStoreError, match="other candidates"):
        store.save_candidates(generation=0, candidates=(generated("alpha"),))
    with pytest.raises(TaskStoreError, match="no candidate 2"):
        store.publish_task(generation=0, index=2, split="train")
    with pytest.raises(TaskStoreError, match="unfinished candidates"):
        store.complete_generation(generation=0)

    store.publish_task(generation=0, index=0, split="train")
    store.reject_task(generation=0, index=1, reason="invalid")
    with pytest.raises(TaskConflict, match="published to train, not eval"):
        store.publish_task(generation=0, index=0, split="eval")
    with pytest.raises(TaskStoreError, match="is published"):
        store.reject_task(generation=0, index=0, reason="invalid")
    with pytest.raises(TaskStoreError, match="was rejected"):
        store.publish_task(generation=0, index=1, split="train")
    with pytest.raises(TaskStoreError, match="another reason"):
        store.reject_task(generation=0, index=1, reason="duplicate")


def test_a_task_name_is_published_once(session):
    store = session.task_store("tasks", split=SPLIT)
    store.begin_generation(generation=0, source_record_ids=("record-1",))
    store.save_candidates(generation=0, candidates=(generated("alpha"),))
    first = store.publish_task(generation=0, index=0, split="eval")
    store.begin_generation(generation=1, source_record_ids=("record-1",))
    store.save_candidates(
        generation=1, candidates=(generated("alpha"), generated("alpha"), generated("alpha", instruction="Other."))
    )

    with pytest.raises(TaskConflict, match="published to eval, not train"):
        store.publish_task(generation=1, index=0, split="train")
    assert store.publish_task(generation=1, index=1, split="eval") == first
    with pytest.raises(TaskConflict, match="published with digest"):
        store.publish_task(generation=1, index=2, split="eval")
    assert store.tasks() == (first,)


def test_split_settings_are_fixed_and_names_are_separate(session):
    alpha = session.task_store("alpha", split=SPLIT)
    alpha.begin_generation(generation=0, source_record_ids=("record-1",))
    alpha.save_candidates(generation=0, candidates=(generated("one"),))
    alpha.publish_task(generation=0, index=0, split="train")

    with pytest.raises(TaskStoreError, match=r"eval_fraction 0\.25 and seed 7"):
        session.task_store("alpha", split=SplitSettings(eval_fraction=0.5, seed=7))
    beta = session.task_store("beta", split=SplitSettings(eval_fraction=0.0, seed=0))
    assert beta.tasks() == ()
    assert beta.generations() == ()
    assert [task.generated.task.name for task in session.task_store("alpha", split=SPLIT).tasks()] == ["one"]


def test_a_closed_session_opens_no_task_store(storage):
    session = storage.open("math")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.task_store("tasks", split=SPLIT)


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path, schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path)


def test_tasks_survive_a_restart_and_archive_retires_them(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        store = session.task_store("tasks", split=SPLIT)
        store.begin_generation(generation=0, source_record_ids=("record-1",))
        store.save_candidates(generation=0, candidates=(generated("alpha"), generated("beta")))
        store.publish_task(generation=0, index=0, split="train")

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            store = session.task_store("tasks", split=SPLIT)
            state = store.generation(0)
            assert state is not None and state.pending_indices == (1,)
            assert [task.generated for task in store.tasks()] == [generated("alpha")]
        storage.archive("math")
        with closing(storage.open("math")) as session:
            assert session.task_store("tasks", split=SPLIT).generations() == ()


def test_an_archived_postgres_session_writes_no_task(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as session,
    ):
        store = session.task_store("tasks", split=SPLIT)
        storage.archive("math")
        with pytest.raises(RuntimeError, match="archived"):
            store.begin_generation(generation=0, source_record_ids=("record-1",))
