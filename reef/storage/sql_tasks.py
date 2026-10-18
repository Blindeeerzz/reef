"""Task stores in the scenario's record database, for every SQL backend.

The task tables live in the same database as the records, and every operation
uses the record store's transactions. Adapters supply the tables and their
namespace; the task store behavior is implemented here once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from sqlalchemy import Table, and_, select, true
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.tasks import GeneratedTask, HarborTask, PlayArm
from reef.storage.sql_records import SQLRecordStore
from reef.storage.tasks import (
    SPLIT_NAMES,
    CandidateOutcome,
    GenerationState,
    SplitName,
    SplitSettings,
    StoredTask,
    TaskConflict,
    TaskStorage,
    TaskStore,
    TaskStoreError,
)


@dataclass(frozen=True)
class TaskTables:
    """Tables for task store settings, generations, candidates, and published tasks.

    Every table has a ``store_name`` column and the adapter's namespace columns
    in ``scope``. ``candidates`` holds one row for each saved candidate, with
    its outcome. ``published`` holds one row for each published task name.
    """

    stores: Table
    generations: Table
    candidates: Table
    published: Table
    scope: Mapping[str, str] = field(default_factory=dict)


class SQLTaskStorage(TaskStorage):
    """Open the task stores kept in one record store's database."""

    def __init__(self, records: SQLRecordStore, tables: TaskTables) -> None:
        self.records = records
        self.tables = tables

    def task_store(self, name: str, *, split: SplitSettings) -> SQLTaskStore:
        return SQLTaskStore(self.records, self.tables, name, split=split)


class SQLTaskStore(TaskStore):
    """One named task store; each operation is one transaction of the record database."""

    def __init__(self, records: SQLRecordStore, tables: TaskTables, name: str, *, split: SplitSettings) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a task store name must be a non-empty string")
        if not isinstance(split, SplitSettings):
            raise TypeError("split must be SplitSettings")
        self.records = records
        self.tables = tables
        self.name = name
        self.key = {**tables.scope, "store_name": name}
        with records._transaction(write=True) as connection:
            records._insert_if_absent(
                connection,
                tables.stores,
                {**self.key, "eval_fraction": split.eval_fraction, "seed": split.seed},
            )
            row = connection.execute(select(tables.stores).where(self.condition(tables.stores))).mappings().one()
        stored = SplitSettings(eval_fraction=row["eval_fraction"], seed=row["seed"])
        if stored != split:
            raise TaskStoreError(
                f"task store {name!r} was created with eval_fraction {stored.eval_fraction} and seed {stored.seed}, "
                f"not eval_fraction {split.eval_fraction} and seed {split.seed}"
            )

    def condition(self, table: Table) -> ColumnElement[bool]:
        """Restrict a query to this store's rows."""
        return and_(true(), *(table.c[key] == value for key, value in self.key.items()))

    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        checked_position(generation)
        if (
            not isinstance(source_record_ids, tuple)
            or any(not isinstance(record_id, str) or not record_id for record_id in source_record_ids)
            or len(set(source_record_ids)) != len(source_record_ids)
        ):
            raise ValueError("source_record_ids must be a tuple of distinct non-empty strings")
        with self.records._transaction(write=True) as connection:
            self.records._insert_if_absent(
                connection,
                self.tables.generations,
                {
                    **self.key,
                    "generation": generation,
                    "source_record_ids_json": json.dumps(list(source_record_ids), ensure_ascii=False),
                    "candidate_count": None,
                    "is_complete": False,
                },
            )
            state = self.read_generation(connection, generation)
        if state is None or state.source_record_ids != source_record_ids:
            raise TaskStoreError(f"generation {generation} of task store {self.name!r} has other source records")
        return state

    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        checked_position(generation)
        if not isinstance(candidates, tuple) or any(not isinstance(item, GeneratedTask) for item in candidates):
            raise TypeError("candidates must be a tuple of GeneratedTask values")
        documents = [generated_document(item) for item in candidates]
        with self.records._transaction(write=True) as connection:
            state = self.read_generation(connection, generation)
            if state is None:
                raise TaskStoreError(f"generation {generation} of task store {self.name!r} has not begun")
            if state.candidates is not None:
                saved = connection.execute(
                    select(self.tables.candidates.c.generated_json)
                    .where(self.condition(self.tables.candidates), self.tables.candidates.c.generation == generation)
                    .order_by(self.tables.candidates.c.candidate_index)
                ).scalars()
                if list(saved) != documents:
                    raise TaskStoreError(f"generation {generation} of task store {self.name!r} has other candidates")
                return state
            if candidates:
                connection.execute(
                    self.tables.candidates.insert(),
                    [
                        {
                            **self.key,
                            "generation": generation,
                            "candidate_index": index,
                            "task_name": item.task.name,
                            "digest": item.task.digest,
                            "generated_json": document,
                            "split": None,
                            "error": None,
                        }
                        for index, (item, document) in enumerate(zip(candidates, documents, strict=True))
                    ],
                )
            connection.execute(
                self.tables.generations.update()
                .where(self.condition(self.tables.generations), self.tables.generations.c.generation == generation)
                .values(candidate_count=len(candidates))
            )
        return replace(state, candidates=candidates)

    def generation(self, generation: int) -> GenerationState | None:
        checked_position(generation)
        with self.records._transaction(write=False) as connection:
            return self.read_generation(connection, generation)

    def generations(self) -> tuple[GenerationState, ...]:
        with self.records._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(self.tables.generations)
                    .where(self.condition(self.tables.generations))
                    .order_by(self.tables.generations.c.generation)
                )
                .mappings()
                .all()
            )
            candidates = (
                connection.execute(
                    select(self.tables.candidates)
                    .where(self.condition(self.tables.candidates))
                    .order_by(self.tables.candidates.c.generation, self.tables.candidates.c.candidate_index)
                )
                .mappings()
                .all()
            )
        by_generation: dict[int, list[RowMapping]] = {}
        for candidate in candidates:
            by_generation.setdefault(int(candidate["generation"]), []).append(candidate)
        return tuple(generation_state(row, by_generation.get(int(row["generation"]), [])) for row in rows)

    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        checked_position(generation, index)
        if split not in SPLIT_NAMES:
            raise ValueError("split must be 'train' or 'eval'")
        candidates = self.tables.candidates
        published = self.tables.published
        with self.records._transaction(write=True) as connection:
            candidate = self.read_candidate(connection, generation, index)
            if candidate["error"] is not None:
                raise TaskStoreError(f"candidate {index} of generation {generation} was rejected")
            if candidate["split"] is not None and candidate["split"] != split:
                raise TaskConflict(
                    f"candidate {index} of generation {generation} is published to {candidate['split']}, not {split}"
                )
            name = candidate["task_name"]
            existing = (
                connection.execute(select(published).where(self.condition(published), published.c.task_name == name))
                .mappings()
                .first()
            )
            if existing is None:
                connection.execute(
                    published.insert().values(
                        **self.key,
                        task_name=name,
                        digest=candidate["digest"],
                        split=split,
                        generation=generation,
                        candidate_index=index,
                    )
                )
                publisher = candidate
            elif existing["digest"] != candidate["digest"]:
                raise TaskConflict(
                    f"task {name!r} is published with digest {existing['digest']}, not {candidate['digest']}"
                )
            elif existing["split"] != split:
                raise TaskConflict(f"task {name!r} is published to {existing['split']}, not {split}")
            else:
                publisher = self.read_candidate(
                    connection, int(existing["generation"]), int(existing["candidate_index"])
                )
            if candidate["split"] is None:
                connection.execute(
                    candidates.update()
                    .where(
                        self.condition(candidates),
                        candidates.c.generation == generation,
                        candidates.c.candidate_index == index,
                    )
                    .values(split=split)
                )
        return StoredTask(
            generated=generated_from_document(publisher["generated_json"]),
            split=split,
            generation=int(publisher["generation"]),
            index=int(publisher["candidate_index"]),
        )

    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        checked_position(generation, index)
        if not isinstance(reason, str) or not reason:
            raise ValueError("a rejection needs a reason")
        candidates = self.tables.candidates
        with self.records._transaction(write=True) as connection:
            candidate = self.read_candidate(connection, generation, index)
            if candidate["split"] is not None:
                raise TaskStoreError(f"candidate {index} of generation {generation} is published")
            if candidate["error"] is not None:
                if candidate["error"] != reason:
                    raise TaskStoreError(
                        f"candidate {index} of generation {generation} was rejected for another reason"
                    )
                return
            connection.execute(
                candidates.update()
                .where(
                    self.condition(candidates),
                    candidates.c.generation == generation,
                    candidates.c.candidate_index == index,
                )
                .values(error=reason)
            )

    def complete_generation(self, *, generation: int) -> GenerationState:
        checked_position(generation)
        with self.records._transaction(write=True) as connection:
            state = self.read_generation(connection, generation)
            if state is None:
                raise TaskStoreError(f"generation {generation} of task store {self.name!r} has not begun")
            if state.candidates is None or state.pending_indices:
                raise TaskStoreError(f"generation {generation} of task store {self.name!r} has unfinished candidates")
            connection.execute(
                self.tables.generations.update()
                .where(self.condition(self.tables.generations), self.tables.generations.c.generation == generation)
                .values(is_complete=True)
            )
        return replace(state, is_complete=True)

    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        if split is not None and split not in SPLIT_NAMES:
            raise ValueError("split must be 'train', 'eval' or None")
        candidates = self.tables.candidates
        published = self.tables.published
        statement = (
            select(
                published.c.split, candidates.c.generation, candidates.c.candidate_index, candidates.c.generated_json
            )
            .join(
                candidates,
                and_(
                    *(candidates.c[key] == published.c[key] for key in self.key),
                    candidates.c.generation == published.c.generation,
                    candidates.c.candidate_index == published.c.candidate_index,
                ),
            )
            .where(self.condition(published))
            .order_by(published.c.generation, published.c.candidate_index)
        )
        if split is not None:
            statement = statement.where(published.c.split == split)
        with self.records._transaction(write=False) as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(
            StoredTask(
                generated=generated_from_document(row["generated_json"]),
                split=row["split"],
                generation=int(row["generation"]),
                index=int(row["candidate_index"]),
            )
            for row in rows
        )

    def read_generation(self, connection: Connection, generation: int) -> GenerationState | None:
        """One generation checkpoint, read in the caller's transaction."""
        row = (
            connection.execute(
                select(self.tables.generations).where(
                    self.condition(self.tables.generations), self.tables.generations.c.generation == generation
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        candidates = (
            connection.execute(
                select(self.tables.candidates)
                .where(self.condition(self.tables.candidates), self.tables.candidates.c.generation == generation)
                .order_by(self.tables.candidates.c.candidate_index)
            )
            .mappings()
            .all()
        )
        return generation_state(row, candidates)

    def read_candidate(self, connection: Connection, generation: int, index: int) -> RowMapping:
        """One saved candidate with its outcome, read in the caller's transaction."""
        candidates = self.tables.candidates
        row = (
            connection.execute(
                select(candidates).where(
                    self.condition(candidates),
                    candidates.c.generation == generation,
                    candidates.c.candidate_index == index,
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise TaskStoreError(f"generation {generation} of task store {self.name!r} has no candidate {index}")
        return row


def checked_position(generation: int, index: int | None = None) -> None:
    """Refuse a generation or candidate index that is not a non-negative integer."""
    for label, value in (("generation", generation), ("candidate index", index)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"a {label} must be a non-negative integer")


def generation_state(row: RowMapping, candidates: Sequence[RowMapping]) -> GenerationState:
    """A generation checkpoint from its row and its candidate rows in index order."""
    saved = row["candidate_count"] is not None
    outcomes = []
    for candidate in candidates:
        if candidate["split"] is not None:
            outcomes.append(CandidateOutcome(index=int(candidate["candidate_index"]), split=candidate["split"]))
        elif candidate["error"] is not None:
            outcomes.append(CandidateOutcome(index=int(candidate["candidate_index"]), error=candidate["error"]))
    return GenerationState(
        generation=int(row["generation"]),
        source_record_ids=tuple(json.loads(row["source_record_ids_json"])),
        candidates=tuple(generated_from_document(item["generated_json"]) for item in candidates) if saved else None,
        outcomes=tuple(outcomes),
        is_complete=bool(row["is_complete"]),
    )


def generated_document(generated: GeneratedTask) -> str:
    """A generated task as canonical JSON text, so that an identical retry stores the same text."""
    return json.dumps(
        {
            "task": generated.task.document(),
            "measurement_arms": [
                {"name": arm.name, "count": arm.count, "extra_instruction_files": list(arm.extra_instruction_files)}
                for arm in generated.measurement_arms
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def generated_from_document(text: str) -> GeneratedTask:
    """The generated task that :func:`generated_document` wrote; its constructors validate the content again."""
    document = json.loads(text)
    task = document["task"]
    return GeneratedTask(
        task=HarborTask(**{**task, "source_agent_record_ids": tuple(task["source_agent_record_ids"])}),
        measurement_arms=tuple(
            PlayArm(
                name=arm["name"],
                count=arm["count"],
                extra_instruction_files=tuple(arm["extra_instruction_files"]),
            )
            for arm in document["measurement_arms"]
        ),
    )
