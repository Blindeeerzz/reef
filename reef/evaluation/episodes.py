"""One harness evaluation episode: run it, score it, and record what it showed.

An :class:`EpisodeEvaluationWorker` runs one task against one rendered harness
tree with :func:`reef.harness.episodes.run.run_episode` and scores the result
with the method's :class:`EpisodeScorer`. The worker returns a
:class:`ScoredEpisode`: the score, or a :class:`FailureObservation` that says
why the episode has none, and what the trajectory showed. Workers run in the
pool of :mod:`reef.evaluation.workers`; :mod:`reef.evaluation.harness` runs
the evaluation tasks of a harness candidate through that pool.
"""

from __future__ import annotations

import json
import math
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from reef.core.plugins import ConfiguredPlugin
from reef.core.tasks import primary_reward
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import EPISODE_OWNER_LEASE, EpisodeExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, final_assistant_text
from reef.harness.tree.nodes import redact_secret_shaped
from reef.runtime.executor.requirements import ExecutionRequirements

#: Characters kept per text in the step record; a longer text ends in a clip marker.
RECORD_TEXT_CAP = 20_000
#: The record file beside one evaluation episode's kept trajectory.
RECORD_EPISODE_FILE = "episode.json"


def clip_record_text(text: str) -> str:
    """``text`` with credential-shaped literals redacted, cut at the record cap with a marker naming how much was dropped.

    The record holds model traffic and proposals before the tree boundary
    saw them, so the boundary's credential tripwire runs here too."""
    text = redact_secret_shaped(text)
    if len(text) <= RECORD_TEXT_CAP:
        return text
    return f"{text[:RECORD_TEXT_CAP]}... [clipped {len(text) - RECORD_TEXT_CAP} chars]"


class EpisodeScorer(ConfiguredPlugin, ABC):
    """Base class for scoring one harness-evolution episode.

    Implement:
        ``score`` — given a task name and its episode result, return a
        float score. Higher is better; non-finite values raise.
    """

    @abstractmethod
    def score(self, task: str, result: EpisodeResult) -> float:
        """Score one episode result for a task."""

    def execution_requirements(self) -> ExecutionRequirements:
        """Override when scoring loads a local GPU model or needs cluster placement.

        API calls do not request local GPUs. GPU scorers must initialize their
        model lazily in the allocated worker, not while the recipe is built.
        """
        return ExecutionRequirements()

    def score_with_models(self, task: str, result: EpisodeResult, models: ModelBindings) -> float:
        """Model-based judges override this method and use the supplied bindings."""
        return self.score(task, result)


def verifier_reward(task: str, result: EpisodeResult) -> float:
    """The Harbor verifier's reward for a task directory episode; :class:`VerifierRewardScorer` scores with it.

    The terminus runner writes one ``verifier`` row per episode with the task it played and the rewards
    its verifier wrote; Harbor's primary reward is the ``reward`` entry, else the sole entry. A failed
    episode or a verifier that wrote nothing scores 0 (the gate has already set aside an episode whose
    trial never ran); an episode that exited without a row scores 0; a row for another task, several
    rewards without a ``reward`` entry, or a reward that is not a finite number is an error.
    """
    rows = [event for event in result.trajectory if event.get("type") == "verifier"]
    if not rows and result.exit_code:
        return 0.0
    if len(rows) != 1:
        raise ValueError(f"expected one verifier record for {task!r}, found {len(rows)}")
    row = rows[0]
    if row.get("task") != task:
        raise ValueError(f"the verifier record names {row.get('task')!r}, not {task!r}")
    if row.get("failed"):
        return 0.0
    rewards = row.get("rewards")
    reward = primary_reward(rewards) if isinstance(rewards, Mapping) else row.get("reward")
    if reward is None:
        if isinstance(rewards, Mapping) and rewards:
            raise ValueError(f"the verifier for {task!r} wrote {sorted(rewards)} and no 'reward' entry")
        return 0.0
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"the verifier reward for {task!r} must be a finite number, not {reward!r}")
    return float(reward)


class VerifierRewardScorer(EpisodeScorer):
    """Score an episode by its Harbor verifier's reward (:func:`verifier_reward`); for a task manifest evaluation."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return verifier_reward(task, result)


@dataclass(frozen=True)
class FailureObservation:
    """One failed episode as the evaluation saw it, before any diffing."""

    task: str
    stage: str
    cause: str

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task, "stage": self.stage, "cause": self.cause}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureObservation:
        return cls(task=str(value["task"]), stage=str(value["stage"]), cause=str(value["cause"]))


@dataclass(frozen=True)
class ScoredEpisode:
    """One evaluation episode as the evaluation keeps it: the score, or why it has none, and what the trajectory showed."""

    score: float | None
    failure: FailureObservation | None
    residue: int = 0
    agents: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The root's stage path and end reason; ``None`` when no trajectory was read.
    path: dict[str, Any] | None = None
    #: The final assistant text the trajectory holds, the reply a text grader reads; ``None`` when it holds none.
    reply: str | None = None
    #: Whether the reader found a session log; a scored episode without one left a text grader nothing to read.
    transcript_read: bool = True
    #: Remote workers return the kept trajectory; the driver owns its durable path.
    record_archive: bytes | None = field(default=None, repr=False)


@dataclass(repr=False)
class EpisodeEvaluationWorker:
    """Only episode/scorer state crosses the process boundary, never the recipe tree."""

    descriptor: AdapterDescriptor
    scorer: EpisodeScorer
    binary: str | None
    timeout: float
    executor: EpisodeExecutor
    forbid_residue: bool
    owner_lease: bool = False
    transfer_records: bool = False

    def __post_init__(self) -> None:
        self.executor.preflight()

    def run(
        self, files: Mapping[str, str], task: str, keep_dir: Path | None = None, models: ModelBindings | None = None
    ) -> ScoredEpisode:
        if keep_dir is None or not self.transfer_records:
            return self._run_and_score(files, task, keep_dir, models)
        # Remote workers must not interpret the driver's path as a local path.
        # Keep the trajectory on the worker, then return it with the scored result.
        with tempfile.TemporaryDirectory(prefix="reef-worker-record-") as temporary:
            record_dir = Path(temporary) / "episode"
            scored = self._run_and_score(files, task, record_dir, models)
            buffer = BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(record_dir, arcname=".")
            return replace(scored, record_archive=buffer.getvalue())

    def _run_and_score(
        self, files: Mapping[str, str], task: str, keep_dir: Path | None, models: ModelBindings | None
    ) -> ScoredEpisode:
        """Score one side's episode; a ``None`` score marks an episode that
        could not run. The observation keeps what the exception handling
        would otherwise discard: the failure's stage and cause. A native turn
        that ended on an error could not run either; any other nonzero exit
        still scores, as before, and is observed alongside the score.
        The residue counts files the episode left outside the cleanup
        whitelist; with ``forbid_residue`` a littering episode scores as one
        that could not run. ``keep_dir`` receives the episode's trajectory
        files before its root is removed; a copy that fails is not an
        episode failure and propagates, so the step aborts instead of
        scoring a result shaped by a disk error."""
        token = EPISODE_OWNER_LEASE.set(self.owner_lease)
        try:
            result = run_episode(
                self.descriptor,
                files,
                task,
                binary=self.binary,
                timeout=self.timeout,
                executor=self.executor,
                keep_dir=keep_dir,
            )
        except EpisodeError as error:
            scored = ScoredEpisode(None, FailureObservation(task=task, stage="launch", cause=str(error)))
            write_episode_record(keep_dir, task, None, scored)
            return scored
        except TrajectoryError as error:
            scored = ScoredEpisode(None, FailureObservation(task=task, stage="trajectory", cause=str(error)))
            write_episode_record(keep_dir, task, None, scored)
            return scored
        finally:
            EPISODE_OWNER_LEASE.reset(token)
        scored = self._score_result(result, task, models)
        write_episode_record(keep_dir, task, result, scored)
        return scored

    def _score_result(self, result: EpisodeResult, task: str, models: ModelBindings | None = None) -> ScoredEpisode:
        """The score and the observations of an episode that ran."""
        residue = len(result.residue)
        agents = agent_work(result.trajectory)
        path = stage_path(result.trajectory)
        if residue and self.forbid_residue:
            cause = f"{residue} file(s) outside the cleanup whitelist: {result.residue[0]}"
            return ScoredEpisode(
                None, FailureObservation(task=task, stage="residue", cause=cause), residue, agents, path
            )
        trial_error = failed_trial_error(result.trajectory)
        if trial_error:
            # The terminus runner recorded a trial that never ran (the image did not build, the agent could not
            # start): no answer was given, so it ranks below every real score instead of tying a zero.
            return ScoredEpisode(
                None, FailureObservation(task=task, stage="trial", cause=trial_error), residue, agents, path
            )
        if path.get("error") is not None:
            # The native loop ended its turn on an error (a tree that cannot load, a graph that cannot run, an
            # agent whose failure ended the run): nothing it wrote is an answer, so it ranks below every real
            # score instead of tying a zero.
            error = path["error"]
            cause = f"{error.get('code', 'error')}: {error.get('message', '')}".strip(": ")
            if path.get("errored_agent"):
                cause = f"agent {path['errored_agent']}: {cause}"
            # A loop turn walks no graph: the failure names the loop when the root's header does.
            stage = "loop" if root_header(result.trajectory).get("loop") else "graph"
            return ScoredEpisode(None, FailureObservation(task=task, stage=stage, cause=cause), residue, agents, path)
        score = float(
            self.scorer.score(task, result) if models is None else self.scorer.score_with_models(task, result, models)
        )
        if not math.isfinite(score):
            raise ValueError(f"episode scorer returned a non-finite score {score!r} for task {task!r}")
        reply = final_assistant_text(result.trajectory)
        if result.exit_code != 0:
            stderr_lines = result.stderr.strip().splitlines()
            cause = f"exit {result.exit_code}: {stderr_lines[-1] if stderr_lines else ''}".strip()
            return ScoredEpisode(
                score, FailureObservation(task=task, stage="exit", cause=cause), residue, agents, path, reply
            )
        # An empty trajectory is no failure: a grader that reads files scored the run as it stands. The summary
        # still says no transcript was read, so a low score a text grader gave is not blamed on the request.
        return ScoredEpisode(score, None, residue, agents, path, reply, transcript_read=bool(result.trajectory))


def failed_trial_error(trajectory: Sequence[Mapping[str, Any]]) -> str:
    """The error of a terminus trial that never ran (a failed ``verifier`` row with an error), else empty."""
    for event in trajectory:
        if event.get("type") == "verifier" and event.get("failed") and event.get("error"):
            return str(event["error"])
    return ""


def write_episode_record(
    keep_dir: Path | None, task: str, result: EpisodeResult | None, scored: ScoredEpisode
) -> None:
    """``episode.json`` beside the kept trajectory: what the scorer saw, so a result can be re-derived from the record."""
    if keep_dir is None:
        return
    record = {
        "task": clip_record_text(task),
        "score": scored.score,
        "failure": None if scored.failure is None else scored.failure.to_dict(),
        "path": scored.path,
        "exit_code": None if result is None else result.exit_code,
        "stdout": None if result is None else clip_record_text(result.stdout),
        "stderr": None if result is None else clip_record_text(result.stderr),
        "residue": None if result is None else [clip_record_text(str(path)) for path in result.residue],
    }
    # An episode that never wrote an event has no trajectory copy, so the directory may not exist yet.
    keep_dir.mkdir(parents=True, exist_ok=True)
    with open(keep_dir / RECORD_EPISODE_FILE, "x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2, default=str) + "\n")


def root_header(trajectory: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The root's ``session`` line of a native-jsonl trajectory (the agents' files sort first); empty for other formats."""
    for event in trajectory:
        data = event.get("data") or {}
        if event.get("type") == "session" and str(data.get("agent") or "root") == "root":
            return data
    return {}


def stage_path(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The root's ``stage/exit`` stage names in order and its ``turn/end`` reason kind; empty for other formats."""
    stages: list[str] = []
    reason: str | None = None
    errors: dict[str, Mapping[str, Any]] = {}
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
        elif type_ == "turn/end" and agent is not None:
            details = data.get("reason") or {}
            kind = details.get("kind")
            if kind == "error" and isinstance(details.get("error"), Mapping):
                errors[agent] = details["error"]
            if agent == "root":
                reason = None if kind is None else str(kind)
        elif agent != "root":
            continue
        elif type_ == "stage/exit":
            stages.append(str(data.get("stage")))
    path: dict[str, Any] = {"stages": stages, "reason": reason}
    # The root's own error, else the error of an agent whose failure ended the run before the root wrote its end:
    # a subagent's model error aborts the whole run with no root turn/end.
    errored = "root" if "root" in errors else next((name for name in errors if reason is None), None)
    if errored is not None:
        path["error"] = dict(errors[errored])
        if errored != "root":
            path["errored_agent"] = errored
    return path


#: The counters a result carries per agent; the token pair is what the endpoint reported, zero when it reported none.
AGENT_COUNTERS = ("turns", "steps", "tool_calls", "tool_errors", "input_tokens", "output_tokens")


def agent_work(trajectory: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Turns, steps, tool calls, tool errors and reported tokens per agent of a
    native-jsonl trajectory; empty for other formats."""
    work: dict[str, dict[str, int]] = {}
    agent: str | None = None
    for event in trajectory:
        type_, data = event.get("type"), event.get("data") or {}
        if type_ == "session":
            agent = str(data.get("agent") or "root")
            work.setdefault(agent, dict.fromkeys(AGENT_COUNTERS, 0))["turns"] += 1
        elif agent is None:
            continue
        elif type_ in ("assistant/message", "context/compacted") and isinstance(data.get("usage"), Mapping):
            work[agent]["input_tokens"] += int(data["usage"].get("input_tokens", 0) or 0)
            work[agent]["output_tokens"] += int(data["usage"].get("output_tokens", 0) or 0)
        elif type_ == "step/start":
            work[agent]["steps"] += 1
        elif type_ == "tool/call":
            work[agent]["tool_calls"] += 1
        elif type_ == "tool/result" and data.get("is_error"):
            work[agent]["tool_errors"] += 1
    return work


def sum_agents(runs: Iterable[Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, int]]:
    """The per-agent counters of several episodes, summed per agent and sorted by agent name."""
    total: dict[str, dict[str, int]] = {}
    for work in runs:
        for agent, counts in work.items():
            sums = total.setdefault(agent, dict.fromkeys(AGENT_COUNTERS, 0))
            for key, value in counts.items():
                sums[key] = sums.get(key, 0) + value
    return {agent: total[agent] for agent in sorted(total)}
