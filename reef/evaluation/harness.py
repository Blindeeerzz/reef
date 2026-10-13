"""Harness candidate evaluation: run the evaluation tasks, then decide by the scores.

:class:`HarnessEpisodeEvaluation` runs each evaluation task on the candidate
tree and, for a comparison, on the current tree, in the worker pool of
:mod:`reef.evaluation.workers`. It returns one :class:`EvaluationResult` with
the scores of each side. Two selection policies read that result:
:class:`ScoreComparisonPlugin` selects a candidate with more task wins than
losses, and :class:`FloorPlugin` runs the candidate alone and selects it when
every task scores at least ``floor_score``.
"""

from __future__ import annotations

import math
import tarfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Self

from reef.core.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)
from reef.core.plugins import PluginConfigError
from reef.evaluation.episodes import EpisodeEvaluationWorker, EpisodeScorer, ScoredEpisode, sum_agents
from reef.evaluation.evaluators import BackendEvaluateMixin, SelectionPlugin
from reef.evaluation.workers import EvaluationWorkerPool
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import EpisodeExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import TrajectoryKeepError
from reef.harness.tree.nodes import redact_secret_shaped
from reef.runtime.executor import WorkerSpec
from reef.runtime.executor.config import ExecutorSelection
from reef.runtime.executor.requirements import ExecutionRequirements

#: The trees an evaluation step runs, in pairing order; a policy that evaluates the candidate alone names the first only.
EVALUATION_SIDES: tuple[str, ...] = ("candidate", "current")
#: One evaluation episode: the rendered files, the task, and the directory that keeps its record (``None``: no record).
EpisodePairing = tuple[Mapping[str, str], str, Path | None]
#: How many candidate episodes a step's evaluation summarizes, and how much of each text it keeps: a summary for
#: the pages, never the traffic.
EPISODE_SUMMARIES = 8
EPISODE_SUMMARY_CHARS = 240


def clip_redacted(text: str, limit: int) -> str:
    """``text`` redacted as the record is, cut at ``limit`` characters with an ellipsis marker."""
    text = redact_secret_shaped(text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def episode_record_name(side: str, task_index: int, repeat: int) -> str:
    """The record directory of one evaluation episode: ``<side>-<task index>``, a repeat adding ``-<repeat>``."""
    return f"{side}-{task_index}" if repeat == 0 else f"{side}-{task_index}-{repeat}"


class HarnessEpisodeEvaluation:
    """Run the evaluation tasks of a harness candidate in a worker pool and collect the scores of each side.

    The caller renders the tree of each side and validates its arguments; this
    object owns the worker pool, so the caller must :meth:`close` it.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        scorer: EpisodeScorer,
        binary: str | None,
        episode_timeout_s: float,
        executor: EpisodeExecutor,
        forbid_residue: bool,
        models: ModelBindings,
        episode_repeats: int,
        worker_selection: ExecutorSelection,
        worker_requirements: ExecutionRequirements,
    ) -> None:
        self.episode_repeats = episode_repeats
        self.worker_selection = worker_selection
        self.worker_requirements = worker_requirements
        self._models = models
        self._worker_pool = EvaluationWorkerPool(
            worker_selection,
            worker_requirements,
            WorkerSpec(
                EpisodeEvaluationWorker,
                args=(
                    descriptor,
                    scorer,
                    binary,
                    episode_timeout_s,
                    executor,
                    forbid_residue,
                    worker_selection.settings.backend != "uni",
                    worker_selection.settings.backend not in ("uni", "mp"),
                ),
            ),
        )

    def close(self) -> None:
        self._worker_pool.close()

    def episode_count(self, task_count: int, side_count: int) -> int:
        """How many episodes :meth:`evaluate` runs for ``task_count`` tasks on ``side_count`` sides."""
        return task_count * self.episode_repeats * side_count

    def evaluate(
        self,
        files: Mapping[str, Mapping[str, str]],
        evaluation_tasks: Sequence[str],
        *,
        episodes_dir: Path | None,
    ) -> EvaluationResult:
        """Run every task on each side of ``files`` and return the scores of each side.

        ``files`` maps each side to run, a name from :data:`EVALUATION_SIDES`
        in that order, to the rendered tree of that side. With
        ``episodes_dir``, each episode keeps its record in a directory under it.
        """
        sides = tuple(files)
        # Episodes interleave candidate and current inside each pairing, so
        # anything that drifts during the run (upstream load, rate limits)
        # lands on both sides of a pair instead of one whole side. A repeat
        # is one more pairing of the same task; the selector compares the
        # vectors positionally, so every pairing tallies on its own. Each
        # episode is independent work in its own root, so with more than one
        # worker the pairings run in one pool - a large task set costs one
        # wave instead of a long turn-taking pass - and the results are read
        # back in submission order either way.
        pairings: list[EpisodePairing] = [
            (
                files[side],
                task,
                None if episodes_dir is None else episodes_dir / episode_record_name(side, index, repeat),
            )
            for index, task in enumerate(evaluation_tasks)
            for repeat in range(self.episode_repeats)
            for side in sides
        ]
        scored = self.run_episodes(pairings)
        runs = {side: scored[offset :: len(sides)] for offset, side in enumerate(sides)}
        tasks = {side: [pairing[1] for pairing in pairings][offset :: len(sides)] for offset, side in enumerate(sides)}
        scores = {side: tuple(run.score for run in runs[side]) for side in sides}
        metrics: dict[str, Any] = {
            "candidate_scores": scores.get("candidate", ()),
            "current_scores": scores.get("current", ()),
            "episode_failures": sum(score is None for side in sides for score in scores[side]),
            "episode_repeats": self.episode_repeats,
        }
        for side in sides:
            # Failure observations ride the evaluation so settlement can
            # build the committed side's manifest from the decision alone.
            metrics[f"{side}_failures"] = tuple(run.failure.to_dict() for run in runs[side] if run.failure is not None)
            metrics[f"{side}_residue"] = sum(run.residue for run in runs[side])
            metrics[f"{side}_score"] = float(sum(score for score in scores[side] if score is not None))
            # Per agent sums over the side's episodes, so a result says which agent did the work.
            metrics[f"{side}_agents"] = sum_agents(run.agents for run in runs[side])
            # Per episode, in pairing order: the root's stage path and how its turn ended.
            metrics[f"{side}_paths"] = tuple(run.path for run in runs[side])
        if "candidate" in sides:
            # What the first candidate episodes were graded on, so a rejected step names the task, the reply and why.
            metrics["candidate_episodes"] = [
                {
                    "task": clip_redacted(task, EPISODE_SUMMARY_CHARS),
                    "score": run.score,
                    "failure": (
                        None if run.failure is None else clip_redacted(run.failure.cause, EPISODE_SUMMARY_CHARS)
                    ),
                    "reply": None if run.reply is None else clip_redacted(run.reply, EPISODE_SUMMARY_CHARS),
                    # Only when missing, so a summary of an ordinary episode keeps its shape.
                    **({} if run.transcript_read else {"transcript_read": False}),
                }
                for task, run in list(zip(tasks["candidate"], runs["candidate"], strict=True))[:EPISODE_SUMMARIES]
            ]
        if sides != EVALUATION_SIDES:
            metrics["evaluation_sides"] = list(sides)
        return EvaluationResult(evaluator="harness_episode_pairs", evaluator_version="1", metrics=metrics)

    def run_episodes(self, pairings: Sequence[EpisodePairing]) -> list[ScoredEpisode]:
        """Run and score each pairing in the worker pool, in submission order.

        A remote worker returns the kept trajectory as an archive; it is
        extracted here into the pairing's record directory.
        """
        scored = self._worker_pool.evaluate(pairings, models=self._models)
        for (_, _, keep_dir), result in zip(pairings, scored, strict=True):
            # A worker returns an archive only for a pairing that keeps its record.
            if keep_dir is None or result.record_archive is None:
                continue
            try:
                # Never merge into an earlier episode's record. Extraction
                # rejects path traversal, escaping links and special files.
                keep_dir.mkdir(parents=True)
                with tarfile.open(fileobj=BytesIO(result.record_archive), mode="r:gz") as archive:
                    archive.extractall(keep_dir, filter="data")
            except (OSError, tarfile.TarError) as exc:
                raise TrajectoryKeepError(f"cannot keep remote episode trajectory under {keep_dir}: {exc}") from exc
        return scored


class ScoreComparisonMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when wins exceed losses by ``min_win_margin`` (0: plain majority)."""

    def __init__(self, *, min_win_margin: int = 0) -> None:
        if isinstance(min_win_margin, bool) or not isinstance(min_win_margin, int) or min_win_margin < 0:
            raise ValueError("min_win_margin must be an integer of at least 0")
        super().__init__()
        self._min_win_margin = min_win_margin

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        candidate_scores, current_scores = score_vectors(evaluation)
        wins, losses = score_comparison_tally(candidate_scores, current_scores)
        ties = len(candidate_scores) - wins - losses
        selected = wins - losses > self._min_win_margin
        metrics: dict[str, Any] = {"wins": wins, "losses": losses, "ties": ties}
        if self._min_win_margin:
            metrics["min_win_margin"] = self._min_win_margin
        return SelectionDecision(
            outcome="select" if selected else "reject",
            policy="score_comparison",
            policy_version="1",
            reason=(
                f"candidate won {wins} task pairings and lost {losses}"
                if selected
                else f"candidate did not exceed its {losses} losses with {wins} wins"
            ),
            evaluation=evaluation,
            metrics=metrics,
        )


class ScoreComparisonPlugin(ScoreComparisonMixin, BackendEvaluateMixin, SelectionPlugin):
    """Cordis's default evaluation: measure through the candidate backend, decide by score comparison."""

    def __init__(self, candidate_backend: Any, *, min_win_margin: int = 0) -> None:
        super().__init__(min_win_margin=min_win_margin)
        self._candidate_backend = candidate_backend

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        unknown = sorted(str(key) for key in config if key != "min_win_margin")
        if unknown:
            raise PluginConfigError(f"the score_comparison selection takes min_win_margin, not {', '.join(unknown)}")
        return cls(candidate_backend, min_win_margin=config.get("min_win_margin", 0))


class FloorMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when every evaluation task scores at least ``floor_score``.

    A floor is absolute, not a comparison: only the candidate's scores are
    read, and an episode that could not run (score ``None``) missed it.
    """

    def __init__(self, *, floor_score: float = 1.0) -> None:
        if (
            isinstance(floor_score, bool)
            or not isinstance(floor_score, (int, float))
            or not math.isfinite(floor_score)
            or floor_score <= 0
        ):
            raise ValueError("floor_score must be a positive number")
        super().__init__()
        self._floor_score = float(floor_score)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        scores = tuple(evaluation.metrics.get("candidate_scores", ()))
        passed = sum(1 for score in scores if score is not None and score >= self._floor_score)
        failed = len(scores) - passed
        if not scores:
            reason = "no evaluation task was scored"
        elif failed:
            reason = f"candidate missed the floor on {failed} of {len(scores)} tasks"
        else:
            reason = f"candidate met the floor on all {len(scores)} tasks"
        return SelectionDecision(
            outcome="select" if scores and not failed else "reject",
            policy="floor",
            policy_version="1",
            reason=reason,
            evaluation=evaluation,
            metrics={"passed": passed, "failed": failed, "floor_score": self._floor_score},
        )


class FloorPlugin(FloorMixin, BackendEvaluateMixin, SelectionPlugin):
    """Evaluate the candidate alone through the candidate backend, decide by the floor; the current release is not run."""

    def __init__(self, candidate_backend: Any, *, floor_score: float = 1.0) -> None:
        super().__init__(floor_score=floor_score)
        self._candidate_backend = candidate_backend

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        return self._candidate_backend.evaluate(candidate, sides=("candidate",))

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        unknown = sorted(str(key) for key in config if key != "floor_score")
        if unknown:
            raise PluginConfigError(f"the floor selection takes floor_score, not {', '.join(unknown)}")
        return cls(candidate_backend, floor_score=config.get("floor_score", 1.0))


def score_vectors(
    evaluation: EvaluationResult,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    candidate = tuple(evaluation.metrics.get("candidate_scores", ()))
    current = tuple(evaluation.metrics.get("current_scores", ()))
    return candidate, current


def score_comparison_tally(candidate: tuple[float | None, ...], current: tuple[float | None, ...]) -> tuple[int, int]:
    wins = losses = 0
    for candidate_score, current_score in zip(candidate, current, strict=True):
        candidate_rank = candidate_score if candidate_score is not None else float("-inf")
        current_rank = current_score if current_score is not None else float("-inf")
        if candidate_rank > current_rank:
            wins += 1
        elif candidate_rank < current_rank:
            losses += 1
    return wins, losses
