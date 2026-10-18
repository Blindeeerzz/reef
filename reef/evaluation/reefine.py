"""Independent Reefine request, protected-task and model review checks.

This plugin alone selects a candidate. Proposer design notes and trial scores
are not evaluation inputs. Each model call starts with a fresh message list.
The plan reads the harness description, so requested behavior the harness
cannot deliver is reported as a limit instead of being run and failed; a
candidate that passes with limits waits for the person's review.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from reef.core.evaluation import (
    CandidateEvaluator,
    EvaluationResult,
    SelectionDecision,
    ServedComposition,
    UpdateCandidate,
)
from reef.core.plugins import ConfiguredPlugin, PluginConfigError
from reef.core.prompts import untrusted_text
from reef.core.training_request import TrainingRequest
from reef.evaluation.evaluators import SelectionPlugin
from reef.evaluation.harness import HarnessCandidate, verifier_reward
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.harness.episodes.run import EpisodeError, EpisodeResult, TrajectoryKeepError, run_episode
from reef.harness.episodes.trajectory import TrajectoryError, final_assistant_text
from reef.harness.step_result import evaluation_report
from reef.sandbox.process import EpisodeExecutor


@dataclass(frozen=True)
class EvaluationContext:
    request: TrainingRequest | None
    current: ServedComposition


@dataclass(frozen=True, kw_only=True)
class ReefineCandidate(HarnessCandidate):
    context: EvaluationContext
    requires: tuple[Mapping[str, object], ...] = ()


class ReefineEvaluationBackend(CandidateEvaluator):
    """Episode inputs and progress reporting for independent Reefine evaluation."""

    episode_timeout_seconds: float

    @property
    @abstractmethod
    def evaluation_models(self) -> ModelBindings: ...

    @property
    @abstractmethod
    def episode_descriptor(self) -> AdapterDescriptor: ...

    @property
    @abstractmethod
    def episode_binary(self) -> str: ...

    @property
    @abstractmethod
    def episode_executor(self) -> EpisodeExecutor: ...

    @property
    @abstractmethod
    def harness_description(self) -> Mapping[str, object]:
        """What a change to the evaluated harness may contain and what the harness offers, for the plan."""

    @abstractmethod
    def episode_files(self, entries: tuple[Mapping[str, object], ...]) -> dict[str, str]: ...

    @abstractmethod
    def show_evaluation_checks(self, checks: tuple[Mapping[str, object], ...]) -> None: ...


CheckStatus = Literal["pending", "running", "pass", "fail", "invalid", "not_run", "unsupported"]


@dataclass(frozen=True)
class CheckResult:
    id: str
    group: str
    target: str
    status: CheckStatus = "pending"
    expected: str = ""
    observed: str = ""
    reason: str = ""
    score: float | None = None
    current_score: float | None = None
    seconds: float = 0.0
    record: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool) or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("request verification requires a boolean passed and a nonempty reason")


class RequestVerifier(ConfiguredPlugin, ABC):
    """Optional operator-owned validation, outside the candidate's writable workspace."""

    @abstractmethod
    def verify(self, result: EpisodeResult, workspace: Path) -> VerificationResult: ...


@dataclass(frozen=True)
class RequestContext:
    prompt: str = ""
    files: Mapping[str, str] | None = None
    initialize_git: bool = False
    verifier: RequestVerifier | None = None
    version: str = "1"


@dataclass(frozen=True)
class EvaluationInstructions:
    plan: str
    behavior_review: str
    change_review: str


@dataclass(frozen=True)
class ReefineEvaluationSettings:
    reviewer_model: str = "served"
    request_context: RequestContext = RequestContext()
    protected_tasks: tuple[str, ...] = ()
    timeout_seconds: float = 900.0
    max_output_tokens: int = 8192
    scorer_version: str = "1"
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("evaluation.timeout_s must be positive and finite")
        if self.max_output_tokens < 256:
            raise ValueError("evaluation.max_output_tokens must be at least 256")


class ReefineEvaluation(SelectionPlugin):
    def __init__(
        self,
        backend: ReefineEvaluationBackend,
        settings: ReefineEvaluationSettings,
        instructions: EvaluationInstructions,
    ):
        self.backend = backend
        self.settings = settings
        self.instructions = instructions

    @classmethod
    def from_config(cls, config: Mapping[str, object], *, candidate_backend: CandidateEvaluator) -> ReefineEvaluation:
        if not isinstance(candidate_backend, ReefineEvaluationBackend):
            raise PluginConfigError("Reefine evaluation requires a ReefineEvaluationBackend")
        settings = config.get("settings")
        instructions = config.get("instructions")
        if (
            set(config) != {"settings", "instructions"}
            or not isinstance(settings, ReefineEvaluationSettings)
            or not isinstance(instructions, EvaluationInstructions)
        ):
            raise PluginConfigError("Reefine evaluation requires parsed settings and instructions")
        return cls(candidate_backend, settings, instructions)

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, ReefineCandidate):
            raise TypeError("Reefine evaluation requires a ReefineCandidate")
        started = time.monotonic()
        deadline = started + self.settings.timeout_seconds
        models = self.backend.evaluation_models
        reviewer = models[self.settings.reviewer_model]
        checks = [
            CheckResult("request-plan", "request", "plan", expected="a real episode covering the original request"),
            CheckResult("request-behavior", "request", "candidate", expected="all requested behavior"),
        ]
        for index, task in enumerate(self.settings.protected_tasks):
            checks.append(CheckResult(f"regression-{index}", "regression", "candidate/current", expected=task))
        if not self.settings.protected_tasks:
            checks.append(
                CheckResult(
                    "regression-missing",
                    "regression",
                    "candidate/current",
                    "not_run",
                    reason="no protected tasks configured",
                )
            )
        checks.append(
            CheckResult(
                "change-review", "review", "candidate", expected="request implemented; no unsafe or unrelated change"
            )
        )
        self.update_progress(checks)
        root = candidate.record_dir
        with TemporaryDirectory(prefix="reefine-evaluation-") as temporary:
            records = Path(temporary) if root is None else root / "reefine-evaluation"
            records.mkdir(parents=True, exist_ok=True)
            # Plan from the original instruction, operator context and the harness description, before viewing any
            # candidate content.
            request = candidate.context.request
            plan_index = 0
            plan = None
            if request is None:
                checks[plan_index] = replace(
                    checks[plan_index], status="invalid", reason="no original training request"
                )
            else:
                self.begin(checks, plan_index)
                plan_started = time.monotonic()
                try:
                    context = self.settings.request_context
                    reply = self.ask_json(
                        reviewer,
                        self.instructions.plan,
                        {
                            "request": request.text,
                            "prompt": context.prompt,
                            "files": dict(context.files or {}),
                            "harness": dict(self.backend.harness_description),
                        },
                        deadline,
                    )
                    prompt, requirements, limits = validate_plan(reply)
                    if context.prompt and requirements:
                        prompt = context.prompt
                    plan = {"prompt": prompt, "checks": requirements, "limits": limits}
                    checks[plan_index] = replace(
                        checks[plan_index],
                        status="pass",
                        observed="; ".join(requirements),
                        seconds=time.monotonic() - plan_started,
                    )
                    # Behavior the harness cannot deliver is reported, never run: running it could only fail.
                    checks[plan_index + 1 : plan_index + 1] = [
                        CheckResult(
                            f"limit-{index}",
                            "limit",
                            "harness",
                            "unsupported",
                            expected=limit["requirement"],
                            reason=limit["reason"],
                        )
                        for index, limit in enumerate(limits)
                    ]
                except (ValueError, ModelBindingError) as exc:
                    checks[plan_index] = replace(
                        checks[plan_index], status="invalid", reason=str(exc), seconds=time.monotonic() - plan_started
                    )
            self.update_progress(checks)
            request_index = next(i for i, check in enumerate(checks) if check.id == "request-behavior")
            if plan is None or request is None:
                checks[request_index] = replace(
                    checks[request_index], status="not_run", reason="request plan unavailable"
                )
            elif not plan["checks"]:
                checks[request_index] = replace(
                    checks[request_index],
                    status="not_run",
                    reason="this harness cannot deliver any of the requested behavior; see its limits",
                )
            else:
                checks[request_index] = replace(checks[request_index], expected=checks[plan_index].observed)
                self.begin(checks, request_index)
                before = time.monotonic()
                record_dir = records / "request"
                try:
                    context = self.settings.request_context
                    result = self.episode(
                        candidate,
                        "candidate",
                        str(plan["prompt"]),
                        context.files or {},
                        context.initialize_git,
                        record_dir,
                        deadline,
                    )
                    passed, reason = self.judge_episode(
                        reviewer, request.text, plan, result, candidate.candidate_files, candidate.requires, deadline
                    )
                    if context.verifier is not None:
                        verified = context.verifier.verify(result, record_dir / "workspace")
                        passed = passed and verified.passed
                        reason = reason + "; " + verified.reason
                    checks[request_index] = replace(
                        checks[request_index],
                        status="pass" if passed else "fail",
                        score=float(passed),
                        observed=reason,
                        reason="" if passed else reason,
                        seconds=time.monotonic() - before,
                        record=str(record_dir) if root else None,
                    )
                except (
                    EpisodeError,
                    TrajectoryError,
                    TrajectoryKeepError,
                    ValueError,
                    ModelBindingError,
                    OSError,
                ) as exc:
                    checks[request_index] = replace(
                        checks[request_index], status="invalid", reason=str(exc), seconds=time.monotonic() - before
                    )
            self.update_progress(checks)
            for index, task in enumerate(self.settings.protected_tasks):
                check_index = next(i for i, check in enumerate(checks) if check.id == f"regression-{index}")
                self.begin(checks, check_index)
                before = time.monotonic()
                scores = {}
                try:
                    # Matched prompt, workspace, model and budget; fresh sessions on both sides.
                    for side in ("current", "candidate"):
                        checks[check_index] = replace(checks[check_index], observed=f"running {side} episode")
                        self.update_progress(checks)
                        result = self.episode(
                            candidate,
                            side,
                            task,
                            self.settings.request_context.files or {},
                            False,
                            records / f"regression-{index}-{side}",
                            deadline,
                        )
                        passed, reason = self.judge_episode(
                            reviewer,
                            task,
                            {"prompt": task, "checks": [task]},
                            result,
                            candidate.current_files if side == "current" else candidate.candidate_files,
                            (),
                            deadline,
                        )
                        scores[side] = float(passed)
                    passed = scores["candidate"] >= scores["current"] and scores["candidate"] == 1.0
                    checks[check_index] = replace(
                        checks[check_index],
                        status="pass" if passed else "fail",
                        score=scores["candidate"],
                        current_score=scores["current"],
                        observed=f"current {scores['current']:g}; candidate {scores['candidate']:g}",
                        reason="" if passed else "protected capability failed or regressed",
                        seconds=time.monotonic() - before,
                    )
                except (EpisodeError, TrajectoryError, TrajectoryKeepError, ValueError, ModelBindingError) as exc:
                    checks[check_index] = replace(
                        checks[check_index], status="invalid", reason=str(exc), seconds=time.monotonic() - before
                    )
                self.update_progress(checks)
            review_index = len(checks) - 1
            self.begin(checks, review_index)
            before = time.monotonic()
            try:
                change_review = self.ask_json(
                    reviewer,
                    self.instructions.change_review,
                    {
                        "request": "" if request is None else request.text,
                        "current": candidate.current_files,
                        "candidate": candidate.candidate_files,
                        "requires": list(candidate.requires),
                        "checks": [check.to_dict() for check in checks[:-1]],
                    },
                    deadline,
                )
                passed, reason = validate_review(change_review)
                checks[review_index] = replace(
                    checks[review_index],
                    status="pass" if passed else "fail",
                    score=float(passed),
                    observed=reason,
                    reason="" if passed else reason,
                    seconds=time.monotonic() - before,
                )
            except (ValueError, ModelBindingError) as exc:
                checks[review_index] = replace(
                    checks[review_index], status="invalid", reason=str(exc), seconds=time.monotonic() - before
                )
            self.update_progress(checks)
            same_model = reviewer.model == models.served.model
            report = {
                "version": "1",
                "current_release_id": candidate.context.current.release_id,
                "request_id": None if request is None else request.id,
                "model": models.served.model,
                "reviewer_model": reviewer.model,
                "reviewer_binding": self.settings.reviewer_model,
                "reviewer_context": "fresh",
                "same_model": same_model,
                "adapter": self.backend.episode_descriptor.name,
                "requires": list(candidate.requires),
                "task_version": self.settings.request_context.version,
                "scorer_version": self.settings.scorer_version,
                "protected_tasks_checksum": hashlib.sha256(
                    json.dumps(self.settings.protected_tasks).encode()
                ).hexdigest(),
                "plan": plan,
                "checks": [check.to_dict() for check in checks],
                "seconds": round(time.monotonic() - started, 3),
            }
            if root is not None:
                (records / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return EvaluationResult("reefine", "1", {"evaluation_report": report, "evaluation_sides": ["candidate"]})

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        report = evaluation_report(evaluation.metrics)
        checks = report.get("checks") if isinstance(report, Mapping) else None
        failures = []
        limits = []
        if not isinstance(checks, list) or not checks:
            failures.append("evaluation returned no checks")
        else:
            groups = {check.get("group") for check in checks if isinstance(check, Mapping)}
            if not {"request", "regression", "review"} <= groups or groups - {
                "request",
                "regression",
                "review",
                "limit",
            }:
                failures.append("evaluation is missing a required check group")
            limits = [check for check in checks if isinstance(check, Mapping) and check.get("group") == "limit"]
            if isinstance(candidate, ReefineCandidate):
                expected_ids = {"request-plan", "request-behavior", "change-review"}
                expected_ids.update(f"regression-{index}" for index in range(len(self.settings.protected_tasks)))
                expected_ids.update(f"limit-{index}" for index in range(len(limits)))
                ids = [check.get("id") for check in checks if isinstance(check, Mapping)]
                if set(ids) != expected_ids or len(ids) != len(expected_ids):
                    failures.append("evaluation is missing required checks or contains duplicate checks")
                request_id = None if candidate.context.request is None else candidate.context.request.id
                if isinstance(report, Mapping) and (
                    report.get("request_id") != request_id
                    or report.get("current_release_id") != candidate.context.current.release_id
                ):
                    failures.append("evaluation does not match this request and serving release")
            failures.extend(
                (
                    str(check.get("id", "unknown")) + ": " + str(check.get("reason") or check.get("status"))
                    if isinstance(check, Mapping)
                    else "malformed check"
                )
                for check in checks
                if not isinstance(check, Mapping)
                or check.get("status") != ("unsupported" if check.get("group") == "limit" else "pass")
            )
        selected = not failures
        if not selected:
            reason = "; ".join(failures)
        elif limits:
            reason = (
                "required Reefine checks passed; waiting for your review because this harness cannot deliver: "
                + ("; ".join(str(check.get("expected")) for check in limits))
            )
        else:
            reason = "all required Reefine checks passed"
        feedback = []
        if isinstance(checks, list):
            feedback = [
                {
                    "id": check.get("id"),
                    "status": check.get("status"),
                    "reason": check.get("reason") or check.get("observed"),
                }
                for check in checks
                if isinstance(check, Mapping)
                and check.get("group") in ("request", "limit")
                and check.get("status") != "pass"
            ]
        return SelectionDecision(
            selected=selected,
            policy="reefine",
            reason=reason,
            evaluation=evaluation,
            metrics={"request_feedback": feedback},
            requires_review=selected and bool(limits),
        )

    def begin(self, checks: list[CheckResult], index: int) -> None:
        checks[index] = replace(checks[index], status="running")
        self.update_progress(checks)

    def update_progress(self, checks: list[CheckResult]) -> None:
        self.backend.show_evaluation_checks(tuple(check.to_dict() for check in checks))

    def ask_json(
        self, model: ModelBinding, instructions: str, payload: Mapping[str, object], deadline: float
    ) -> dict[str, object]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("evaluation time budget exhausted")
        # No proposer messages, design, review notes or trial results are carried into this conversation.
        messages = [
            {
                "role": "user",
                "content": instructions + "\n" + untrusted_text(json.dumps(payload), "evaluation inputs"),
            }
        ]
        # Retry transient transport/server failures once, within the original evaluation deadline.
        for attempt in range(2):
            try:
                reply = model.chat(messages, timeout_s=remaining, max_tokens=self.settings.max_output_tokens)
                break
            except ModelBindingError as exc:
                if attempt == 1 or (exc.status is not None and exc.status not in {408, 429, 500, 502, 503, 504}):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("evaluation time budget exhausted") from exc
        if reply.strip().startswith("```"):
            reply = "\n".join(reply.strip().splitlines()[1:-1])
        parsed = json.loads(reply)
        if not isinstance(parsed, dict):
            raise ValueError("reviewer must return a JSON object")
        return parsed

    def episode(
        self,
        candidate: ReefineCandidate,
        side: str,
        prompt: str,
        files: Mapping[str, str],
        initialize_git: bool,
        record_dir: Path,
        deadline: float,
    ) -> EpisodeResult:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("evaluation time budget exhausted")
        entries = candidate.candidate_entries if side == "candidate" else candidate.current_entries
        # Real tasks run tools that cache under the episode's home; the root is removed after the run, so the review
        # reads what was written outside the workspace instead of the run being refused for it.
        return run_episode(
            self.backend.episode_descriptor,
            self.backend.episode_files(entries),
            prompt,
            binary=self.backend.episode_binary,
            timeout=min(remaining, self.backend.episode_timeout_seconds),
            executor=self.backend.episode_executor,
            keep_dir=record_dir,
            workspace_files=files,
            initialize_git=initialize_git,
            keep_workspace=True,
            online=not prompt.startswith("[health]"),
            task_environment=self.settings.environment,
        )

    def judge_episode(
        self,
        model: ModelBinding,
        request: str,
        plan: Mapping[str, object],
        result: EpisodeResult,
        harness_files: Mapping[str, str],
        requires: tuple[Mapping[str, object], ...],
        deadline: float,
    ) -> tuple[bool, str]:
        if result.exit_code != 0:
            return False, f"harness exited {result.exit_code}: {result.stderr[-500:]}"
        if not result.trajectory:
            raise ValueError("episode produced no trajectory")
        review = self.ask_json(
            model,
            self.instructions.behavior_review,
            {
                "request": request,
                "plan": plan,
                "harness": harness_files,
                "requires": list(requires),
                "trajectory": result.trajectory,
                "stderr": result.stderr[-2000:],
                "files_outside_workspace": list(result.residue[:50]),
            },
            deadline,
        )
        return validate_review(review)


HEALTH_TASK_DIRECTORY = str(Path(__file__).parents[1] / "recipe" / "reefine" / "health")


def evaluate_health(task: str, result: EpisodeResult) -> float:
    """Score the bundled health prompt or its Harbor task directory."""
    if task == HEALTH_TASK_DIRECTORY:
        return verifier_reward(task, result)
    return grade_health_text(task, final_assistant_text(result.trajectory))


def grade_health_text(task: str, text: str | None) -> float:
    """Match reef-ok on the final nonempty line of a health-task response."""
    if not task.startswith("[health]") or text is None:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return 1.0 if lines and lines[-1] == "reef-ok" else 0.0


def validate_plan(value: Mapping[str, object]) -> tuple[str, list[str], list[dict[str, str]]]:
    prompt = value.get("prompt")
    checks = value.get("checks")
    limits = value.get("limits", [])
    if (
        not isinstance(checks, list)
        or len(checks) > 20
        or any(not isinstance(check, str) or not check.strip() for check in checks)
    ):
        raise ValueError("request plan needs at most 20 observable checks")
    if (
        not isinstance(limits, list)
        or len(limits) > 10
        or any(
            not isinstance(limit, Mapping)
            or set(limit) != {"requirement", "reason"}
            or any(not isinstance(text, str) or not text.strip() for text in limit.values())
            for limit in limits
        )
    ):
        raise ValueError("request plan limits must be at most 10 objects with a requirement and a reason")
    if not checks and not limits:
        raise ValueError("request plan needs at least one check or limit")
    if checks and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000):
        raise ValueError("request plan needs a nonempty episode prompt of at most 4000 characters")
    return (
        prompt if isinstance(prompt, str) else "",
        checks,
        [{"requirement": limit["requirement"], "reason": limit["reason"]} for limit in limits],
    )


def validate_review(value: Mapping[str, object]) -> tuple[bool, str]:
    passed = value.get("passed")
    reason = value.get("reason")
    if not isinstance(passed, bool) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("review requires a boolean passed and a nonempty reason")
    return passed, reason[:4000]
