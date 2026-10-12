"""The OpenClaw-RL processor: one judged turn, one sample.

A computed-feedback recipe on the shared engine
(:class:`~reef.train.processors.computed.ComputedFeedbackProcessor`). The
correlation and the judge are :class:`~recipes.openclawrl.judging.OpenClawRLJudging`,
which runs in the record-to-record stage. This file validates the tensors
(``make_sample``), shapes the batch (``make_batch``), and writes the PRM
record file. Rewards pass through raw (upstream ``--n-samples-per-prompt 1``;
no grouping, no normalization).
"""

from __future__ import annotations

import collections
import json
import logging
import pathlib
from dataclasses import asdict, replace

from recipes.openclawrl.judging import OpenClawRLJudging
from recipes.openclawrl.turns import TurnJudgment, validate_teacher_cands
from reef.core.records_types import AgentRecord, RequestType
from reef.core.trajectories import source_record_id, trajectory_reward
from reef.record2record import JudgingStage
from reef.train.processors.common import make_policy_trajectory
from reef.train.processors.computed import ComputedFeedbackProcessor
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem, policy_row_violation

logger = logging.getLogger(__name__)


class OpenClawRLProcessor(ComputedFeedbackProcessor):
    """Judged turns into samples; a turn whose tensors the top-K select loss cannot train on retires.

    The capture must cover every response token, and every trained sample
    carries at least one teacher candidate.
    """

    output_schema = TrainingBatch
    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    def __init__(self, context: ProcessorContext, *, record_stage: JudgingStage) -> None:
        # What the judge actually saw goes to the PRM record file, one line
        # per batch. Upstream persists this per turn; without it a batch mean
        # is the only visible number. A file, not a log line: reef configures
        # no logging, so anything below WARNING is discarded.
        record_file = str(context.config.get("prm_record_file", "") or "").strip()
        self._record_path = pathlib.Path(record_file) if record_file else None
        super().__init__(context, record_stage=record_stage)
        if not isinstance(self.judging, OpenClawRLJudging):
            raise TypeError("OpenClawRLProcessor needs a stage with OpenClawRLJudging")

    @property
    def openclawrl_judging(self) -> OpenClawRLJudging:
        judging = self.judging
        if not isinstance(judging, OpenClawRLJudging):
            raise TypeError("OpenClawRLProcessor needs OpenClawRLJudging")
        return judging

    def make_sample(self, record: AgentRecord, judgment: TurnJudgment) -> TrajectoryItem | None:
        # 1. A declined judgment never trains. (Narrow on the score itself:
        #    it is what step 2 needs, and a property would not narrow it.)
        if judgment.score is None:
            return None
        # 2. The policy tensors must satisfy the bridge contract, and the
        #    top-K capture must cover every response token.
        sample = make_policy_trajectory(record, judgment.score)
        if (
            policy_row_violation(
                sample.training.get("tokens", []),
                sample.training.get("loss_mask", []),
                sample.training.get("rollout_log_probs", []),
            )
            is not None
        ):
            return None
        if not sample.training.get("topk_indices", []) or len(sample.training.get("topk_indices", [])) != len(
            sample.training.get("loss_mask", [])
        ):
            return None
        # 3. Every trained sample carries at least one teacher candidate —
        #    upstream constructs one per sample by design (RL-only turns get
        #    the anchor), so a candidate-less judgment here means candidate
        #    construction failed and the turn retires instead of training a
        #    shape the reference never produces.
        if not judgment.teacher_cands:
            return None
        validated = validate_teacher_cands(judgment.teacher_cands, sample)
        if validated is None:
            return None
        return sample.with_training(extras={**sample.training.get("extras", {}), "teacher_cands": list(validated)})

    def make_batch(self, samples: tuple[TrajectoryItem, ...], batch_number: int) -> TrainingBatch:
        self._record_batch(samples, batch_number)
        return TrainingBatch(
            f"{self.scenario}:openclawrl:{batch_number}",
            tuple(replace(sample, source_agent_record_ids=(source_record_id(sample),)) for sample in samples),
        )

    def _record_batch(self, samples: tuple[TrajectoryItem, ...], batch_number: int) -> None:
        """Append this batch's judged population to the PRM record file.

        One line per batch: what the judges returned since the previous one,
        the reward mix that reached the trainer, and how correlation is doing
        overall. A batch mean alone cannot distinguish "the judges disagree"
        from "the judges never ran", and neither can it show which next
        states are being judged.
        """
        record = {
            "batch": batch_number,
            "scenario": self.scenario,
            "samples": len(samples),
            "rewards": dict(sorted(collections.Counter(f"{trajectory_reward(s):+.0f}" for s in samples).items())),
            "judged": dict(sorted(self.openclawrl_judging.judged.items())),
            # Cumulative, unlike the per-batch counters above: these answer
            # "is correlation working" for the run as a whole.
            "correlation": asdict(self.openclawrl_judging.index.stats),
        }
        self.openclawrl_judging.judged.clear()
        if self._record_path is None:
            return
        try:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            with self._record_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError:
            # Diagnostics must never take a training step down with them.
            logger.warning("openclawrl: could not append to %s", self._record_path, exc_info=True)
