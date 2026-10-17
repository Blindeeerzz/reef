"""Run the episodes of one saved task: stage it, play one arm, and keep what each episode returned.

The task store holds the task; the executor holds a staged copy of it. Staging the same task again returns
the staged copy, so a play that a stop interrupted can stage and play again, and a generator service with an
empty tasks root can still play a saved task. Staged copies stay after a play: the generator service checks
new tasks for duplicates against them.

The agent's model calls go through Reef's scenario proxy, so each episode's inference records and its
report reach Reef as records, tagged with the play's labels. The runner plays episodes; how many plays a
task gets, and how they group into training batches, belong to its caller.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.core.tasks import HarborTask
from reef.record2task.generator.client import Generator


@dataclass(frozen=True)
class Episode:
    """One played episode: its reward, the reports that reached Reef, and its error.

    An episode with an error and no reward could not run: the sandbox, the agent or the verifier failed
    before it scored. That is an infrastructure failure, not a failure of the solver.
    """

    episode_id: str
    reward: float | None
    report_record_ids: tuple[str, ...]
    error: str

    @property
    def is_reported(self) -> bool:
        return bool(self.report_record_ids)

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.reward is None and bool(self.error)


class EpisodeRunner:
    """Stage a saved task through the generator service and play its episodes in the service's sandbox."""

    def __init__(self, service: Generator) -> None:
        self.service = service

    async def stage(self, task: HarborTask) -> Path:
        """Stage the task and return its path; staging the same task again returns the same path.

        Raises ``TaskNameConflict`` when a different task holds the name, and ``DuplicateTask`` when the same
        content is staged under another name.
        """
        return (await self.service.write_task(task)).path

    async def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        plays: int,
        extra_instruction_files: Sequence[str],
        labels: Mapping[str, str],
        model: str | None,
    ) -> tuple[Episode, ...]:
        """Play ``plays`` episodes of a staged task on one arm; each episode's records carry ``labels``."""
        played = await self.service.play(
            task_path,
            scenario=scenario,
            arm=arm,
            plays=plays,
            is_reporting=True,
            extra_instruction_files=extra_instruction_files,
            tags=labels,
            model=model,
        )
        return tuple(
            Episode(play.episode_id, play.reward, play.report_agent_record_ids, play.error) for play in played
        )
