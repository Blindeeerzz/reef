"""Task identity and arm labels shared by generation and training stages."""

from collections.abc import Mapping

from reef.core.records_types import AgentRecord

TRAINING_ARM = "plain"
#: The label that names the rollout batch of an episode.
BATCH_LABEL = "batch"
#: The label that names the task use of an episode within its rollout batch.
TASK_USE_LABEL = "task_use"


def reported_task_name(report: AgentRecord) -> str | None:
    """Read the task player's task name from a report."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


def reported_arm(report: AgentRecord) -> str | None:
    """Read the episode's arm label from a task-player report."""
    return reported_label(report, "arm")


def reported_label(report: AgentRecord, name: str) -> str | None:
    """Read one of the episode's labels from a task-player report."""
    labels = reported_episode(report).get("labels")
    label = labels.get(name) if isinstance(labels, Mapping) else None
    return label if isinstance(label, str) else None


def reported_episode_id(report: AgentRecord) -> str | None:
    """Read the episode ID from a task-player report; several reports of one episode share it."""
    episode_id = reported_episode(report).get("id")
    return episode_id if isinstance(episode_id, str) and episode_id else None


def reported_episode(report: AgentRecord) -> Mapping[str, object]:
    metadata = report.payload.get("metadata")
    episode = metadata.get("episode") if isinstance(metadata, Mapping) else None
    return episode if isinstance(episode, Mapping) else {}
