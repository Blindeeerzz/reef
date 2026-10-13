"""AutoEnvScaling (preprint forthcoming): a coding agent rebuilds recorded sessions as Harbor tasks, a task generator.

The author is a coding agent: it writes a Harbor task in a workspace from one recorded coding-agent
session and the repository the session changed, runs the task's oracle and nop check itself through a
tool, and revises until the check passes. The generator service's one-reply prompt Designer is the
baseline author on the same sessions. A pass-rate band marks the tasks the solver can reach. The built-in
generated-task recipe plays the tasks and trains on them.

- ``sources``: the sources file (session records and their repositories) and the session renderer.
- ``author``: the author contract and its two implementations: the agent, pi in an E2B sandbox behind a
  gateway that records its model calls on Reef, and the one-reply prompt Designer.
"""

from recipes.beta.autoenvscaling.author import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    SourceError,
    TaskAuthor,
    workspace_reply,
)
from recipes.beta.autoenvscaling.sources import Repository, SessionSource, load_sources, render_session

__all__ = [
    "AgentAuthor",
    "AuthorError",
    "AuthoredTask",
    "CandidateChecks",
    "PromptAuthor",
    "Repository",
    "SessionSource",
    "SourceError",
    "TaskAuthor",
    "load_sources",
    "render_session",
    "workspace_reply",
]
