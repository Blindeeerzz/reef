"""Episode executor configuration and compatibility imports for ``reef.sandbox``."""

from __future__ import annotations

import os
from collections.abc import Mapping

from reef.core.errors import ReefError
from reef.sandbox.process import (
    EPISODE_OWNER_LEASE,
    ISOLATION_ENV,
    EpisodeExecutor,
    EpisodeLaunchError,
    EpisodeTimeout,
    LocalExecutor,
    ProcessOutcome,
    SandboxExecutor,
    SandboxLimits,
    SandboxUnavailable,
)

__all__ = [
    "EPISODE_OWNER_LEASE",
    "ISOLATION_ENV",
    "EpisodeExecutor",
    "EpisodeLaunchError",
    "EpisodeTimeout",
    "LocalExecutor",
    "ProcessOutcome",
    "SandboxExecutor",
    "SandboxLimits",
    "SandboxUnavailable",
    "build_executor",
]


def build_executor(
    config: Mapping[str, object] | None, *, environ: Mapping[str, str] | None = None
) -> EpisodeExecutor:
    """Build the executor a deployment selected.

    ``executor`` is ``local`` (default), ``sandbox`` or ``e2b``; a sandbox reads its
    ``egress_hosts``, ``limits``, and an explicit ``env_from`` list of host
    variable names from the same section. The executor is
    preflighted at build so a hosted deployment that requires the sandbox
    fails to start, not at the first episode.
    """
    config = config or {}
    kind = config.get("executor", "local")
    if kind == "local":
        return LocalExecutor()
    if kind == "e2b":
        from reef.harness.adapters import get_adapter
        from reef.harness.episodes.e2b import E2BExecutor

        section = config.get("sandbox") or {}
        if not isinstance(section, Mapping):
            raise ReefError("evolution.sandbox must be a mapping")
        adapter = config.get("adapter", "pi")
        binary = config.get("binary")
        if not isinstance(adapter, str) or (binary is not None and not isinstance(binary, str)):
            raise ReefError("evolution.adapter and evolution.binary must be strings")
        remote = E2BExecutor.from_config(section, os.environ if environ is None else environ)
        remote = remote.for_adapter(get_adapter(adapter), binary=binary)
        remote.preflight()
        return remote
    if kind != "sandbox":
        raise ReefError(f"unknown episode executor {kind!r}; use 'local', 'sandbox' or 'e2b'")
    sandbox_config = config.get("sandbox") or {}
    if not isinstance(sandbox_config, Mapping):
        raise ReefError("evolution.sandbox must be a mapping")
    egress = tuple(str(host) for host in (sandbox_config.get("egress_hosts") or ()))
    limits_config = sandbox_config.get("limits") or {}
    if not isinstance(limits_config, Mapping):
        raise ReefError("evolution.sandbox.limits must be a mapping")
    limits = SandboxLimits(
        cpu_seconds=int(limits_config.get("cpu_seconds", 0)),
        memory_bytes=int(limits_config.get("memory_bytes", 0)),
        processes=int(limits_config.get("processes", 0)),
        file_bytes=int(limits_config.get("file_bytes", 0)),
    )
    names = sandbox_config.get("env_from", [])
    if not isinstance(names, (list, tuple)) or not all(
        isinstance(name, str) and name.isidentifier() for name in names
    ):
        raise ReefError("evolution.sandbox.env_from must be a list of environment variable names")
    values = os.environ if environ is None else environ
    env = {}
    for name in names:
        if name not in values or not values[name]:
            raise ReefError(f"evolution.sandbox.env_from requires environment variable {name!r}")
        env[name] = values[name]
    executor = SandboxExecutor(egress_hosts=egress, limits=limits, env=env)
    executor.preflight()
    return executor
