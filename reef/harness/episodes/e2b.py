"""Harness runtime selection and compatibility imports for E2B execution."""

from __future__ import annotations

from dataclasses import replace

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.sandbox.e2b import RELAY_SCRIPT, SANDBOX_MAX_TIMEOUT_S
from reef.sandbox.e2b import E2BExecutor as CloudExecutor
from reef.sandbox.e2b import (
    E2BSession,
    SandboxKeeper,
    TunnelPump,
    deployment_owner,
    ensure_template,
    pack,
    protected_command,
    remote_command,
    remote_root,
    template_alias,
    unpack,
)
from reef.sandbox.process import SandboxUnavailable


class E2BExecutor(CloudExecutor):
    """Keep adapter selection at the harness boundary."""

    def for_adapter(self, descriptor: AdapterDescriptor, *, binary: str | None = None) -> E2BExecutor:
        if descriptor.self_isolating:
            raise SandboxUnavailable(
                f"adapter {descriptor.name!r} manages its own containers; use its hosted configuration"
            )
        remote_binary = binary or self.binary or descriptor.binary
        if self.template:
            if remote_binary == self.binary:
                return self
            return replace(self, binary=remote_binary)
        install = descriptor.install
        if install is None or install.kind != "npm":
            raise SandboxUnavailable(
                f"E2B adapter {descriptor.name!r} requires an explicit e2b_template containing its binary and bubblewrap"
            )
        return replace(
            self,
            template=template_alias(descriptor.name, install.version),
            npm_package=f"{install.package}@{install.version}",
            binary=remote_binary,
        )


__all__ = [
    "RELAY_SCRIPT",
    "SANDBOX_MAX_TIMEOUT_S",
    "E2BExecutor",
    "E2BSession",
    "SandboxKeeper",
    "TunnelPump",
    "deployment_owner",
    "ensure_template",
    "pack",
    "protected_command",
    "remote_command",
    "remote_root",
    "template_alias",
    "unpack",
]
