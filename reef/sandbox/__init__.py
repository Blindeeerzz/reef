"""Shared process execution and isolation through Harbor's environment contract.

``environment`` adapts the existing local, bubblewrap and E2B executors to
``harbor.environments.base.BaseEnvironment``. ``process`` owns local execution,
resource limits and process cleanup. ``e2b`` owns cloud sessions, file transfer
and gateway tunnels. Configuration policy, harness rendering and evaluation
remain with their callers. This package depends on core values and Harbor,
never on harness, recipes or training.
"""
