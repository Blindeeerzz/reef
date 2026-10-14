Sandbox environments
====================

Use ``reef.sandbox.environment.ExecutorEnvironment`` to run commands and move
files through Harbor's ``BaseEnvironment`` interface. It supports Reef's local,
bubblewrap and E2B execution implementations. Harness rendering and candidate
evaluation remain outside this package.


Run a command
-------------

Install Reef and create the root and workspace before constructing an
environment. This local example returns ``hello`` and exit code zero::

    import asyncio
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from harbor.environments.base import BaseEnvironment
    from reef.sandbox.environment import ExecutorEnvironment
    from reef.sandbox.process import LocalExecutor

    async def example():
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            environment: BaseEnvironment = ExecutorEnvironment(
                LocalExecutor(), root=root, workspace=workspace
            )
            try:
                await environment.start(force_build=False)
                result = await environment.exec("printf hello")
                print(result.stdout, result.return_code)
            finally:
                await environment.stop(delete=True)

    asyncio.run(example())

``exec`` accepts a shell command, a working directory inside the root,
environment variables and a timeout in seconds. It returns Harbor's
``ExecResult`` with ``stdout``, ``stderr`` and ``return_code``. A nonzero command
exit is a result. Launch failures raise ``EpisodeLaunchError``; timeouts raise
``EpisodeTimeout``. Commands run as the executor's configured user; selecting
another ``user`` is unsupported.


Files and lifecycle
-------------------

Upload and download files or directories with Harbor's corresponding methods.
Environment paths refer to the local staged root, including with E2B. Relative
paths start at the workspace. Transfers reject paths outside the root.
E2B downloads refresh the staged copy before copying it to the destination.
Upload files before starting commands that use them.

The caller owns the staged root. ``stop`` terminates the environment's local
processes but does not delete caller-owned files. An ``E2BExecutor`` creates an
owned VM; ``stop(delete=True)`` destroys it. Passing an existing ``E2BSession``
borrows that VM for a trial. Stopping the trial's environment leaves the shared
VM running; its owner must close it after the proposer finishes.

Cancelling a command terminates its environment's processes before returning.
For E2B, cancellation retires the VM, including a borrowed session. Ordinary
timeouts preserve the executor's file-copy behavior. Transfer and cleanup
failures propagate. Retaining a stopped owned E2B VM and forcing template
rebuilds are unsupported.

Local execution provides no sandbox isolation. Bubblewrap and E2B retain their
existing read-only inputs, writable state and network behavior. Configuration
remains independent for evaluation and proposer execution; see
`Running episodes <../user-guide/evolve-your-harness.rst#running-episodes>`__.

``execute`` and ``run_sync`` bridge existing synchronous callers to this async
interface. Use the async methods directly inside a running event loop. Imports
from ``reef.harness.episodes.executor`` and ``reef.harness.episodes.e2b`` remain
available for compatibility. Adapter-specific binary selection stays in the
harness layer.
