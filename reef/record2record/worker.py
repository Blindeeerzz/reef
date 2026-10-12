"""Asynchronous jobs off the trainer thread: a job runner and the worker that runs it."""

from __future__ import annotations

import asyncio
import logging
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock, Thread

logger = logging.getLogger(__name__)

#: How long ``close()`` lets in-flight jobs finish before cancelling.
CLOSE_GRACE_S = 5.0


@dataclass(frozen=True)
class SupportsReceipt:
    """What the worker needs of a job and of its result alike: the receipt they concern.

    The caller defines both types. The worker only reads ``receipt``, so this
    shared value is the base of both types and of :class:`Failed`.
    """

    receipt: str


@dataclass(frozen=True)
class Failed(SupportsReceipt):
    """The worker's result for a job that never finished."""


class JobRunner(ABC):
    """Run one job and return its result. The worker calls ``run`` on its own thread."""

    @abstractmethod
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Return the result for ``job``. An exception resolves the job as :class:`Failed`."""


class JudgingWorker:
    """Jobs off the trainer thread: jobs in, results out, both queues.

    The worker owns only execution — one daemon thread, an event loop, and
    the two queues; what a job *does* stays the runner's ``run`` coroutine,
    handed in at construction. Faults degrade, never hang: a failed or
    cancelled job resolves as :class:`Failed`, and a worker whose event loop
    dies marks itself broken, resolves everything queued as :class:`Failed`,
    and rejects new work so the caller retires records directly.
    """

    def __init__(self, runner: JobRunner, *, concurrency: int = 8) -> None:
        self.runner = runner
        self.concurrency = max(1, concurrency)
        self.jobs: queue.SimpleQueue[SupportsReceipt | None] = queue.SimpleQueue()
        self.results: queue.SimpleQueue[SupportsReceipt] = queue.SimpleQueue()
        self.thread: Thread | None = None
        # Serializes submit against the broken/closed transitions so a job
        # can never slip in after the final drain and leave its receipt
        # unresolved.
        self.gate = Lock()
        self.broken = False
        self.closed = False

    def submit(self, job: SupportsReceipt) -> bool:
        """Queue one job; ``False`` means it will never run."""
        with self.gate:
            if self.closed or self.broken:
                return False
            if self.thread is None:
                self.thread = Thread(target=self.run_thread, name="computed-feedback-judging", daemon=True)
                self.thread.start()
            self.jobs.put(job)
            return True

    def poll(self) -> list[SupportsReceipt]:
        """Drain every finished result without blocking."""
        judgments: list[SupportsReceipt] = []
        try:
            while True:
                judgments.append(self.results.get_nowait())
        except queue.Empty:
            return judgments

    def close(self) -> None:
        """Stop accepting work, give in-flight jobs a bounded grace, join."""
        with self.gate:
            if self.closed:
                return
            self.closed = True
        if self.thread is None:
            return
        self.jobs.put(None)
        self.thread.join(timeout=CLOSE_GRACE_S + 10.0)

    # ------------------------------------------------------------ worker side

    def run_thread(self) -> None:
        try:
            asyncio.run(self.run_jobs())
        except Exception:
            logger.exception("judging worker failed; queued jobs resolve untrainable")
        finally:
            with self.gate:
                self.broken = True
            while True:  # no submit can race this drain: broken is set
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self.results.put(Failed(job.receipt))

    async def run_jobs(self) -> None:
        # At most ``concurrency`` jobs run at once; the semaphore lives on
        # this loop, like the rest of the worker's state.
        semaphore = asyncio.Semaphore(self.concurrency)
        loop = asyncio.get_running_loop()
        tasks: set[asyncio.Task] = set()
        while True:
            job = await asyncio.to_thread(self.jobs.get)
            if job is None:
                break
            task = loop.create_task(self.run_one(job, semaphore))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            # Whatever the grace period does not finish, asyncio.run cancels
            # on the way out — and each job still resolves its receipt
            # from its finally.
            await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)

    async def run_one(self, job: SupportsReceipt, semaphore: asyncio.Semaphore) -> None:
        resolved = False
        try:
            async with semaphore:
                self.results.put(await self.runner.run(job))
            resolved = True
        except Exception:
            logger.exception("judgment for receipt %s failed; record resolves untrainable", job.receipt)
        finally:
            if not resolved:  # exception or teardown cancellation
                self.results.put(Failed(job.receipt))
