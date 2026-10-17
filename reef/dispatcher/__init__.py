"""The process-wide coordinator that owns every scenario and its training threads.

The dispatcher delegates scenario table management (creation, resolution,
per-scenario locking) to :class:`reef.scenario.ScenarioRegistry`, then layers
record acceptance, background training drains, and publication on top.
Request handling lives in ``reef.service``; the dispatcher is transport-free.

- ``dispatcher`` holds :class:`Dispatcher`: the public scenario API,
  the training thread, the local backend workers, the training status and
  the lifecycle of the background threads.
- ``ingest`` validates records and appends them to one loaded scenario.
- ``errors`` keeps the training errors, failure counts and stale-result
  refusals that the status and the operational metrics report.
"""

from reef.dispatcher.dispatcher import Dispatcher, build_default_dispatcher

__all__ = ["Dispatcher", "build_default_dispatcher"]
