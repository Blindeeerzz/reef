"""The unit that a processor puts in a batch: one sample, or one ready group."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class BatchUnit:
    """One unit that a processor can put in a batch: one sample, or one ready group.

    A processor subclasses this type to add its own data, such as the sample or
    the reports of a group. The buffer, the selection policies, and the
    eligibility checks read only these fields:

    * ``arrival_index`` tells when the processor received the unit. A group
      uses the index of its oldest member.
    * ``group_key`` names the group of a unit that is a group.
    * ``runtime_load_id`` is the runtime load ID that produced the unit, if known.
    * ``source_index`` tells when the source record of the unit arrived, if it
      differs from ``arrival_index``.
    """

    unit_id: Hashable
    arrival_index: int
    group_key: Hashable | None = None
    runtime_load_id: str | None = None
    source_index: int | None = None
