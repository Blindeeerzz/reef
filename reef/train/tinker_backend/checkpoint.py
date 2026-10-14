"""Versioned local manifests referencing durable Tinker checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from reef.runtime.recovery import write_json

MANIFEST = "tinker-checkpoint.json"


@dataclass(frozen=True)
class TinkerCheckpoint:
    base_model: str
    lora_rank: int
    state_path: str
    sampler_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.base_model or self.lora_rank <= 0:
            raise ValueError("invalid Tinker checkpoint version or model configuration")
        for path in (self.state_path, self.sampler_path):
            if not isinstance(path, str) or not path.startswith("tinker://"):
                raise ValueError("Tinker checkpoints require remote tinker:// paths")

    def validate_model(self, base_model: str, lora_rank: int) -> None:
        if (self.base_model, self.lora_rank) != (base_model, lora_rank):
            raise ValueError("Tinker checkpoint model/rank does not match the runtime")

    def write(self, directory: Path) -> None:
        write_json(directory / MANIFEST, asdict(self))

    @classmethod
    def read(cls, directory: Path) -> TinkerCheckpoint:
        value = json.loads((directory / MANIFEST).read_text())
        if not isinstance(value, dict):
            raise ValueError("Tinker checkpoint manifest must be an object")
        return cls(**value)
