"""Versioned model snapshots with rollback.

A snapshot is a full, serialisable copy of a learner's state plus the
metrics observed when it was taken. The registry is append-only for
history (old versions are never overwritten); rollback only moves the
``current`` pointer, so a bad rollback is itself reversible.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ModelSnapshot:
    version: int
    learner_kind: str
    state: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    note: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ModelRegistry:
    def __init__(self) -> None:
        self._snapshots: dict[int, ModelSnapshot] = {}
        self._current: int | None = None
        self._next_version = 1

    def snapshot(
        self,
        learner: Any,
        metrics: dict[str, float] | None = None,
        note: str = "",
    ) -> ModelSnapshot:
        snap = ModelSnapshot(
            version=self._next_version,
            learner_kind=type(learner).__name__,
            state=copy.deepcopy(learner.state_dict()),
            metrics=dict(metrics or {}),
            note=note,
        )
        self._snapshots[snap.version] = snap
        self._current = snap.version
        self._next_version += 1
        return snap

    def get(self, version: int) -> ModelSnapshot:
        if version not in self._snapshots:
            raise KeyError(f"no snapshot version {version}")
        return self._snapshots[version]

    def current(self) -> ModelSnapshot | None:
        if self._current is None:
            return None
        return self._snapshots[self._current]

    def rollback(self, version: int) -> ModelSnapshot:
        """Move the current pointer back; history is preserved."""
        snap = self.get(version)  # raises KeyError for unknown versions
        self._current = version
        return snap

    def restore_into(self, learner: Any, version: int | None = None) -> ModelSnapshot:
        """Load the current (or given) snapshot's state into a learner."""
        target = version if version is not None else self._current
        if target is None:
            raise ValueError("no snapshot to restore")
        snap = self.get(target)
        learner.load_state_dict(copy.deepcopy(snap.state))
        self._current = target
        return snap

    def history(self) -> list[ModelSnapshot]:
        return [self._snapshots[v] for v in sorted(self._snapshots)]

    def __len__(self) -> int:
        return len(self._snapshots)
