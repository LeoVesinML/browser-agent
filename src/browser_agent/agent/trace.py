"""Append-only run log. Every run is reproducible from its trace."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, root: Path, name: str | None = None) -> None:
        stamp = name or time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"
        self.t0 = time.time()

    def write(self, kind: str, **payload: Any) -> None:
        record = {"t": round(time.time() - self.t0, 2), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def screenshot(self, data: bytes, step: int) -> Path:
        path = self.dir / f"step-{step:03d}.jpg"
        path.write_bytes(data)
        return path
