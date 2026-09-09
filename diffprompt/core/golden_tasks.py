"""
Loads golden tasks from .yaml/.yml or .jsonl files.
"""
from __future__ import annotations
import json
from pathlib import Path

import yaml

from diffprompt.models import GoldenTask


def load_golden_tasks(path: str) -> list[GoldenTask]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"golden tasks file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        tasks = _load_yaml(p)
    elif suffix == ".jsonl":
        tasks = _load_jsonl(p)
    else:
        raise ValueError(
            f"unsupported golden tasks format {suffix!r} — use .yaml, .yml, or .jsonl"
        )

    if not tasks:
        raise ValueError(f"no golden tasks found in {path}")
    return tasks


def _load_yaml(p: Path) -> list[GoldenTask]:
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw_tasks = data.get("tasks", [])
    return [GoldenTask(**t) for t in raw_tasks]


def _load_jsonl(p: Path) -> list[GoldenTask]:
    tasks = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(GoldenTask(**json.loads(line)))
    return tasks
