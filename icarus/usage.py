"""Token accounting, written into Mission Control's usage_stats.json.

MC's Token Usage tab reads four shapes out of that file — `models` (lifetime),
`recent_calls` (rolling, capped at 500), `daily`, and `hourly` (7-day
retention). Every other local caller on this box writes all four; Icarus does
too, tagged source="icarus", so its turns show up next to the rest instead of
being invisible.

Writes are best-effort: a locked or missing stats file never fails a turn.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

RECENT_CAP = 500
HOURLY_RETENTION_DAYS = 7
SOURCE = "icarus"


def _local_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _local_hour() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H")


def record(
    stats_file: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if not stats_file or (prompt_tokens <= 0 and completion_tokens <= 0):
        return
    p = Path(stats_file)
    if not p.parent.is_dir():
        return

    try:
        data: Dict[str, Any] = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return  # never clobber a file we failed to parse

    total = prompt_tokens + completion_tokens
    now = time.time()

    # 1. lifetime per-model
    models = data.setdefault("models", {})
    m = models.setdefault(
        model,
        {"total_calls": 0, "total_prompt_tokens": 0,
         "total_completion_tokens": 0, "total_tokens": 0},
    )
    m["total_calls"] = m.get("total_calls", 0) + 1
    m["total_prompt_tokens"] = m.get("total_prompt_tokens", 0) + prompt_tokens
    m["total_completion_tokens"] = m.get("total_completion_tokens", 0) + completion_tokens
    m["total_tokens"] = m.get("total_tokens", 0) + total

    # 2. rolling recent calls
    recent = data.setdefault("recent_calls", [])
    recent.append(
        {
            "timestamp": now,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "source": SOURCE,
        }
    )
    del recent[:-RECENT_CAP]

    # 3. per-day (local date — MC's reader compares against local dates)
    day = data.setdefault("daily", {}).setdefault(_local_date(), {})
    dm = day.setdefault(model, {"calls": 0, "prompt_tokens": 0,
                                "completion_tokens": 0, "total_tokens": 0,
                                "sources": {}})
    dm["calls"] += 1
    dm["prompt_tokens"] += prompt_tokens
    dm["completion_tokens"] += completion_tokens
    dm["total_tokens"] += total
    dm.setdefault("sources", {})[SOURCE] = dm.get("sources", {}).get(SOURCE, 0) + 1

    # 4. per-hour, pruned to the retention window
    hourly = data.setdefault("hourly", {})
    hm = hourly.setdefault(_local_hour(), {}).setdefault(
        model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "sources": {}}
    )
    hm["calls"] += 1
    hm["prompt_tokens"] += prompt_tokens
    hm["completion_tokens"] += completion_tokens
    hm["total_tokens"] += total
    hm.setdefault("sources", {})[SOURCE] = hm.get("sources", {}).get(SOURCE, 0) + 1

    cutoff = time.time() - HOURLY_RETENTION_DAYS * 86400
    for key in list(hourly):
        try:
            if datetime.strptime(key, "%Y-%m-%dT%H").timestamp() < cutoff:
                del hourly[key]
        except ValueError:
            continue

    # Atomic replace — MC polls this file and must never read a half-write.
    try:
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)  # type: ignore[name-defined]
        except Exception:
            pass
