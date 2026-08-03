"""Session persistence.

Everything about a conversation lives in one JSON file under
~/.icarus/sessions, including which model produced which turn — useful once
you start switching models mid-conversation with /model.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    model: str = ""
    workdir: str = ""
    title: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    # (timestamp, from_model, to_model) — the audit trail for /model switches.
    model_history: List[List[Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    turns: int = 0

    @property
    def path(self) -> Path:
        return config.SESSIONS_DIR / f"{self.id}.json"

    def note_model(self, new_model: str) -> None:
        if new_model and new_model != self.model:
            self.model_history.append([time.time(), self.model, new_model])
            self.model = new_model

    def save(self) -> None:
        config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.updated = time.time()
        if not self.title:
            self.title = _derive_title(self.messages)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1))
        tmp.replace(self.path)  # atomic; a killed process can't truncate history

    @classmethod
    def load(cls, sid: str) -> Optional["Session"]:
        p = config.SESSIONS_DIR / f"{sid}.json"
        if not p.exists():
            # Allow a unique prefix, like git short hashes.
            matches = sorted(config.SESSIONS_DIR.glob(f"{sid}*.json"))
            if len(matches) != 1:
                return None
            p = matches[0]
        try:
            return cls(**json.loads(p.read_text()))
        except Exception:
            return None

    @classmethod
    def latest(cls) -> Optional["Session"]:
        items = list_sessions()
        return cls.load(items[0]["id"]) if items else None


def _derive_title(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            t = " ".join(m["content"].split())
            return (t[:70] + "…") if len(t) > 70 else t
    return "(empty)"


def list_sessions(limit: int = 30) -> List[Dict[str, Any]]:
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in config.SESSIONS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        out.append(
            {
                "id": d.get("id", p.stem),
                "updated": d.get("updated", 0),
                "title": d.get("title", ""),
                "model": d.get("model", ""),
                "turns": d.get("turns", 0),
                "tokens": (d.get("prompt_tokens", 0) + d.get("completion_tokens", 0)),
            }
        )
    out.sort(key=lambda x: x["updated"], reverse=True)
    return out[:limit]
