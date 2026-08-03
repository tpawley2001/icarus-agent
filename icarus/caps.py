"""Per-model capability detection, cached to disk.

The models behind one llama-swap install are wildly inconsistent: some support
native tool calls, some don't; some are reasoning models whose template hides
the answer inside <think> unless told otherwise. Probing costs a model load, so
we do it once per model and cache the answer in ~/.icarus/capabilities.json.

Both probes are cheap (a handful of tokens) and run against an already-resident
model whenever possible.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .llm import LLMClient, LLMError

# Name fragments that identify reasoning-tuned families. Used only as the
# opening guess — an actual probe overrides it.
REASONING_HINTS = (
    "qwen3", "qwq", "deepseek-r", "magistral", "thinking", "reason",
    "glm4", "granite4", "gemma4", "qwen3.5", "qwen3.6", "mythos",
)

PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "icarus_probe",
            "description": "Echo a token back. Used only for capability detection.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]


@dataclass
class Caps:
    model: str
    native_tools: bool = True
    reasoning: bool = False
    # Whether disable_thinking actually changed the output. If a model ignores
    # the flag we keep stripping <think> client-side instead.
    honors_thinking_flag: bool = True
    ctx: int = 0
    probed_at: float = 0.0
    note: str = ""


class CapsStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = path or config.CAPS_PATH
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def get(self, model: str) -> Optional[Caps]:
        raw = self._data.get(model)
        return Caps(**raw) if raw else None

    def put(self, caps: Caps) -> None:
        self._data[caps.model] = asdict(caps)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        except Exception:
            pass  # a read-only home must not break the run

    def forget(self, model: str) -> None:
        self._data.pop(model, None)
        try:
            self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        except Exception:
            pass

    def all(self) -> Dict[str, Caps]:
        return {k: Caps(**v) for k, v in self._data.items()}


def guess_reasoning(model: str) -> bool:
    low = model.lower()
    return any(h in low for h in REASONING_HINTS)


def probe(
    client: LLMClient,
    model: str,
    ctx: int = 0,
    timeout: int = 600,
    log=lambda _s: None,
) -> Caps:
    """Detect native tool support and reasoning behaviour for one model."""
    caps = Caps(model=model, ctx=ctx, probed_at=time.time(),
                reasoning=guess_reasoning(model))

    # --- native tool calling -------------------------------------------
    log("probing tool support")
    try:
        r = client.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You must call the icarus_probe tool."},
                {"role": "user", "content": "Call icarus_probe with value='ok'."},
            ],
            tools=PROBE_TOOL,
            max_tokens=128,
            temperature=0.0,
            timeout=timeout,
        )
        caps.native_tools = bool(r.tool_calls)
        if not caps.native_tools:
            caps.note = "no tool_calls returned; using text protocol"
    except LLMError as e:
        # A 500 on a tools payload is the classic "template has no tool support"
        # signature — that is a definitive negative, not an outage.
        msg = str(e).lower()
        if "http 5" in msg or "template" in msg or "tool" in msg:
            caps.native_tools = False
            caps.note = "server rejected tools payload; using text protocol"
        else:
            caps.native_tools = False
            caps.note = f"probe failed: {str(e)[:120]}"

    # --- reasoning / thinking -------------------------------------------
    log("probing reasoning behaviour")
    try:
        thinking_off = LLMClient(
            base_url=client.base_url,
            api_key=client.api_key,
            request_timeout=client.request_timeout,
            disable_thinking=True,
        )
        r2 = thinking_off.complete(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: READY"}],
            max_tokens=64,
            temperature=0.0,
            timeout=timeout,
        )
        # Empty content with the flag set is the known llama-swap/--jinja
        # failure: the whole reply went into a reasoning channel we can't see.
        if not (r2.content or "").strip():
            caps.reasoning = True
            caps.honors_thinking_flag = False
            caps.note = (caps.note + "; " if caps.note else "") + (
                "returns empty content with thinking disabled — "
                "keep thinking ON for this model"
            )
        elif re.search(r"<think>", r2.content, re.IGNORECASE):
            caps.reasoning = True
            caps.honors_thinking_flag = False
    except LLMError as e:
        caps.note = (caps.note + "; " if caps.note else "") + f"reasoning probe failed: {str(e)[:80]}"

    return caps


def ensure(
    store: CapsStore,
    client: LLMClient,
    model: str,
    ctx: int = 0,
    force: bool = False,
    log=lambda _s: None,
) -> Caps:
    """Cached probe. Call before the first real turn on a model."""
    if not force:
        cached = store.get(model)
        if cached:
            if ctx and cached.ctx != ctx:
                cached.ctx = ctx
                store.put(cached)
            return cached
    caps = probe(client, model, ctx=ctx, log=log)
    store.put(caps)
    return caps
