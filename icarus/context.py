"""Context budgeting and compaction.

Local models are context-poor: 8K is common on this box and 32K is a luxury,
versus the 200K a cloud agent assumes. Two consequences drive this module:

1. Every turn must be budgeted *before* it is sent, because llama.cpp silently
   truncates from the left when the prompt overflows — which usually eats the
   system prompt and makes the agent forget its tools mid-task.
2. Switching models in flight can shrink the window underneath an existing
   conversation, so the budget has to be re-checked on every model change.

Token counting uses a calibrated chars-per-token ratio rather than a tokenizer:
llama-server reports real prompt_tokens on every reply, so we learn the true
ratio for the running model within a turn or two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Starting guess before any reply has calibrated us. English prose over common
# BPE vocabularies sits near 4.0; code and JSON run denser.
DEFAULT_CHARS_PER_TOKEN = 3.6


@dataclass
class Budget:
    ctx: int
    reserve_output: int = 4096
    threshold: float = 0.75
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    _samples: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def usable(self) -> int:
        """Prompt tokens we may spend, leaving room for the reply."""
        return max(1024, self.ctx - self.reserve_output)

    @property
    def limit(self) -> int:
        """Soft ceiling that triggers compaction."""
        return int(self.usable * self.threshold)

    def calibrate(self, messages: List[Dict[str, Any]], actual_prompt_tokens: int) -> None:
        """Learn the real chars-per-token from a served request."""
        if actual_prompt_tokens <= 0:
            return
        chars = _chars(messages)
        if chars <= 0:
            return
        self._samples.append((chars, actual_prompt_tokens))
        self._samples = self._samples[-8:]
        tot_c = sum(c for c, _ in self._samples)
        tot_t = sum(t for _, t in self._samples)
        if tot_t > 0:
            # Clamp: a wild ratio from one odd turn shouldn't wreck budgeting.
            self.chars_per_token = min(8.0, max(1.5, tot_c / tot_t))

    def estimate(self, messages: List[Dict[str, Any]]) -> int:
        return int(_chars(messages) / self.chars_per_token) + 8 * len(messages)


def _chars(messages: List[Dict[str, Any]]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif c:
            n += len(json.dumps(c))
        for tc in m.get("tool_calls") or []:
            n += len(json.dumps(tc))
    return n


def trim_tool_output(text: str, max_chars: int) -> str:
    """Middle-out truncation — the head and tail of output carry the meaning."""
    if len(text) <= max_chars:
        return text
    keep = max_chars // 2 - 40
    dropped = len(text) - 2 * keep
    return (
        text[:keep]
        + f"\n\n... [{dropped:,} characters elided by icarus] ...\n\n"
        + text[-keep:]
    )


def compact(
    messages: List[Dict[str, Any]],
    budget: Budget,
    summarize=None,
    keep_recent: int = 6,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Shrink history to fit the budget.

    Strategy, cheapest first:
      1. Always keep the system prompt and the last `keep_recent` messages.
      2. Squeeze old tool results, which dominate token count in agent traces.
      3. If still over, summarize the middle via `summarize` (a callable taking
         messages and returning text). Falls back to dropping if unavailable.

    Returns (new_messages, note) where note describes what happened, or None if
    nothing was needed.
    """
    if budget.estimate(messages) <= budget.limit:
        return messages, None

    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    if len(rest) <= keep_recent:
        return messages, None

    head, tail = rest[:-keep_recent], rest[-keep_recent:]

    # Pass 1 — clamp old tool results hard. They are the bulk of the tokens and
    # the least useful once the agent has moved on.
    squeezed = []
    for m in head:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            m = dict(m)
            m["content"] = trim_tool_output(m["content"], 400)
        squeezed.append(m)

    candidate = system + squeezed + tail
    if budget.estimate(candidate) <= budget.limit:
        return candidate, f"compacted: trimmed {len(squeezed)} older tool results"

    # Pass 2 — summarize the middle.
    if summarize is not None:
        try:
            digest = summarize(squeezed)
        except Exception:
            digest = ""
        if digest:
            note_msg = {
                "role": "user",
                "content": (
                    "[Earlier conversation, summarized to fit this model's "
                    f"context window]\n\n{digest}"
                ),
            }
            candidate = system + [note_msg] + tail
            if budget.estimate(candidate) <= budget.limit:
                return candidate, f"compacted: summarized {len(squeezed)} earlier messages"

    # Pass 3 — drop oldest until it fits. Guaranteed to terminate.
    kept = list(squeezed)
    while kept and budget.estimate(system + kept + tail) > budget.limit:
        kept.pop(0)
    dropped = len(squeezed) - len(kept)
    return system + kept + tail, f"compacted: dropped {dropped} oldest messages"


def fits(messages: List[Dict[str, Any]], ctx: int, reserve: int = 4096) -> bool:
    b = Budget(ctx=ctx, reserve_output=reserve)
    return b.estimate(messages) <= b.usable
