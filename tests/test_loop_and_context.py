"""Regression tests for the tool-repetition loop and context compaction.

Both bugs here were found in a real session: the agent called `read_file` on
the same path with the same offset nine times in a row, byte-identical result
each time, and only stopped at the iteration cap. Two independent causes:

  1. Nothing detected the repetition, and appending an identical (call, result)
     pair to the context makes the next identical call *more* likely.
  2. `compact()` clamped old tool results to 400 chars and wrote the result
     back over the session, so the directory listing that held the answer was
     destroyed permanently — and its truncation marker made the clipped listing
     read as a complete one.

Stdlib only, no test runner required, in keeping with the rest of the project:

    python3 tests/test_loop_and_context.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from icarus import config  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="icarus-tests-"))
config.SESSIONS_DIR = _TMP / "sessions"

from icarus.caps import Caps  # noqa: E402
from icarus.context import (  # noqa: E402
    Budget,
    TOOL_CLAMP_HARD,
    TOOL_CLAMP_SOFT,
    compact,
    trim_tool_output,
)
from icarus.llm import Reply  # noqa: E402
from icarus.loop import MAX_IDENTICAL_REPEATS, REPEAT_NUDGE, Agent  # noqa: E402
from icarus.session import Session  # noqa: E402
from icarus.tools.registry import Registry, ToolResult  # noqa: E402

FAILURES: list = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------
# Fakes. The point is to drive the REAL Agent.run_turn, so the wiring is under
# test and not just the helper functions.
# --------------------------------------------------------------------------
class LoopingClient:
    """Always asks for the same tool call with the same arguments."""

    def __init__(self) -> None:
        self.disable_thinking = False
        self.calls = 0
        self.nudges_seen = 0

    def _call(self, args: dict) -> Reply:
        self.calls += 1
        return Reply(
            content="",
            tool_calls=[{
                "id": f"c{self.calls}",
                "function": {"name": "read_file", "arguments": json.dumps(args)},
            }],
            prompt_tokens=100,
            completion_tokens=10,
        )

    def complete(self, model, messages, **kw):  # noqa: ANN001
        self.nudges_seen = sum(
            1 for m in messages if m.get("role") == "tool" and m.get("content") == REPEAT_NUDGE
        )
        return self._call({"path": "/repo/app.py", "offset": 790})


class PagingClient(LoopingClient):
    """Same tool, advancing offset — legitimate paging, never a loop."""

    def complete(self, model, messages, **kw):  # noqa: ANN001
        return self._call({"path": "/repo/app.py", "offset": self.calls * 100})


def build_agent(client, tool_content, ctx: int = 32768, max_iterations: int = 20) -> Agent:
    reg = Registry()

    @reg.tool("read_file", "read a file", {"type": "object", "properties": {}})
    def _read(**kwargs):
        return ToolResult(True, tool_content(kwargs))

    return Agent(
        client=client,
        swap=None,
        registry=reg,
        session=Session(),
        caps_store=type("NullCapsStore", (), {"put": lambda self, c: None})(),
        model="fake",
        ctx=ctx,
        caps=Caps(model="fake", native_tools=True, honors_thinking_flag=True),
        cfg={
            "agent": {"max_iterations": max_iterations, "stream": False},
            "usage": {"enabled": False},
            "model": {"max_tokens": 1024},
        },
        workdir=_TMP,
    )


# --------------------------------------------------------------------------
section("[1] identical call + identical result -> nudge, then the turn is abandoned")
client = LoopingClient()
agent = build_agent(client, lambda a: "IDENTICAL BYTES\n" * 20)
stats = agent.run_turn("add favorites to the files tab")
msgs = agent.session.messages

check("turn stopped well short of the iteration cap",
      stats.iterations <= MAX_IDENTICAL_REPEATS + 2, f"iterations={stats.iterations}")
check("stats.looped set", stats.looped is True)
check("repeated calls counted", stats.repeated_calls == MAX_IDENTICAL_REPEATS,
      f"got {stats.repeated_calls}")
check("nudge replaced the duplicate payload",
      sum(1 for m in msgs if m.get("content") == REPEAT_NUDGE) == MAX_IDENTICAL_REPEATS)
check("the duplicated result bytes were appended exactly once",
      sum(1 for m in msgs if m.get("role") == "tool" and "IDENTICAL BYTES" in str(m.get("content"))) == 1)
check("the model actually saw the nudge in its context", client.nudges_seen >= 1,
      f"seen={client.nudges_seen}")
check("a closing assistant message explains the stop",
      msgs[-1]["role"] == "assistant" and "icarus stopped" in msgs[-1]["content"])
check("every tool_call has a matching tool reply",
      len([m for m in msgs if m.get("role") == "tool"]) ==
      sum(len(m.get("tool_calls", [])) for m in msgs if m.get("role") == "assistant"))

section("[2] same tool with advancing arguments is paging, not looping")
paging = PagingClient()
a2 = build_agent(paging, lambda a: f"chunk at {a.get('offset')}")
s2 = a2.run_turn("read the file in pieces")
check("not treated as a loop", s2.looped is False)
check("no repeats counted", s2.repeated_calls == 0, f"got {s2.repeated_calls}")

section("[3] same call whose result keeps changing is progress, not looping")
counter = iter(range(1000))
a3 = build_agent(LoopingClient(), lambda a: f"content v{next(counter)}")
s3 = a3.run_turn("watch the file")
check("not treated as a loop", s3.looped is False)
check("no repeats counted", s3.repeated_calls == 0, f"got {s3.repeated_calls}")

# --------------------------------------------------------------------------
section("[4] compact() never mutates the caller's history")
big = "\n".join(f"entry-{i:04d}-directory-name" for i in range(400))
history = [{"role": "system", "content": "sys"}]
for i in range(12):
    history.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]})
    history.append({"role": "tool", "tool_call_id": str(i), "name": "list_dir", "content": big})
snapshot = json.dumps(history)

budget = Budget(ctx=4096, reserve_output=512, threshold=0.75)
out, note = compact(history, budget)
check("compaction triggered", note is not None, str(note))
check("input messages not mutated", json.dumps(history) == snapshot)
check("originals still at full length",
      all(len(m["content"]) == len(big) for m in history if m["role"] == "tool"))
check("the returned copy was reduced", sum(len(str(m.get("content"))) for m in out) < len(snapshot))

section("[5] the soft clamp is tried before the hard one, and only as far as needed")
mild = [{"role": "system", "content": "sys"}]
for i in range(6):
    mild.append({"role": "assistant", "content": "", "tool_calls": [{"id": str(i)}]})
    mild.append({"role": "tool", "tool_call_id": str(i), "name": "list_dir", "content": big})
out5, note5 = compact(mild, Budget(ctx=16384, reserve_output=512, threshold=0.75))
squeezed = [m for m in out5 if m.get("role") == "tool" and len(m["content"]) < len(big)]
if note5:
    check("squeezed results keep far more than the old 400-char hard clamp",
          all(len(m["content"]) > TOOL_CLAMP_HARD for m in squeezed),
          f"sizes={[len(m['content']) for m in squeezed]}")
    check("only as many results squeezed as the budget required", len(squeezed) < 6,
          f"squeezed {len(squeezed)}/6")
else:
    check("compaction triggered for the mild case", False, "expected a note")

# --------------------------------------------------------------------------
section("[6] truncation is line-aware and says entries are hidden, not absent")
# Shaped like the listing that caused the original failure: ~180 entries, with
# the directory the agent needed sitting in the middle where a character-based
# middle-out cut would silently drop it.
names = sorted([f"dir-{i:03d}" for i in range(178)] + ["mission-control-next"])
listing = "/repo\n" + "\n".join(f"d               {n}" for n in names)

full = trim_tool_output(listing, TOOL_CLAMP_SOFT)
check("the needed entry survives the soft clamp", "mission-control-next" in full)
check("soft-clamped output respects its budget", len(full) <= TOOL_CLAMP_SOFT, f"{len(full)}")

hard = trim_tool_output(listing, TOOL_CLAMP_HARD)
check("hard-clamped output respects its budget", len(hard) <= TOOL_CLAMP_HARD, f"{len(hard)}")
check("the hard clamp still reports how many entries are missing", "lines elided" in hard)
check("the marker says the entries are hidden rather than absent", "NOT absent" in hard)
check("cuts land on line boundaries", all(
    ln.startswith(("d ", "/repo", "...")) for ln in hard.split("\n") if ln))

check("short output is returned untouched", trim_tool_output("a\nb\nc", 500) == "a\nb\nc")
check("degenerate clamps do not crash", isinstance(trim_tool_output("abc", 2), str))

section("[7] trim_tool_output stays within budget across shapes")
import random  # noqa: E402

random.seed(11)
over = 0
for _ in range(5000):
    n = random.choice([0, 1, 2, 3, 5, 20, 178, 300, 5000])
    text = "\n".join("x" * random.randint(0, 120) for _ in range(n))
    clamp = random.choice([120, 400, 2000, 30000])
    result = trim_tool_output(text, clamp)
    if len(text) <= clamp:
        if result != text:
            over += 1
    elif len(result) > clamp:
        over += 1
check("no over-budget or altered results in 5000 cases", over == 0, f"{over} bad")

# --------------------------------------------------------------------------
shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
