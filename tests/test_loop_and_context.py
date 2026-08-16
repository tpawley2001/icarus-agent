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
from icarus.llm import LLMError, Reply  # noqa: E402
from icarus.loop import (  # noqa: E402
    MAX_IDENTICAL_REPEATS,
    REPEAT_NUDGE,
    Agent,
    _is_context_overflow,
    _is_dead_model,
)
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

section("[8] a served context-overflow is recovered, not fatal")

# The exact body llama-swap returned in the real incident, plus a few phrasings
# llama.cpp uses, must all be recognised as overflow (and unrelated 5xx must not).
check("recognises the incident's error body", _is_context_overflow(
    'HTTP 500 from model server: {"error":{"code":500,'
    '"message":"Context size has been exceeded.","type":"server_error"}}'))
check("recognises llama.cpp n_ctx phrasing",
      _is_context_overflow("the request exceeds the available n_ctx"))
check("does not mistake an unrelated 5xx for overflow",
      not _is_context_overflow("HTTP 500: upstream command exited prematurely"))


class OverflowThenClient:
    """Rejects the first N prompts for context overflow, then answers.

    Mirrors the incident: a giant tool result has bloated the history, the
    server rejects the prompt, and the loop must shrink and retry rather than
    give up. Records the largest tool message it was handed each call, so the
    test can prove the retry prompt actually shrank.
    """

    def __init__(self, reject_times: int) -> None:
        self.disable_thinking = False
        self.reject_times = reject_times
        self.calls = 0
        self.biggest_tool_chars: list = []

    def complete(self, model, messages, **kw):  # noqa: ANN001
        self.calls += 1
        self.biggest_tool_chars.append(max(
            (len(m.get("content") or "") for m in messages if m.get("role") == "tool"),
            default=0,
        ))
        if self.calls <= self.reject_times:
            raise LLMError(
                'HTTP 500 from model server: '
                '{"error":{"message":"Context size has been exceeded."}}'
            )
        return Reply(content="done", prompt_tokens=100, completion_tokens=5)


agent = build_agent(OverflowThenClient(reject_times=1), lambda k: "")
# Seed history with a huge tool dump — the kind a $HOME-wide search produced.
agent.session.messages.append({"role": "user", "content": "find the scraper"})
agent.session.messages.append(
    {"role": "tool", "tool_call_id": "c0", "name": "search_files",
     "content": "\n".join(f"/home/tyson/.cache/junk/file{i}.py:1:x" for i in range(4000))}
)
huge_before = len(agent.session.messages[-1]["content"])
stats = agent.run_turn("continue")

last = agent.session.messages[-1]
check("the turn recovered and produced an answer",
      last.get("role") == "assistant" and last.get("content") == "done")
check("no [model error] was recorded",
      not any("[model error" in str(m.get("content")) for m in agent.session.messages))
check("an emergency shrink was noted",
      any("context-overflow" in n for n in stats.compactions))
check("the bloated tool dump was hard-trimmed in stored history",
      len(agent.session.messages[1]["content"]) < huge_before)
check("the retry prompt was smaller than the one that overflowed",
      agent.client.biggest_tool_chars[1] < agent.client.biggest_tool_chars[0],
      f"{agent.client.biggest_tool_chars}")

# When the server never stops rejecting, the turn ends cleanly with an error,
# not an exception or a hang.
agent2 = build_agent(OverflowThenClient(reject_times=99), lambda k: "")
agent2.session.messages.append(
    {"role": "tool", "tool_call_id": "c0", "name": "search_files",
     "content": "x" * 200_000}
)
stats2 = agent2.run_turn("go")
check("unrecoverable overflow degrades to a recorded model error",
      any("[model error" in str(m.get("content")) for m in agent2.session.messages))
check("it stopped retrying rather than looping forever",
      agent2.client.calls <= 3, f"{agent2.client.calls} calls")


section("[9] search excludes dependency/cache trees")
from icarus.tools import builtin  # noqa: E402

sandbox = _TMP / "search_sandbox"
(sandbox / "src").mkdir(parents=True)
(sandbox / "node_modules" / "pkg").mkdir(parents=True)
(sandbox / ".cache" / "uv").mkdir(parents=True)
(sandbox / "src" / "app.py").write_text("import needle\n")
(sandbox / "node_modules" / "pkg" / "index.js").write_text("var needle = 1\n")
(sandbox / ".cache" / "uv" / "x.py").write_text("needle = 2\n")

reg = builtin.build(workdir=sandbox)
hits = reg.dispatch("search_files", json.dumps({"pattern": "needle", "path": str(sandbox)}))
check("finds the real source hit", "src/app.py" in hits.content)
check("excludes node_modules", "node_modules" not in hits.content)
check("excludes hidden cache dirs", "/.cache/" not in hits.content)

globbed = reg.dispatch("glob_files", json.dumps({"pattern": "**/*.py", "path": str(sandbox)}))
check("glob finds the source file", "src/app.py" in globbed.content)
check("glob excludes cache trees", "/.cache/" not in globbed.content)


# --------------------------------------------------------------------------
section("[10] an unloadable model reports actionably instead of raw JSON")

# The body llama-swap returns when llama-server dies at load (the incident:
# huihui_ai/gemma-4-abliterated:latest, a 2131-tensor Gemma-4n MatFormer export
# this build cannot construct), plus the name-not-in-config case.
check("recognises an upstream that died at load", _is_dead_model(
    'HTTP 500 from model server: {"src":"llama-swap", '
    '"error": "unspecific error: upstream command exited prematurely"}'))
check("recognises an unknown model name", _is_dead_model(
    'HTTP 500 from model server: {"src":"llama-swap", '
    '"error": "no router for requested model"}'))
check("recognises llama.cpp's own load failure",
      _is_dead_model("error loading model: done_getting_tensors: wrong number "
                     "of tensors; expected 2131, got 720"))
check("does not mistake a context overflow for a dead model",
      not _is_dead_model("Context size has been exceeded."))
check("does not mistake an ordinary refusal for a dead model",
      not _is_dead_model("I can't help with that."))


class DeadModelClient:
    """Every call fails the way an unloadable model does."""

    def __init__(self) -> None:
        self.disable_thinking = False
        self.calls = 0

    def complete(self, model, messages, **kw):  # noqa: ANN001
        self.calls += 1
        raise LLMError(
            'HTTP 500 from model server: {"src":"llama-swap", '
            '"error": "unspecific error: upstream command exited prematurely"}'
        )


client = DeadModelClient()
agent = build_agent(client, lambda k: "")
agent.model = "huihui_ai/gemma-4-abliterated:latest"
agent.run_turn("how might I build a news app?")

recorded = str(agent.session.messages[-1].get("content"))
check("the turn is not silently lost", "[model error" in recorded)
check("the dead model is named", "gemma-4-abliterated" in recorded)
check("the user is told how to recover", "/model" in recorded)
check("the raw upstream body is preserved for debugging",
      "exited prematurely" in recorded)
check("it does not hammer the dead upstream", client.calls <= 2,
      f"{client.calls} calls")


# --------------------------------------------------------------------------
shutil.rmtree(_TMP, ignore_errors=True)
print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
sys.exit(1 if FAILURES else 0)
