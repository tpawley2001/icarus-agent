"""The agent loop.

One user turn drives: budget → model call → tool dispatch → repeat, until the
model answers without asking for a tool or the iteration cap is hit.

Two things here exist because the models are local:

* `switch_model` can be called between iterations. The conversation survives;
  only the budget, the capability set, and the tool-calling mode change. That
  is what makes /model work mid-conversation.
* The tool-call path has two implementations (native and text protocol) chosen
  per model from the capability cache, and it will fall back at runtime if a
  model that claimed native support returns prose instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import protocol, usage
from .caps import Caps, CapsStore, ensure as ensure_caps
from .context import Budget, compact, trim_tool_output
from .llm import LLMClient, LLMError, Reply
from .session import Session
from .swap import SwapClient
from .tools.registry import Registry

BASE_PROMPT = """You are Icarus, a command-line agent running entirely on the user's own machine against a local model served by llama-swap. Nothing you do leaves this machine.

Operating rules:
- Work in small, verifiable steps. Take an action, read the real result, then decide the next one. Never claim you did something you did not actually do with a tool.
- Prefer tools over speculation. If a fact is on disk or obtainable from a command, go get it rather than guessing.
- You are running on a local model with a limited context window. Be concise. Do not restate long file contents back to the user; refer to paths and line numbers.
- When a command fails, read the error and adapt. Do not repeat an identical failing call.
- Destructive shell commands require the user's approval; if one is denied, do not try to route around it.
- When the task is done, say so plainly and stop calling tools.

Working directory: {workdir}
Today is {date}."""


# How many times the same tool call may come back with a byte-identical result
# before the turn is abandoned. Small models fall into repetition attractors:
# an identical (call, result) pair appended to the context makes the next
# identical call *more* likely, so the trace can only be broken from outside.
MAX_IDENTICAL_REPEATS = 3

REPEAT_NUDGE = (
    "[icarus] You already made this exact call with these exact arguments, and "
    "the result is byte-for-byte identical to the one above. Repeating it "
    "cannot tell you anything new. Change the tool or the arguments — look "
    "somewhere you have not looked yet — or stop and tell the user what you "
    "found and what is blocking you."
)


def _call_signature(name: str, raw_args: Any) -> str:
    """Stable key for a tool call, insensitive to key order and whitespace."""
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        args = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (json.JSONDecodeError, TypeError):
        args = str(raw_args)
    return f"{name}({args})"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


@dataclass
class TurnStats:
    iterations: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed: float = 0.0
    compactions: List[str] = field(default_factory=list)
    interrupted: bool = False
    steered: int = 0
    repeated_calls: int = 0
    looped: bool = False


@dataclass
class Agent:
    client: LLMClient
    swap: SwapClient
    registry: Registry
    session: Session
    caps_store: CapsStore
    model: str
    ctx: int
    caps: Caps
    cfg: Dict[str, Any]
    workdir: Path
    skills_catalog: str = ""
    # The list object the `todo` tool closed over. /new and /resume must mutate
    # this in place rather than rebinding session.todos, or the tool keeps
    # writing into the previous conversation's list.
    todo_ref: Optional[List[Dict[str, Any]]] = None
    # UI callbacks — the CLI supplies these; batch mode leaves them as no-ops.
    on_text: Optional[Callable[[str], None]] = None
    on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_tool_end: Optional[Callable[[str, Any], None]] = None
    on_status: Optional[Callable[[str], None]] = None
    # Runtime toggle, flipped by /think. None means "follow the model's caps".
    thinking_override: Optional[bool] = None
    # Type-ahead / Esc watcher. An InputWatcher in the REPL, a NullWatcher in
    # one-shot mode. See interrupt.py.
    watcher: Any = None

    def adopt_session(self, new: Session) -> None:
        """Point the agent at a different conversation without breaking the
        `todo` tool's captured list reference."""
        if self.todo_ref is not None:
            self.todo_ref.clear()
            self.todo_ref.extend(new.todos)
            new.todos = self.todo_ref
        self.session = new

    # ---- prompt ---------------------------------------------------------
    def system_prompt(self) -> str:
        custom = (self.cfg.get("agent", {}) or {}).get("system_prompt") or ""
        base = custom or BASE_PROMPT.format(
            workdir=self.workdir, date=time.strftime("%Y-%m-%d")
        )
        parts = [base]
        if self.skills_catalog:
            parts.append(
                "\n## Skills available\n"
                "These are playbooks you can load with the `skill` tool when relevant:\n"
                + self.skills_catalog
            )
        if not self.caps.native_tools:
            parts.append("\n" + protocol.render_instructions(self._schemas()))
        return "\n".join(parts)

    def _tools(self):
        tcfg = self.cfg.get("tools", {}) or {}
        return self.registry.enabled(tcfg.get("enabled") or [], tcfg.get("disabled") or [])

    def _schemas(self) -> List[Dict[str, Any]]:
        return self.registry.schemas(self._tools())

    # ---- thinking control ----------------------------------------------
    @property
    def thinking_enabled(self) -> bool:
        """Whether reasoning is left on for this model.

        A model that returns empty content when thinking is disabled must keep
        it ON — we strip the <think> block client-side instead. The user's
        /think override wins otherwise.
        """
        if not self.caps.honors_thinking_flag:
            return True
        if self.thinking_override is not None:
            return self.thinking_override
        return False  # default: off, for speed and for shorter local outputs

    def _sync_thinking(self) -> None:
        self.client.disable_thinking = not self.thinking_enabled

    # ---- model switching -------------------------------------------------
    def switch_model(self, new_model: str, unload_previous: bool = True) -> str:
        """Change model mid-conversation. Returns a human-readable summary.

        The message history is kept verbatim. What changes: context budget,
        capability set (which can flip tool-calling mode), and the system
        prompt, which is rebuilt on the next turn.
        """
        old = self.model
        if new_model == old:
            return f"Already using {new_model}."

        notes = []
        if unload_previous and old:
            # Freeing first avoids a window where both models are resident,
            # which is what OOMs the big ones on a two-GPU box.
            if self.swap.unload_all():
                notes.append(f"unloaded {old}")

        self.model = new_model
        self.ctx = self.swap.context_window(new_model) or self.ctx
        if self.on_status:
            self.on_status(f"probing {new_model}")
        self._sync_thinking()
        self.caps = ensure_caps(
            self.caps_store, self.client, new_model, ctx=self.ctx,
            log=lambda s: self.on_status and self.on_status(s),
        )
        self._sync_thinking()
        self.session.note_model(new_model)

        notes.append(f"ctx {self.ctx:,}")
        notes.append("native tools" if self.caps.native_tools else "text-protocol tools")
        if not self.caps.honors_thinking_flag:
            notes.append("thinking forced ON (model returns empty output otherwise)")

        # The new model may have a smaller window than the conversation needs.
        # Say so, but don't shrink the stored history to fit: run_turn compacts
        # per request against the current budget, so switching to a small model
        # and back no longer costs the conversation anything permanently.
        budget = self._budget()
        if budget.estimate(self.session.messages) > budget.limit:
            notes.append("history exceeds the new window; it will be compacted per request")
        return f"Switched {old or '(none)'} → {new_model}: " + ", ".join(notes)

    # ---- context --------------------------------------------------------
    def _budget(self) -> Budget:
        acfg = self.cfg.get("agent", {}) or {}
        return Budget(
            ctx=self.ctx,
            reserve_output=int((self.cfg.get("model", {}) or {}).get("max_tokens", 4096)),
            threshold=float(acfg.get("context_threshold", 0.75)),
        )

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Ask the model to compress its own earlier history."""
        transcript = []
        for m in messages[-40:]:
            role = m.get("role", "?")
            body = m.get("content")
            if not isinstance(body, str):
                body = json.dumps(body)[:400]
            transcript.append(f"[{role}] {body[:600]}")
        try:
            r = self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content":
                     "Summarize this agent transcript for your own future reference. "
                     "Keep decisions made, file paths touched, commands run and their "
                     "outcomes, and anything still unfinished. Be terse. No preamble."},
                    {"role": "user", "content": "\n".join(transcript)},
                ],
                max_tokens=700,
                temperature=0.0,
            )
            return r.content.strip()
        except LLMError:
            return ""

    # ---- the loop -------------------------------------------------------
    def run_turn(self, user_input: str) -> TurnStats:
        started = time.time()
        stats = TurnStats()
        self._sync_thinking()

        self.session.messages.append({"role": "user", "content": user_input})
        mcfg = self.cfg.get("model", {}) or {}
        acfg = self.cfg.get("agent", {}) or {}
        max_iter = int(acfg.get("max_iterations", 40))
        stream = bool(acfg.get("stream", True)) and self.on_text is not None
        budget = self._budget()

        # Signature -> (digest of the last result, consecutive identical repeats).
        # Scoped to the turn: repeating a lookup in a later turn is legitimate,
        # since the file or directory may well have changed by then.
        seen_calls: Dict[str, Tuple[str, int]] = {}
        looping = False

        for _ in range(max_iter):
            stats.iterations += 1

            # Anything typed while the last step ran becomes steering, injected
            # here rather than mid-request so the model sees a clean history.
            if self._drain_steering(stats):
                pass
            if self._check_abort(stats):
                break

            # Compaction shapes THIS request only — the session keeps the full
            # history. Writing the squeezed copy back used to destroy the
            # originals permanently, so a tool result that got clamped early
            # (a directory listing, say) could never be consulted again even
            # after later messages fell out of the window.
            history, note = compact(
                self.session.messages, budget, summarize=self._summarize
            )
            if note:
                stats.compactions.append(note)
                if self.on_status:
                    self.on_status(note)

            messages = [{"role": "system", "content": self.system_prompt()}] + history
            tools = self._schemas() if self.caps.native_tools else None

            try:
                reply = self.client.complete(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=float(mcfg.get("temperature", 0.2)),
                    top_p=float(mcfg.get("top_p", 0.95)),
                    max_tokens=int(mcfg.get("max_tokens", 4096)),
                    stream=stream,
                    on_token=self.on_text if stream else None,
                    should_abort=(self.watcher.aborted if self.watcher else None),
                )
            except LLMError as e:
                self.session.messages.append(
                    {"role": "assistant", "content": f"[model error: {e}]"}
                )
                if self.on_status:
                    self.on_status(f"error: {e}")
                break

            stats.prompt_tokens += reply.prompt_tokens
            stats.completion_tokens += reply.completion_tokens
            budget.calibrate(messages, reply.prompt_tokens)
            self._record_usage(reply)

            if reply.interrupted:
                # Keep the partial answer: it is often most of the reply, and
                # dropping it would make the follow-up turn incoherent.
                if (reply.content or "").strip():
                    self.session.messages.append(
                        {"role": "assistant",
                         "content": reply.content + "\n\n[interrupted by user]"}
                    )
                stats.interrupted = True
                if self.on_status:
                    self.on_status("interrupted")
                self._drain_steering(stats)
                break

            calls = list(reply.tool_calls)
            content = reply.content

            # A model advertised as native-capable can still answer in prose
            # with a fenced call. Parse it rather than losing the turn.
            if not calls:
                prose, parsed = protocol.parse(content)
                if parsed:
                    calls, content = parsed, prose
                    if self.caps.native_tools:
                        self.caps.native_tools = False
                        self.caps.note = "downgraded at runtime: emitted text tool calls"
                        self.caps_store.put(self.caps)
                        if self.on_status:
                            self.on_status("model used text tool calls; switching mode")

            if content and not stream and self.on_text:
                self.on_text(content)

            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                assistant_msg["tool_calls"] = calls
            self.session.messages.append(assistant_msg)

            if not calls:
                break

            for call in calls:
                if self.watcher is not None and self.watcher.aborted():
                    # Esc between tool calls: tell the model the rest were
                    # skipped, so the next turn doesn't assume they ran.
                    self.session.messages.append(
                        {"role": "tool", "tool_call_id": call.get("id", ""),
                         "name": (call.get("function") or {}).get("name", ""),
                         "content": "[skipped — user interrupted]"}
                    )
                    stats.interrupted = True
                    continue
                if looping:
                    # Every tool_call in the batch still needs a reply or the
                    # next request is malformed, so answer the rest cheaply.
                    self.session.messages.append(
                        {"role": "tool", "tool_call_id": call.get("id", ""),
                         "name": (call.get("function") or {}).get("name", ""),
                         "content": "[skipped — turn stopped: repeated tool call]"}
                    )
                    continue
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                try:
                    shown = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    shown = {"_raw": str(raw_args)[:200]}

                if self.on_tool_start:
                    self.on_tool_start(name, shown if isinstance(shown, dict) else {})
                result = self.registry.dispatch(name, raw_args)
                stats.tool_calls += 1
                if self.on_tool_end:
                    self.on_tool_end(name, result)

                # Same call, same bytes back? Appending the duplicate would only
                # reinforce the repetition — send a nudge in its place, and give
                # up on the turn if the nudge keeps not landing.
                sig = _call_signature(name, raw_args)
                digest = _digest(result.content)
                prev_digest, repeats = seen_calls.get(sig, ("", 0))
                repeats = repeats + 1 if digest == prev_digest else 0
                seen_calls[sig] = (digest, repeats)

                if repeats:
                    stats.repeated_calls += 1
                    self.session.messages.append(
                        {"role": "tool", "tool_call_id": call.get("id", ""),
                         "name": name, "content": REPEAT_NUDGE}
                    )
                    if repeats >= MAX_IDENTICAL_REPEATS:
                        looping = True
                        stats.looped = True
                        if self.on_status:
                            self.on_status(
                                f"stopped: {name} repeated with identical results"
                            )
                    continue

                self.session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": trim_tool_output(
                            result.content,
                            int((self.cfg.get("tools", {}) or {}).get("max_output_chars", 30000)),
                        ),
                    }
                )

            if looping:
                self.session.messages.append(
                    {"role": "assistant",
                     "content": "[icarus stopped this turn: the same tool call kept "
                                "returning identical results]"}
                )
                break
        else:
            if self.on_status:
                self.on_status(f"stopped at the {max_iter}-iteration cap")

        stats.elapsed = time.time() - started
        self.session.turns += 1
        self.session.prompt_tokens += stats.prompt_tokens
        self.session.completion_tokens += stats.completion_tokens
        self.session.note_model(self.model)
        self.session.save()
        return stats

    # ---- interrupt / steering plumbing ----------------------------------
    def _drain_steering(self, stats: TurnStats) -> bool:
        """Fold anything typed during the last step into the conversation.

        Queued text becomes a normal user message, so the model treats it as
        the latest instruction while keeping everything it has already done.
        """
        if self.watcher is None:
            return False
        pending = self.watcher.take_messages()
        if not pending:
            return False
        for text in pending:
            self.session.messages.append(
                {"role": "user",
                 "content": f"[The user sent this while you were working]\n\n{text}"}
            )
            stats.steered += 1
            if self.on_status:
                self.on_status(f"steering: {text[:70]}")
        return True

    def _check_abort(self, stats: TurnStats) -> bool:
        if self.watcher is not None and self.watcher.aborted():
            stats.interrupted = True
            return True
        return False

    def _record_usage(self, reply: Reply) -> None:
        ucfg = self.cfg.get("usage", {}) or {}
        if not ucfg.get("enabled", True):
            return
        usage.record(
            ucfg.get("stats_file", ""),
            reply.model or self.model,
            reply.prompt_tokens,
            reply.completion_tokens,
        )
