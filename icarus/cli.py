"""Icarus command line."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import caps as caps_mod
from . import config, ctxplan, detect as detect_mod, protocol, skillhub, skills as skills_mod
from .gguf import shape_of
from .interrupt import InputWatcher, NullWatcher
from .llm import LLMClient, LLMError
from .loop import Agent
from .render import Spinner, Style, bar, human_tokens
from .session import Session, list_sessions
from .swap import SwapClient
from .tools import builtin

# The "I" needs a full bottom bar. A narrow |_| stem under a top crossbar
# reads as a T, which made an earlier version of this say "Tcarus".
BANNER = r"""
 _____ ____    _    ____  _   _ ____
|_   _/ ___|  / \  |  _ \| | | / ___|
  | || |     / _ \ | |_) | | | \___ \
  | || |___ / ___ \|  _ <| |_| |___) |
 |___|\____|/_/   \_\_| \_\\___/|____/
"""


class Console:
    def __init__(self, cfg: Dict[str, Any]):
        ui = cfg.get("ui", {}) or {}
        self.style = Style(color=bool(ui.get("color", True)))
        self.spinner = Spinner(self.style, enabled=bool(ui.get("spinner", True)))
        self.show_tool_output = bool(ui.get("show_tool_output", True))
        self._streaming = False
        # Set once the REPL starts; every write clears the pinned type-ahead
        # line first and redraws it after, so streamed output and a half-typed
        # prompt never overwrite each other.
        self.watcher: Any = NullWatcher()

    def _clear(self) -> None:
        self.watcher.clear_line()

    def _restore(self) -> None:
        self.watcher.render()

    def say(self, s: str = "") -> None:
        self.spinner.stop()
        self._clear()
        print(s)
        self._restore()

    def status(self, s: str) -> None:
        self.spinner.stop()
        self._clear()
        print(self.style.grey(f"  · {s}"))
        self._restore()

    def error(self, s: str) -> None:
        self.spinner.stop()
        self._clear()
        print(self.style.red(f"  ✗ {s}"))
        self._restore()

    def token(self, s: str) -> None:
        if not self._streaming:
            self.spinner.stop()
            self._streaming = True
        self._clear()
        sys.stdout.write(s)
        sys.stdout.flush()
        self._restore()

    def end_stream(self) -> None:
        if self._streaming:
            self._clear()
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._streaming = False
            self._restore()


def build_agent(
    cfg: Dict[str, Any],
    console: Console,
    session: Session,
    workdir: Path,
    model: Optional[str] = None,
    probe: bool = True,
) -> Agent:
    mcfg = cfg["model"]
    swap = SwapClient(
        base_url=mcfg["base_url"],
        config_path=(cfg.get("llama_swap", {}) or {}).get("config_path", ""),
    )
    client = LLMClient(
        base_url=mcfg["base_url"],
        api_key=mcfg.get("api_key", "not-needed"),
        request_timeout=int((cfg.get("llama_swap", {}) or {}).get("request_timeout", 900)),
        disable_thinking=bool(mcfg.get("disable_thinking", True)),
    )

    chosen = model or session.model or mcfg.get("default") or ""
    if not chosen:
        # Follow whatever llama-swap already has resident — that's the zero-cost
        # choice, since anything else pays a cold load.
        loaded = swap.loaded_models()
        available = swap.list_models()
        chosen = loaded[0] if loaded else (available[0] if available else "")
    if not chosen:
        console.error(
            f"No models available from {mcfg['base_url']}. Is llama-swap running?"
        )
        sys.exit(2)

    ctx = swap.context_window(chosen)
    store = caps_mod.CapsStore()

    if probe:
        cached = store.get(chosen)
        if cached is None:
            console.spinner.start(f"probing {chosen} (first use — may load the model)")
        model_caps = caps_mod.ensure(
            store, client, chosen, ctx=ctx,
            log=lambda s: console.spinner.update(f"{s} — {chosen}"),
        )
        console.spinner.stop()
    else:
        model_caps = store.get(chosen) or caps_mod.Caps(model=chosen, ctx=ctx)

    found_skills = skills_mod.discover()
    tcfg = cfg.get("tools", {}) or {}
    patterns = [re.compile(p) for p in (tcfg.get("require_approval") or [])]
    approved_patterns: set[str] = set()

    def needs_approval(command: str) -> bool:
        for rx in patterns:
            if rx.search(command):
                return rx.pattern not in approved_patterns
        return False

    def approve(command: str) -> bool:
        console.spinner.stop()
        console.say()
        console.say(console.style.yellow("  ⚠ This command needs your approval:"))
        console.say(console.style.bold(f"    {command}"))
        # input() runs while the turn is in flight, i.e. while the raw-mode
        # watcher thread is also reading stdin. Two readers on one tty fd steal
        # each other's keystrokes, so stop the watcher (restores cooked mode and
        # blocking stdin) for the duration of the prompt and resume after.
        watcher = console.watcher
        was_running = bool(
            watcher is not None
            and watcher.enabled
            and watcher._thread is not None
            and watcher._thread.is_alive()
        )
        if was_running:
            watcher.stop()
        try:
            ans = input(console.style.yellow("    Run it? [y]es / [n]o / [a]lways this kind: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            if was_running:
                watcher.start()
        if ans.startswith("a"):
            for rx in patterns:
                if rx.search(command):
                    approved_patterns.add(rx.pattern)
            return True
        return ans.startswith("y")

    registry = builtin.build(
        workdir=workdir,
        terminal_timeout=int(tcfg.get("terminal_timeout", 120)),
        max_output_chars=int(tcfg.get("max_output_chars", 30000)),
        approve=approve,
        needs_approval=needs_approval,
        todo_state=session.todos,
        memory_dir=config.MEMORY_DIR,
        # Esc during a tool call kills the command, not just the turn.
        should_abort=lambda: console.watcher.aborted(),
    )
    skills_mod.register(registry, found_skills)

    session.note_model(chosen)
    agent = Agent(
        client=client,
        swap=swap,
        registry=registry,
        session=session,
        caps_store=store,
        model=chosen,
        ctx=ctx,
        caps=model_caps,
        cfg=cfg,
        workdir=workdir,
        skills_catalog=skills_mod.catalog_lines(found_skills),
        todo_ref=session.todos,
    )
    _wire_ui(agent, console)
    return agent


def _wire_ui(agent: Agent, console: Console) -> None:
    st = console.style

    def on_text(chunk: str) -> None:
        console.token(chunk)

    def on_tool_start(name: str, args: Dict[str, Any]) -> None:
        console.end_stream()
        console.spinner.stop()
        detail = ""
        for key in ("command", "path", "pattern", "name", "action", "item"):
            if key in args and isinstance(args[key], str):
                detail = args[key].replace("\n", " ")[:110]
                break
        console.say(f"  {st.magenta('⚒')} {st.bold(name)} {st.grey(detail)}")
        console.spinner.start(f"running {name}")

    def on_tool_end(name: str, result: Any) -> None:
        console.spinner.stop()
        mark = st.green("✓") if getattr(result, "ok", True) else st.red("✗")
        console.say(f"    {mark} {st.grey(getattr(result, 'display', '') or 'done')}")
        if console.show_tool_output and getattr(result, "content", ""):
            body = result.content.strip()
            lines = body.splitlines()
            head = lines[:12]
            for l in head:
                console.say(st.grey(f"    │ {l[:160]}"))
            if len(lines) > len(head):
                console.say(st.grey(f"    │ … {len(lines) - len(head)} more lines"))
        console.spinner.start("thinking")

    def on_status(s: str) -> None:
        console.end_stream()
        console.status(s)

    agent.on_text = on_text
    agent.on_tool_start = on_tool_start
    agent.on_tool_end = on_tool_end
    agent.on_status = on_status


# ---------------------------------------------------------------------------
# slash commands
# ---------------------------------------------------------------------------

HELP = """
Slash commands
  /help                  this list
  /model                 pick a model interactively
  /model <name|number>   switch model in flight (conversation is kept)
  /models                list every model llama-swap can serve
  /running               what is resident in VRAM right now
  /unload                unload all models, freeing VRAM
  /think [on|off|auto]   reasoning output: off is fastest, auto follows the model
  /context               how much context this conversation is using
  /ctx [N|max|auto|reset] resize the model's context window (reloads it)
  /profile [name]        task shape: quick / balanced / code / deep
  /compact               summarize history now to reclaim context
  /tools                 list available tools
  /skills                list installed skills
  /skills search <topic> find skills in trusted public repos
  /skills install <name> install one (scanned, then you confirm)
  /skills remove <name>  uninstall
  /skills update         check installed skills for upstream changes
  /skills sources        which repositories are trusted
  /caps [--reprobe]      what Icarus detected about the current model
  /new                   start a fresh conversation
  /sessions              recent conversations
  /resume <id>           reopen a conversation
  /cost                  tokens used this session
  /cwd [path]            show or change the working directory
  /quit                  leave (Ctrl-D also works)

While the model is working
  type + Enter           queue a message; injected at the next step
  Esc                    interrupt the turn now
""".strip()


def cmd_models(agent: Agent, console: Console) -> None:
    st = console.style
    console.spinner.start("querying llama-swap")
    inv = agent.swap.inventory()
    console.spinner.stop()
    if not inv:
        console.error("llama-swap returned no models.")
        return
    console.say()
    console.say(st.bold(f"  {len(inv)} models on {agent.swap.root}"))
    for i, m in enumerate(inv, 1):
        marker = st.green("●") if m.loaded else st.grey("○")
        active = st.cyan(" ← active") if m.id == agent.model else ""
        gpu = st.grey(f" gpu{m.gpus}") if m.gpus else ""
        console.say(
            f"  {marker} {i:2d}. {m.id:<52} {st.grey(f'{m.ctx // 1024}K ctx')}{gpu}{active}"
        )
    console.say(st.grey("\n  ● resident   ○ not loaded   — /model <number|name> to switch"))


def cmd_model(agent: Agent, console: Console, arg: str) -> None:
    """In-flight model switch, Claude Code style."""
    st = console.style
    inv = agent.swap.inventory()
    if not inv:
        console.error("llama-swap returned no models.")
        return

    target = ""
    if not arg:
        # Interactive picker.
        cmd_models(agent, console)
        console.say()
        try:
            choice = input(st.cyan("  model> ")).strip()
        except (EOFError, KeyboardInterrupt):
            console.say()
            return
        if not choice:
            return
        arg = choice

    if arg.isdigit():
        idx = int(arg)
        if not (1 <= idx <= len(inv)):
            console.error(f"No model #{idx}; there are {len(inv)}.")
            return
        target = inv[idx - 1].id
    else:
        target = agent.swap.resolve(arg) or ""
        if not target:
            cands = agent.swap.candidates(arg)
            if not cands:
                console.error(f"No model matches {arg!r}.")
            else:
                console.error(f"{arg!r} is ambiguous — {len(cands)} matches:")
                for c in cands[:12]:
                    console.say(st.grey(f"    {c}"))
            return

    if target == agent.model:
        console.status(f"already on {target}")
        return

    unload = bool((agent.cfg.get("llama_swap", {}) or {}).get("unload_before_switch", True))
    console.spinner.start(f"switching to {target} (cold load can take a while)")
    try:
        summary = agent.switch_model(target, unload_previous=unload)
    except LLMError as e:
        console.spinner.stop()
        console.error(f"switch failed: {e}")
        return
    console.spinner.stop()
    console.say(f"  {st.green('✓')} {summary}")


def cmd_think(agent: Agent, console: Console, arg: str) -> None:
    st = console.style
    a = (arg or "").strip().lower()
    if a in ("on", "true", "1", "yes"):
        agent.thinking_override = True
    elif a in ("off", "false", "0", "no"):
        agent.thinking_override = False
    elif a in ("auto", "default", ""):
        if not a:
            # Bare /think reports state rather than silently toggling.
            mode = ("auto" if agent.thinking_override is None
                    else ("on" if agent.thinking_override else "off"))
            forced = "" if agent.caps.honors_thinking_flag else st.yellow(
                "  (forced ON: this model returns empty output with thinking disabled)")
            console.say(f"  thinking: {st.bold(mode)} → effective: "
                        f"{st.bold('on' if agent.thinking_enabled else 'off')}{forced}")
            return
        agent.thinking_override = None
    else:
        console.error("usage: /think [on|off|auto]")
        return
    agent._sync_thinking()
    eff = "on" if agent.thinking_enabled else "off"
    note = ""
    if not agent.caps.honors_thinking_flag and agent.thinking_enabled:
        note = st.yellow("  (this model needs it — it returns nothing otherwise)")
    console.say(f"  {st.green('✓')} thinking {st.bold(eff)}{note}")


def cmd_context(agent: Agent, console: Console) -> None:
    st = console.style
    budget = agent._budget()
    msgs = [{"role": "system", "content": agent.system_prompt()}] + agent.session.messages
    used = budget.estimate(msgs)
    frac = used / budget.usable if budget.usable else 0
    color = st.green if frac < 0.6 else (st.yellow if frac < 0.85 else st.red)
    console.say()
    console.say(f"  {st.bold(agent.model)}")
    console.say(f"  {color(bar(frac))} {human_tokens(used)} / {human_tokens(budget.usable)} usable "
                f"({frac * 100:.0f}%)")
    console.say(st.grey(f"  window {agent.ctx:,} · reserved for output "
                        f"{budget.reserve_output:,} · compacts at {budget.threshold * 100:.0f}%"))
    console.say(st.grey(f"  {len(agent.session.messages)} messages · "
                        f"~{budget.chars_per_token:.2f} chars/token (calibrated)"))


def cmd_compact(agent: Agent, console: Console) -> None:
    from .context import compact as do_compact
    budget = agent._budget()
    before = budget.estimate(agent.session.messages)
    # Force a pass by pretending the ceiling is much lower.
    budget.threshold = 0.35
    agent.session.messages, note = do_compact(
        agent.session.messages, budget, summarize=agent._summarize
    )
    after = agent._budget().estimate(agent.session.messages)
    agent.session.save()
    if note:
        console.say(f"  {console.style.green('✓')} {note} "
                    f"({human_tokens(before)} → {human_tokens(after)})")
    else:
        console.status("nothing to compact")


def cmd_caps(agent: Agent, console: Console, arg: str) -> None:
    st = console.style
    if "--reprobe" in (arg or ""):
        console.spinner.start(f"re-probing {agent.model}")
        agent.caps = caps_mod.ensure(
            agent.caps_store, agent.client, agent.model, ctx=agent.ctx, force=True,
            log=lambda s: console.spinner.update(s),
        )
        agent._sync_thinking()
        console.spinner.stop()
    c = agent.caps
    console.say()
    console.say(f"  {st.bold(c.model)}")
    console.say(f"    tool calling   {st.green('native') if c.native_tools else st.yellow('text protocol')}")
    console.say(f"    reasoning      {'yes' if c.reasoning else 'no'}")
    console.say(f"    thinking flag  {'honored' if c.honors_thinking_flag else st.yellow('ignored — kept on')}")
    console.say(f"    context        {c.ctx:,}")
    if c.note:
        console.say(st.grey(f"    note           {c.note}"))


def _swapcfg(agent: Agent) -> ctxplan.SwapConfig:
    return ctxplan.SwapConfig(
        (agent.cfg.get("llama_swap", {}) or {}).get("config_path", "")
    )


def _calibrate_if_loaded(agent: Agent) -> None:
    """Learn this model's real weight footprint whenever it happens to be up."""
    try:
        if agent.model not in agent.swap.loaded_models():
            return
        cfg = _swapcfg(agent)
        shape = shape_of(cfg.gguf_path(agent.model))
        if shape:
            ctxplan.record_measurement(
                agent.model, cfg.ctx(agent.model),
                ctxplan.llama_server_vram_mib(), shape,
            )
    except Exception:
        pass


def cmd_ctx(agent: Agent, console: Console, arg: str) -> None:
    """/ctx [N|max|auto|reset] — inspect or change the server-side window."""
    st = console.style
    cfg = _swapcfg(agent)
    if not cfg.path.is_file():
        console.error(f"llama-swap config not found at {cfg.path}; "
                      "set llama_swap.config_path")
        return

    model = agent.model
    _calibrate_if_loaded(agent)
    shape = shape_of(cfg.gguf_path(model))
    arg = (arg or "").strip().lower()

    # ---- report ---------------------------------------------------------
    if not arg:
        p = ctxplan.plan(cfg, model, cfg.ctx(model), shape=shape)
        budget = agent._budget()
        msgs = [{"role": "system", "content": agent.system_prompt()}] + agent.session.messages
        used = budget.estimate(msgs)
        console.say()
        console.say(f"  {st.bold(model)}")
        console.say(f"  window        {st.bold(f'{p.current_ctx:,}')} tokens"
                    + (f"   (trained for {p.trained_ctx:,})" if p.trained_ctx else ""))
        console.say(f"  in use now    {human_tokens(used)} / {human_tokens(budget.usable)} usable")
        if shape:
            per = shape.kv_bytes_per_token() / 1024
            console.say(f"  KV cost       {per:.0f} KB/token"
                        + ("  (sliding-window: most layers are capped)" if shape.uses_swa else ""))
            console.say(f"  KV at {p.current_ctx:,}   {shape.kv_bytes(p.current_ctx)/ctxplan.GB:.2f} GB"
                        f"  + {p.weight_gb:.1f} GB weights"
                        f"{st.grey('  (measured)') if p.calibrated else st.grey('  (estimated from file size)')}")
        console.say(f"  largest safe  {st.bold(f'{p.max_safe_ctx:,}')} on GPU "
                    f"{','.join(map(str, cfg.placement(model).gpus))}")
        for w in p.warnings:
            console.say(st.grey(f"    · {w}"))
        console.say(st.grey("\n  /ctx <N> · /ctx max · /ctx auto · /ctx reset · /profile"))
        return

    # ---- resolve the target --------------------------------------------
    probe = ctxplan.plan(cfg, model, cfg.ctx(model), shape=shape)
    if arg == "reset":
        ctxplan.write_override(cfg.path, model, None)
        orig = ctxplan.original_ctx(model)
        if orig and orig != probe.current_ctx:
            ctxplan.apply(cfg, model, orig, persist=False)
            ctxplan.forget_original(model)
            agent.swap.unload_all()
            agent.swap.wait_unloaded()
            agent.ctx = orig
            console.say(f"  {st.green('✓')} {model} restored to its default "
                        f"{orig:,} context; override cleared")
        else:
            ctxplan.forget_original(model)
            console.say(f"  {st.green('✓')} override cleared for {model} "
                        f"(already at its default {probe.current_ctx:,})")
        return
    if arg == "max":
        target = probe.max_safe_ctx
    elif arg == "auto":
        budget = agent._budget()
        msgs = [{"role": "system", "content": agent.system_prompt()}] + agent.session.messages
        target = ctxplan.suggest_for_usage(
            budget.estimate(msgs), probe.current_ctx, probe.max_safe_ctx)
    else:
        try:
            target = int(arg.replace("k", "000").replace(",", ""))
            if target < 1024:          # allow "/ctx 64" as shorthand for 64K
                target *= 1024
        except ValueError:
            console.error("usage: /ctx [N|max|auto|reset]")
            return

    if target == probe.current_ctx:
        console.status(f"already at {target:,}")
        return

    p = ctxplan.plan(cfg, model, target, shape=shape)
    console.say()
    console.say(f"  {model}: {p.current_ctx:,} → {st.bold(f'{target:,}')}")
    console.say(f"  KV {p.kv_gb:.2f} GB + weights {p.weight_gb:.1f} GB "
                f"vs {p.free_gb:.1f} GB budget")
    for w in p.warnings:
        console.say(st.yellow(f"    ⚠ {w}"))
    if not p.feasible:
        console.error(p.reason or "does not fit")
        return

    ok, msg = ctxplan.apply(cfg, model, target)
    if not ok:
        console.error(msg)
        return
    # llama-swap watches the config; unload so the next call relaunches with
    # it, and wait for the teardown — requests during it return HTTP 500.
    agent.swap.unload_all()
    agent.swap.wait_unloaded()
    agent.ctx = target
    console.say(f"  {st.green('✓')} {msg} — takes effect on the next request "
                f"{st.grey('(the model reloads)')}")


def cmd_profile(agent: Agent, console: Console, arg: str) -> None:
    """/profile [name] — switch the task shape (context + budget knobs)."""
    st = console.style
    acfg = agent.cfg.get("agent", {}) or {}
    profiles: Dict[str, Any] = acfg.get("profiles", {}) or {}
    name = (arg or "").strip().lower()

    if not name:
        console.say()
        cur = acfg.get("profile", "balanced")
        for n, spec in profiles.items():
            mark = st.cyan(" ← active") if n == cur else ""
            ctxv = spec.get("ctx")
            ctxs = "largest that fits" if ctxv == "max" else (f"{int(ctxv):,}" if ctxv else "unchanged")
            console.say(f"  {st.bold(n):<22} {st.grey(f'ctx {ctxs}')}{mark}")
            console.say(st.grey(f"      {spec.get('note','')}"))
        console.say(st.grey("\n  /profile <name> to switch"))
        return

    if name not in profiles:
        console.error(f"No profile {name!r}. Available: {', '.join(profiles)}")
        return

    spec = profiles[name]
    acfg["profile"] = name
    # Client-side knobs apply immediately.
    for key in ("max_iterations", "context_threshold"):
        if key in spec:
            acfg[key] = spec[key]
    if "max_tokens" in spec:
        agent.cfg.setdefault("model", {})["max_tokens"] = spec["max_tokens"]
    if "max_output_chars" in spec:
        agent.cfg.setdefault("tools", {})["max_output_chars"] = spec["max_output_chars"]

    console.say(f"  {st.green('✓')} profile {st.bold(name)} — "
                f"{spec.get('max_iterations','?')} max steps, "
                f"{spec.get('max_tokens','?')} max output tokens")

    want = spec.get("ctx")
    if want:
        cmd_ctx(agent, console, "max" if want == "max" else str(want))


def cmd_skills(agent: Agent, console: Console, arg: str) -> None:
    """/skills [search|install|remove|update|sources] — local list by default."""
    st = console.style
    parts = (arg or "").split(maxsplit=1)
    sub = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("", "list"):
        found = skills_mod.discover()
        lock = skillhub.installed()
        console.say()
        if not found:
            console.say(st.grey("  No skills installed. Try /skills search <topic>"))
            return
        by_ref = {v.get("name"): k for k, v in lock.items()}
        for n in sorted(found):
            src = by_ref.get(n) or found[n].source
            console.say(f"  {st.bold(n):<38} {st.grey(f'[{src}]')} "
                        f"{st.grey(found[n].description[:70])}")
        console.say(st.grey(f"\n  {len(found)} skills · /skills search <topic> for more"))
        return

    if sub == "sources":
        console.say()
        console.say(st.bold("  Trusted repositories"))
        for r in skillhub.TRUSTED_REPOS:
            console.say(f"    github.com/{r}")
        console.say(st.grey("\n  Only these are searched. Installing is the only\n"
                            "  time Icarus touches the network, and only on your command."))
        return

    if sub == "search":
        if not rest:
            console.error("usage: /skills search <topic>")
            return
        console.spinner.start("searching trusted repositories")
        try:
            index = skillhub.load_index()
        except skillhub.HubError as e:
            console.spinner.stop()
            console.error(str(e))
            return
        console.spinner.stop()
        hits = skillhub.search(rest, index)
        if not hits:
            console.say(st.grey(f"  No skill matches {rest!r} in "
                                f"{len(skillhub.TRUSTED_REPOS)} trusted repos."))
            return
        lock = skillhub.installed()
        console.say()
        for s in hits[:25]:
            mark = st.green(" ✓ installed") if s.ref in lock else ""
            console.say(f"  {st.bold(s.name):<30} {st.grey(s.repo)}{mark}")
            if s.description:
                console.say(st.grey(f"      {s.description[:110]}"))
        console.say(st.grey(f"\n  {len(hits)} match(es) · /skills install <name>"))
        return

    if sub == "install":
        if not rest:
            console.error("usage: /skills install <name>")
            return
        console.spinner.start("resolving")
        try:
            index = skillhub.load_index()
        except skillhub.HubError as e:
            console.spinner.stop()
            console.error(str(e))
            return
        hits = skillhub.search(rest, index)
        exact = [s for s in hits if s.name.lower() == rest.lower()]
        pool = exact or hits
        if not pool:
            console.spinner.stop()
            console.error(f"No skill named {rest!r} in the trusted repositories.")
            return
        if len(pool) > 1:
            console.spinner.stop()
            console.error(f"{rest!r} matches {len(pool)} skills — be more specific:")
            for s in pool[:12]:
                console.say(st.grey(f"    {s.name}  ({s.repo})"))
            return

        target = pool[0]
        console.spinner.update(f"downloading {target.name}")
        try:
            files = skillhub.fetch(target)
        except skillhub.HubError as e:
            console.spinner.stop()
            console.error(str(e))
            return
        console.spinner.stop()

        findings = skillhub.scan(files)
        v = skillhub.verdict(findings)
        console.say()
        console.say(f"  {st.bold(target.name)}  {st.grey(target.repo)}")
        if target.description:
            console.say(st.grey(f"  {target.description[:150]}"))
        console.say(st.grey(f"  {len(files)} file(s), {sum(len(t) for t in files.values()):,} bytes"))

        if findings:
            colour = {"critical": st.red, "high": st.yellow, "medium": st.grey}
            console.say()
            console.say(f"  {st.bold('Safety scan')}: {colour.get(v, st.grey)(v)}")
            for f in findings[:10]:
                c = colour.get(f.severity, st.grey)
                console.say(f"    {c(f.severity):<10} {f.reason}")
                console.say(st.grey(f"      {f.file}:{f.line}  {f.excerpt[:100]}"))
        else:
            console.say(f"  {st.bold('Safety scan')}: {st.green('clean')}")

        prompt = "  Install? [y/N]: "
        if v == "dangerous":
            console.say(st.red("\n  This skill instructs an agent to execute fetched code."))
            prompt = st.red("  Install anyway? [y/N]: ")
        try:
            console.watcher.clear_line()
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.say()
            return
        if not ans.startswith("y"):
            console.status("not installed")
            return

        dest = skillhub.install(target, files)
        console.say(f"  {st.green('✓')} installed to {dest}")
        _refresh_skills(agent, console)
        return

    if sub == "remove":
        if not rest:
            console.error("usage: /skills remove <name>")
            return
        ref = skillhub.remove(rest)
        if ref:
            console.say(f"  {st.green('✓')} removed {ref}")
            _refresh_skills(agent, console)
        else:
            console.error(f"{rest!r} is not an installed hub skill "
                          f"(local/hermes skills are not managed here).")
        return

    if sub == "update":
        console.spinner.start("checking for updates")
        try:
            index = skillhub.load_index(refresh=True)
        except skillhub.HubError as e:
            console.spinner.stop()
            console.error(str(e))
            return
        console.spinner.stop()
        stale = skillhub.outdated(index)
        if not stale:
            console.say(f"  {st.green('✓')} all installed skills are current")
            return
        console.say()
        for ref, old, new in stale:
            console.say(f"  {st.yellow('↑')} {ref}  {st.grey(f'{old} → {new}')}")
        console.say(st.grey("\n  re-run /skills install <name> to take the update"))
        return

    console.error(f"Unknown: /skills {sub} — try list, search, install, remove, update, sources")


def _refresh_skills(agent: Agent, console: Console) -> None:
    """Re-discover skills so a freshly installed one is usable immediately."""
    found = skills_mod.discover()
    skills_mod.register(agent.registry, found)
    agent.skills_catalog = skills_mod.catalog_lines(found)
    console.status(f"{len(found)} skills now available to the model")


def cmd_running(agent: Agent, console: Console) -> None:
    st = console.style
    rows = agent.swap.running()
    if not rows:
        console.say(st.grey("  Nothing resident — the next request will cold-load."))
        return
    console.say()
    for r in rows:
        console.say(f"  {st.green('●')} {st.bold(r.get('model',''))} "
                    f"{st.grey(r.get('state',''))} {st.grey(r.get('proxy',''))}")


def cmd_sessions(console: Console, current: Session) -> None:
    st = console.style
    rows = list_sessions()
    if not rows:
        console.say(st.grey("  No saved sessions."))
        return
    console.say()
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r["updated"]))
        here = st.cyan(" ← current") if r["id"] == current.id else ""
        counts = "{}t {}".format(r["turns"], human_tokens(r["tokens"]))
        console.say("  {}  {}  {}  {}{}".format(
            st.bold(r["id"]), st.grey(when), st.grey(counts), r["title"][:52], here))


def cmd_cost(agent: Agent, console: Console) -> None:
    st = console.style
    s = agent.session
    console.say()
    console.say(f"  session {st.bold(s.id)} · {s.turns} turns")
    console.say(f"  prompt      {human_tokens(s.prompt_tokens):>8}")
    console.say(f"  completion  {human_tokens(s.completion_tokens):>8}")
    console.say(f"  total       {human_tokens(s.prompt_tokens + s.completion_tokens):>8}")
    if s.model_history:
        console.say(st.grey("  models used:"))
        for ts, old, new in s.model_history:
            console.say(st.grey(f"    {time.strftime('%H:%M', time.localtime(ts))} "
                                f"{old or '(start)'} → {new}"))
    console.say(st.grey("  $0.00 — local inference"))


def handle_slash(line: str, agent: Agent, console: Console, state: Dict[str, Any]) -> bool:
    """Returns True if the input was a command (and was handled)."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""
    st = console.style

    if cmd in ("help", "?"):
        console.say(HELP)
    elif cmd == "models":
        cmd_models(agent, console)
    elif cmd == "model":
        cmd_model(agent, console, arg)
    elif cmd == "running":
        cmd_running(agent, console)
    elif cmd == "unload":
        console.spinner.start("unloading")
        ok = agent.swap.unload_all()
        console.spinner.stop()
        console.say(f"  {st.green('✓') if ok else st.red('✗')} "
                    f"{'VRAM freed' if ok else 'unload failed'}")
    elif cmd == "think":
        cmd_think(agent, console, arg)
    elif cmd == "context":
        cmd_context(agent, console)
    elif cmd == "ctx":
        cmd_ctx(agent, console, arg)
    elif cmd == "profile":
        cmd_profile(agent, console, arg)
    elif cmd == "compact":
        cmd_compact(agent, console)
    elif cmd == "tools":
        console.say()
        for t in agent._tools():
            flag = st.yellow(" ✎") if t.mutates else "  "
            console.say(f"  {flag} {st.bold(t.name):<28} {st.grey(t.description[:90])}")
    elif cmd == "skills":
        cmd_skills(agent, console, arg)
    elif cmd == "caps":
        cmd_caps(agent, console, arg)
    elif cmd == "new":
        fresh = Session(workdir=str(agent.workdir), model=agent.model)
        agent.adopt_session(fresh)
        state["session"] = fresh
        console.say(f"  {st.green('✓')} new session {st.bold(fresh.id)}")
    elif cmd == "sessions":
        cmd_sessions(console, agent.session)
    elif cmd == "resume":
        s = Session.load(arg) if arg else Session.latest()
        if not s:
            console.error(f"No session matching {arg!r}.")
        else:
            agent.adopt_session(s)
            state["session"] = s
            console.say(f"  {st.green('✓')} resumed {st.bold(s.id)} "
                        f"({len(s.messages)} messages, {s.turns} turns)")
            if s.model and s.model != agent.model:
                console.status(f"session was on {s.model}; use /model {s.model} to match")
    elif cmd == "cost":
        cmd_cost(agent, console)
    elif cmd == "cwd":
        if arg:
            p = Path(os.path.expanduser(arg)).resolve()
            if not p.is_dir():
                console.error(f"Not a directory: {p}")
            else:
                agent.workdir = p
                console.say(f"  {st.green('✓')} working directory: {p}")
        else:
            console.say(f"  {agent.workdir}")
    elif cmd in ("quit", "exit", "q"):
        state["quit"] = True
    else:
        console.error(f"Unknown command /{cmd} — try /help")
    return True


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def repl(agent: Agent, console: Console) -> None:
    st = console.style
    state: Dict[str, Any] = {"quit": False, "session": agent.session}

    ui = (agent.cfg.get("ui", {}) or {})
    watcher = InputWatcher(st, enabled=bool(ui.get("interrupt", True)))
    console.watcher = watcher
    agent.watcher = watcher
    # Spinner and typed input both want the bottom line; input wins.
    console.spinner.suppress = lambda: bool(getattr(watcher, '_buf', ''))

    try:
        import readline  # noqa: F401  — arrow keys and history for free
        hist = config.HOME / "history"
        try:
            readline.read_history_file(str(hist))
        except OSError:
            pass
        readline.set_history_length(2000)
    except ImportError:
        hist = None

    console.say(st.cyan(BANNER))
    console.say(st.grey("  a fully local agent · nothing leaves this machine\n"))
    mode = "native tools" if agent.caps.native_tools else "text-protocol tools"
    think = "on" if agent.thinking_enabled else "off"
    meta = "{:,} ctx · {} · thinking {}".format(agent.ctx, mode, think)
    console.say("  {}  {}".format(st.bold(agent.model), st.grey(meta)))
    console.say(st.grey(f"  {agent.workdir}"))
    hint = ("  /help for commands · Ctrl-D to leave"
            if not watcher.enabled else
            "  /help for commands · while working: type to steer, Esc to interrupt")
    console.say(st.grey(hint + "\n"))

    while not state["quit"]:
        try:
            line = input(st.cyan("› ")).strip()
        except EOFError:
            console.say()
            break
        except KeyboardInterrupt:
            console.say(st.grey("  (Ctrl-D to leave)"))
            continue
        if not line:
            continue
        if handle_slash(line, agent, console, state):
            continue

        # Hand the terminal to the watcher for the duration of the turn: typed
        # lines become steering, Esc aborts. Restored before the next prompt.
        watcher.reset()
        console.spinner.start("thinking  " + st.grey("(type to steer · Esc to interrupt)"))
        try:
            with watcher:
                stats = agent.run_turn(line)
        except KeyboardInterrupt:
            console.spinner.stop()
            console.end_stream()
            console.say(st.yellow("  interrupted"))
            continue
        finally:
            console.spinner.stop()
        console.end_stream()

        tot = stats.prompt_tokens + stats.completion_tokens
        bits = [
            f"{stats.iterations} step{'s' if stats.iterations != 1 else ''}",
            f"{stats.tool_calls} tool call{'s' if stats.tool_calls != 1 else ''}",
            f"{human_tokens(tot)} tokens",
            f"{stats.elapsed:.1f}s",
        ]
        if stats.steered:
            bits.append(f"{stats.steered} steered")
        console.say(st.grey("  " + " · ".join(bits)) +
                    (st.yellow("  ⏹ interrupted") if stats.interrupted else ""))
        console.say()

    if hist:
        try:
            import readline
            readline.write_history_file(str(hist))
        except Exception:
            pass
    agent.session.save()


def _read_piped_stdin(has_argv_prompt: bool) -> str:
    """Read piped input without ever blocking on an idle stdin.

    `icarus "do X"` run from cron, systemd, or a nested shell inherits an open
    pipe that never sends EOF. A bare stdin.read() there hangs the process
    forever, so: never touch stdin when a prompt came in on argv, and even
    without one, only read if there is data actually waiting.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        import select
        import stat as _stat

        # A redirected file always has real content for us: `icarus "q" < f`.
        if _stat.S_ISREG(os.fstat(sys.stdin.fileno()).st_mode):
            return sys.stdin.read().strip()

        # With no argv prompt, piped input is the only possible instruction —
        # block for it the way `cat` would.
        if not has_argv_prompt:
            return sys.stdin.read().strip()

        # With an argv prompt, stdin is ambiguous: it could be `cmd | icarus
        # "question"`, or an fd this process merely inherited. Only real,
        # already-waiting data counts.
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        return sys.stdin.read().strip() if ready else ""
    except Exception:
        return ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="icarus",
        description="A fully local command-line agent for llama-swap.",
    )
    ap.add_argument("prompt", nargs="*", help="Run one turn and exit.")
    ap.add_argument("-m", "--model", help="Model to use (fragment is fine).")
    ap.add_argument("-C", "--cwd", default=".", help="Working directory for tools.")
    ap.add_argument("-r", "--resume", nargs="?", const="__latest__",
                    help="Resume a session (id, or bare for the most recent).")
    ap.add_argument("--think", choices=["on", "off", "auto"],
                    help="Reasoning output for this run.")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--models", action="store_true", help="List models and exit.")
    ap.add_argument("--detect", action="store_true",
                    help="Autodetect the local AI stack and exit.")
    ap.add_argument("--config", action="store_true", help="Print effective config and exit.")
    ap.add_argument("--version", action="version", version="icarus 1.0.0")
    args = ap.parse_args(argv)

    cfg = config.load()
    config.write_default_config()
    if args.no_color:
        cfg["ui"]["color"] = False
    if args.no_stream:
        cfg["agent"]["stream"] = False

    console = Console(cfg)

    if args.detect:
        stack = detect_mod.scan()
        print(detect_mod.render(stack))
        if stack.best:
            print(f"\nConfigured base_url: {cfg['model']['base_url']}")
            if stack.best.base_url != cfg["model"]["base_url"]:
                print("  (differs from what was detected — re-run install.sh to switch)")
        return 0 if stack.best else 1

    if args.config:
        import yaml as _y
        print(_y.safe_dump(cfg, sort_keys=False))
        return 0

    workdir = Path(os.path.expanduser(args.cwd)).resolve()
    if not workdir.is_dir():
        console.error(f"Not a directory: {workdir}")
        return 2

    if args.resume:
        session = (Session.latest() if args.resume == "__latest__"
                   else Session.load(args.resume))
        if not session:
            console.error(f"No session matching {args.resume!r}.")
            return 2
    else:
        session = Session(workdir=str(workdir))

    if args.models:
        swap = SwapClient(
            base_url=cfg["model"]["base_url"],
            config_path=(cfg.get("llama_swap", {}) or {}).get("config_path", ""),
        )
        inv = swap.inventory()
        if not inv:
            console.error(f"No models from {cfg['model']['base_url']}.")
            return 2
        for m in inv:
            mark = "●" if m.loaded else "○"
            print(f"{mark} {m.id:<52} {m.ctx // 1024}K ctx")
        return 0

    # A fragment on the command line resolves the same way /model does.
    model = args.model
    if model:
        probe_swap = SwapClient(
            base_url=cfg["model"]["base_url"],
            config_path=(cfg.get("llama_swap", {}) or {}).get("config_path", ""),
        )
        resolved = probe_swap.resolve(model)
        if not resolved:
            cands = probe_swap.candidates(model)
            console.error(
                f"{model!r} is ambiguous — {len(cands)} matches" if cands
                else f"No model matches {model!r}."
            )
            for c in cands[:12]:
                print(f"    {c}")
            return 2
        model = resolved

    try:
        agent = build_agent(cfg, console, session, workdir, model=model)
    except LLMError as e:
        console.error(str(e))
        return 2

    if args.think:
        agent.thinking_override = None if args.think == "auto" else (args.think == "on")
    agent._sync_thinking()

    # One-shot: argv prompt, or piped stdin.
    oneshot = " ".join(args.prompt).strip()
    piped = _read_piped_stdin(has_argv_prompt=bool(oneshot))
    if piped and oneshot:
        oneshot = f"{oneshot}\n\n{piped}"
    elif piped:
        oneshot = piped

    if oneshot:
        cfg["agent"]["stream"] = cfg["agent"]["stream"] and sys.stdout.isatty()
        if not sys.stdout.isatty():
            agent.on_text = None
            agent.on_tool_start = None
            agent.on_tool_end = None
            agent.on_status = None
        try:
            agent.run_turn(oneshot)
        except KeyboardInterrupt:
            return 130
        finally:
            console.spinner.stop()
        console.end_stream()
        if not sys.stdout.isatty():
            # Piped: emit just the final answer, so icarus composes in a shell.
            for m in reversed(agent.session.messages):
                if m.get("role") == "assistant" and (m.get("content") or "").strip():
                    print(m["content"].strip())
                    break
        return 0

    repl(agent, console)
    return 0
