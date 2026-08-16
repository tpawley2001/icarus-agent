"""Built-in tools — the hermes core set, pared to what a local model can drive.

Deliberately small. An 8B model handed forty tools picks the wrong one; the
same model handed nine picks well. Anything bigger belongs in a skill.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..context import trim_tool_output
from .registry import Registry, ToolResult

MAX_READ_BYTES = 400_000

# Directories that are pure noise for a code search: dependency trees, build
# output, package caches, VCS internals. Excluded from search_files/glob_files
# so a search rooted at $HOME (or any dir with a vendored tree under it) does
# not drown the model's small context in cache and site-package paths — the
# failure that made a single search flood a 32K window past its limit.
IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "bower_components", "vendor",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "venv", ".venv", "virtualenv", "site-packages", ".eggs",
    ".cache", ".bun", ".npm", ".pnpm-store", ".yarn",
    ".cargo", ".rustup", ".gradle", ".m2",
    "dist", "build", "target", ".next", ".nuxt", ".svelte-kit", ".parcel-cache",
    ".idea", ".vscode", ".terraform",
})


def _kill_group(proc) -> None:
    """SIGTERM then SIGKILL the whole process group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=3)
            return
        except Exception:
            continue


def build(
    *,
    workdir: Path,
    terminal_timeout: int = 120,
    max_output_chars: int = 30_000,
    approve: Optional[Callable[[str], bool]] = None,
    needs_approval: Optional[Callable[[str], bool]] = None,
    todo_state: Optional[List[Dict[str, Any]]] = None,
    memory_dir: Optional[Path] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Registry:
    reg = Registry()
    todos: List[Dict[str, Any]] = todo_state if todo_state is not None else []
    mem_dir = memory_dir or (Path.home() / ".icarus" / "memory")

    def resolve(p: str) -> Path:
        """Resolve a model-supplied path against the working directory."""
        q = Path(os.path.expanduser(str(p)))
        return q if q.is_absolute() else (workdir / q)

    # ---- terminal -------------------------------------------------------
    @reg.tool(
        "terminal",
        "Run a shell command and return its combined stdout+stderr. Use for "
        "anything the other tools don't cover: git, package managers, "
        "systemctl, curl against local services, etc.",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
                "cwd": {"type": "string", "description": "Working directory. Defaults to the session directory."},
                "timeout": {"type": "integer", "description": f"Seconds before the command is killed (default {terminal_timeout})."},
            },
            "required": ["command"],
        },
        mutates=True,
    )
    def terminal(command: str, cwd: str = "", timeout: int = 0) -> ToolResult:
        if needs_approval and needs_approval(command):
            if not approve or not approve(command):
                return ToolResult(
                    False,
                    "The user DENIED permission to run this command. Do not retry "
                    "it. Explain what you wanted to do, or choose another approach.",
                    "denied by user",
                )
        run_in = resolve(cwd) if cwd else workdir
        limit = timeout or terminal_timeout

        # Run in its own process group and poll, so Esc can kill a runaway
        # command instead of the user waiting out a `sleep 600`. The whole
        # group is signalled, otherwise `sh -c` dies and its child survives.
        try:
            proc = subprocess.Popen(
                command, shell=True, cwd=str(run_in),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
        except Exception as e:
            return ToolResult(False, f"Could not start command: {e}", "terminal: start failed")

        deadline = time.monotonic() + limit
        aborted = timed_out = False
        while proc.poll() is None:
            if should_abort is not None and should_abort():
                aborted = True
                break
            if time.monotonic() > deadline:
                timed_out = True
                break
            time.sleep(0.1)

        if aborted or timed_out:
            _kill_group(proc)
            try:
                out = proc.communicate(timeout=5)[0] or ""
            except Exception:
                out = ""
            out = trim_tool_output(out.strip(), max_output_chars)
            if aborted:
                return ToolResult(
                    False,
                    "The user interrupted this command; it was killed. Partial "
                    f"output before the kill:\n{out or '(none)'}",
                    "interrupted by user",
                )
            return ToolResult(
                False,
                f"Command exceeded {limit}s and was killed. If it is expected to "
                f"be slow, pass a larger `timeout`. Partial output:\n{out or '(none)'}",
                "timed out",
            )

        out = trim_tool_output((proc.communicate()[0] or "").strip(), max_output_chars)
        rc = proc.returncode
        status = "ok" if rc == 0 else f"exit {rc}"
        body = out if out else "(no output)"
        if rc != 0:
            body = f"exit code {rc}\n{body}"
        return ToolResult(rc == 0, body, f"terminal: {status}")

    # ---- files ----------------------------------------------------------
    @reg.tool(
        "read_file",
        "Read a text file. Returns numbered lines so you can refer to them and "
        "edit precisely.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-based first line to return. Omit to start at the top."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum lines to return (default 400). Omit this "
                                   "to read the whole file — only set it for files "
                                   "you already know are very large.",
                },
            },
            "required": ["path"],
        },
    )
    def read_file(path: str, offset: int = 1, limit: int = 400) -> ToolResult:
        p = resolve(path)
        if not p.exists():
            return ToolResult(False, f"No such file: {p}", "read_file: missing")
        if p.is_dir():
            return ToolResult(False, f"{p} is a directory — use list_dir.", "read_file: is a dir")
        if p.stat().st_size > MAX_READ_BYTES:
            return ToolResult(
                False,
                f"{p} is {p.stat().st_size:,} bytes — too large to read whole. "
                "Use search_files, or read_file with offset/limit.",
                "read_file: too large",
            )
        try:
            lines = p.read_text(errors="replace").splitlines()
        except Exception as e:
            return ToolResult(False, f"Cannot read {p}: {e}", "read_file: error")
        start = max(1, int(offset or 1))
        chunk = lines[start - 1 : start - 1 + int(limit or 400)]
        if not chunk:
            return ToolResult(True, f"(no lines at offset {start}; file has {len(lines)})", "read_file: empty range")
        header = f"{p} — {len(lines)} lines total\n"
        body = "\n".join(f"{start + i:6d}\t{l}" for i, l in enumerate(chunk))
        # Always state the true length. Without it a small model that passed a
        # tiny `limit` concludes "at least N lines" and stops, instead of
        # either answering or reading on.
        shown_to = start - 1 + len(chunk)
        more = ""
        if shown_to < len(lines):
            more = (f"\n\n[showing lines {start}-{shown_to} of {len(lines)}; "
                    f"{len(lines) - shown_to} not shown — re-read with "
                    f"offset={shown_to + 1} to continue]")
        return ToolResult(
            True,
            header + trim_tool_output(body, max_output_chars) + more,
            f"read_file: {p.name} ({len(chunk)}/{len(lines)} lines)",
        )

    @reg.tool(
        "write_file",
        "Create a file or overwrite it completely. For a small change to an "
        "existing file prefer edit_file, which is safer.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        mutates=True,
    )
    def write_file(path: str, content: str) -> ToolResult:
        p = resolve(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            existed = p.exists()
            p.write_text(content)
        except Exception as e:
            return ToolResult(False, f"Cannot write {p}: {e}", "write_file: error")
        verb = "Overwrote" if existed else "Created"
        return ToolResult(True, f"{verb} {p} ({len(content):,} bytes, {content.count(chr(10)) + 1} lines).",
                          f"write_file: {p.name}")

    @reg.tool(
        "edit_file",
        "Replace an exact string in a file. `old` must appear exactly once "
        "unless replace_all is true. Include surrounding lines to make it unique.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact text to find, including indentation."},
                "new": {"type": "string", "description": "Replacement text."},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old", "new"],
        },
        mutates=True,
    )
    def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> ToolResult:
        p = resolve(path)
        if not p.exists():
            return ToolResult(False, f"No such file: {p}", "edit_file: missing")
        text = p.read_text(errors="replace")
        n = text.count(old)
        if n == 0:
            return ToolResult(
                False,
                f"That exact text does not appear in {p}. Read the file again — "
                "whitespace and indentation must match byte for byte.",
                "edit_file: no match",
            )
        if n > 1 and not replace_all:
            return ToolResult(
                False,
                f"That text appears {n} times in {p}. Add surrounding context to "
                "make it unique, or pass replace_all=true.",
                f"edit_file: {n} matches",
            )
        p.write_text(text.replace(old, new))
        return ToolResult(True, f"Edited {p} ({n} replacement{'s' if n > 1 else ''}).",
                          f"edit_file: {p.name}")

    @reg.tool(
        "list_dir",
        "List a directory's contents with sizes.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "all": {"type": "boolean", "description": "Include dotfiles."}},
            "required": [],
        },
    )
    def list_dir(path: str = ".", all: bool = False) -> ToolResult:
        p = resolve(path)
        if not p.is_dir():
            return ToolResult(False, f"Not a directory: {p}", "list_dir: not a dir")
        rows = []
        for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if not all and e.name.startswith("."):
                continue
            try:
                size = "" if e.is_dir() else f"{e.stat().st_size:,}"
            except OSError:
                size = "?"
            rows.append(f"{'d' if e.is_dir() else '-'} {size:>12}  {e.name}")
        body = "\n".join(rows) or "(empty)"
        return ToolResult(True, trim_tool_output(f"{p}\n{body}", max_output_chars), f"list_dir: {len(rows)} entries")

    # ---- search ---------------------------------------------------------
    @reg.tool(
        "search_files",
        "Search file contents for a regular expression, recursively. Uses "
        "ripgrep when available. Returns file:line:match.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "Directory to search (default: session directory)."},
                "glob": {"type": "string", "description": "Only search files matching this glob, e.g. *.py"},
                "max_results": {"type": "integer"},
            },
            "required": ["pattern"],
        },
    )
    def search_files(pattern: str, path: str = ".", glob: str = "", max_results: int = 100) -> ToolResult:
        root = resolve(path)
        limit = int(max_results or 100)
        rg = shutil.which("rg")
        if rg:
            # --max-columns keeps a single minified line (a bundled JS file can
            # run thousands of chars) from swamping the result; the preview flag
            # keeps such a hit visible as a stub rather than dropping it.
            cmd = [rg, "--line-number", "--no-heading", "--color=never",
                   "--max-columns", "300", "--max-columns-preview", "-m", str(limit)]
            for d in IGNORE_DIRS:
                cmd += ["-g", f"!{d}"]
            if glob:
                cmd += ["--glob", glob]
            cmd += ["-e", pattern, str(root)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                hits = [l for l in (proc.stdout or "").splitlines() if l][:limit]
            except Exception:
                hits = []
        else:
            try:
                rx = re.compile(pattern)
            except re.error as e:
                return ToolResult(False, f"Invalid regex: {e}", "search_files: bad regex")
            hits = []
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune noise dirs and hidden trees in place so os.walk never
                # descends into them — matching ripgrep's default reach.
                dirnames[:] = [d for d in dirnames
                               if d not in IGNORE_DIRS and not d.startswith(".")]
                for fn in filenames:
                    if glob and not fnmatch.fnmatch(fn, glob):
                        continue
                    fp = Path(dirpath) / fn
                    try:
                        if fp.stat().st_size > MAX_READ_BYTES:
                            continue
                        for i, line in enumerate(fp.read_text(errors="replace").splitlines(), 1):
                            if rx.search(line):
                                hits.append(f"{fp}:{i}:{line.strip()[:200]}")
                                if len(hits) >= limit:
                                    break
                    except Exception:
                        continue
                    if len(hits) >= limit:
                        break
                if len(hits) >= limit:
                    break
        if not hits:
            return ToolResult(True, f"No matches for {pattern!r} under {root}.", "search_files: 0 hits")
        return ToolResult(True, trim_tool_output("\n".join(hits), max_output_chars), f"search_files: {len(hits)} hits")

    @reg.tool(
        "glob_files",
        "Find files by name pattern, recursively. e.g. **/*.ts",
        {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    )
    def glob_files(pattern: str, path: str = ".") -> ToolResult:
        root = resolve(path)
        try:
            found = []
            for p in sorted(root.glob(pattern)):
                if not p.is_file():
                    continue
                # Skip anything living under a dependency/build/cache tree, the
                # same noise search_files excludes; an explicit pattern reaching
                # into one is honoured (the pattern itself names the dir).
                if set(p.parts) & IGNORE_DIRS:
                    continue
                found.append(str(p))
                if len(found) >= 300:
                    break
        except Exception as e:
            return ToolResult(False, f"Bad glob: {e}", "glob_files: error")
        if not found:
            return ToolResult(True, f"No files match {pattern!r} under {root}.", "glob_files: 0")
        return ToolResult(True, "\n".join(found), f"glob_files: {len(found)}")

    # ---- todo -----------------------------------------------------------
    @reg.tool(
        "todo",
        "Track multi-step work. Actions: list, add, done, clear. Keep this "
        "updated on tasks with more than two steps so you don't lose the thread.",
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "add", "done", "clear"]},
                "item": {"type": "string", "description": "Text for add."},
                "index": {"type": "integer", "description": "1-based index for done."},
            },
            "required": ["action"],
        },
    )
    def todo(action: str, item: str = "", index: int = 0) -> ToolResult:
        a = (action or "").lower()
        if a == "add":
            if not item:
                return ToolResult(False, "add requires `item`.", "todo: missing item")
            todos.append({"text": item, "done": False})
        elif a == "done":
            i = int(index or 0)
            if not (1 <= i <= len(todos)):
                return ToolResult(False, f"No todo #{i}; there are {len(todos)}.", "todo: bad index")
            todos[i - 1]["done"] = True
        elif a == "clear":
            todos.clear()
        elif a != "list":
            return ToolResult(False, f"Unknown action {action!r}.", "todo: bad action")
        if not todos:
            return ToolResult(True, "Todo list is empty.", "todo: empty")
        body = "\n".join(
            f"{i}. [{'x' if t['done'] else ' '}] {t['text']}" for i, t in enumerate(todos, 1)
        )
        return ToolResult(True, body, f"todo: {sum(1 for t in todos if t['done'])}/{len(todos)} done")

    # ---- memory ---------------------------------------------------------
    @reg.tool(
        "memory",
        "Long-term notes that survive across sessions, stored as local markdown "
        "files. Actions: list, read, write, delete. Use it for durable facts "
        "about this machine and the user's preferences.",
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "read", "write", "delete"]},
                "name": {"type": "string", "description": "Short slug, no extension."},
                "content": {"type": "string"},
            },
            "required": ["action"],
        },
        mutates=True,
    )
    def memory(action: str, name: str = "", content: str = "") -> ToolResult:
        mem_dir.mkdir(parents=True, exist_ok=True)
        a = (action or "").lower()
        if a == "list":
            files = sorted(p.stem for p in mem_dir.glob("*.md"))
            return ToolResult(True, "\n".join(files) or "(no memories yet)", f"memory: {len(files)}")
        safe = re.sub(r"[^a-zA-Z0-9_-]", "-", name or "").strip("-")
        if not safe:
            return ToolResult(False, "A valid `name` is required.", "memory: bad name")
        p = mem_dir / f"{safe}.md"
        if a == "read":
            if not p.exists():
                return ToolResult(False, f"No memory named {safe!r}.", "memory: missing")
            return ToolResult(True, p.read_text(errors="replace"), f"memory: read {safe}")
        if a == "write":
            p.write_text(content or "")
            return ToolResult(True, f"Saved memory {safe!r} ({len(content):,} bytes).", f"memory: wrote {safe}")
        if a == "delete":
            if p.exists():
                p.unlink()
                return ToolResult(True, f"Deleted memory {safe!r}.", f"memory: deleted {safe}")
            return ToolResult(False, f"No memory named {safe!r}.", "memory: missing")
        return ToolResult(False, f"Unknown action {action!r}.", "memory: bad action")

    return reg
