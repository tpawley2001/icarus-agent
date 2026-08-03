"""Skills — hermes' progressive-disclosure pattern, kept deliberately simple.

A skill is a directory containing SKILL.md with YAML frontmatter:

    ---
    name: my-skill
    description: One line telling the model when this is relevant.
    ---
    ...body...

Only the name+description of each skill goes into the system prompt; the body
is loaded on demand via the `skill` tool. On an 8K-context model that
distinction is the difference between working and not.

Icarus reads its own ~/.icarus/skills plus, if present, the hermes skills tree —
there is no reason to duplicate a library that already exists on this machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from . import config
from .tools.registry import Registry, ToolResult

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

# Skills bodies can be long; a local model's context can't take a whole one
# blind. Anything past this is truncated with a pointer to the file.
MAX_SKILL_CHARS = 12_000


@dataclass
class Skill:
    name: str
    description: str
    path: Path
    source: str = "icarus"

    def body(self) -> str:
        try:
            raw = self.path.read_text(errors="replace")
        except Exception as e:
            return f"(cannot read {self.path}: {e})"
        m = FRONTMATTER.match(raw)
        text = m.group(2) if m else raw
        if len(text) > MAX_SKILL_CHARS:
            text = (
                text[:MAX_SKILL_CHARS]
                + f"\n\n[truncated — read the rest with read_file {self.path}]"
            )
        return text.strip()


def _parse(path: Path, source: str) -> Optional[Skill]:
    try:
        raw = path.read_text(errors="replace")
    except Exception:
        return None
    m = FRONTMATTER.match(raw)
    if not m:
        return None
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        return None
    name = str(meta.get("name") or path.parent.name).strip()
    desc = str(meta.get("description") or "").strip().replace("\n", " ")
    if not name:
        return None
    return Skill(name=name, description=desc, path=path, source=source)


def discover(extra_roots: Optional[List[Path]] = None) -> Dict[str, Skill]:
    """Find every SKILL.md. Icarus' own skills win on a name collision."""
    roots: List[tuple[Path, str]] = []
    hermes = Path.home() / ".hermes" / "skills"
    if hermes.is_dir():
        roots.append((hermes, "hermes"))
    for r in extra_roots or []:
        if r.is_dir():
            roots.append((r, "extra"))
    # Last wins, so put Icarus' own tree at the end.
    if config.SKILLS_DIR.is_dir():
        roots.append((config.SKILLS_DIR, "icarus"))

    found: Dict[str, Skill] = {}
    for root, source in roots:
        for p in root.rglob("SKILL.md"):
            s = _parse(p, source)
            if s:
                found[s.name] = s
    return found


def catalog_lines(skills: Dict[str, Skill], limit: int = 60) -> str:
    """The one-line-per-skill index that goes in the system prompt."""
    if not skills:
        return ""
    rows = []
    for name in sorted(skills)[:limit]:
        s = skills[name]
        desc = s.description[:150] if s.description else "(no description)"
        rows.append(f"- {name}: {desc}")
    more = "" if len(skills) <= limit else f"\n- ...and {len(skills) - limit} more (use skill action=list)"
    return "\n".join(rows) + more


def register(reg: Registry, skills: Dict[str, Skill]) -> None:
    """Expose the skill library as a single tool."""

    @reg.tool(
        "skill",
        "Look up a reusable playbook. action=list shows every skill; "
        "action=view name=<skill> loads its full instructions. Check here "
        "before improvising a procedure you might already have written down.",
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "view"]},
                "name": {"type": "string"},
            },
            "required": ["action"],
        },
    )
    def skill(action: str, name: str = "") -> ToolResult:
        a = (action or "").lower()
        if a == "list":
            if not skills:
                return ToolResult(True, "No skills installed.", "skill: none")
            body = "\n".join(
                f"- {n} [{skills[n].source}]: {skills[n].description[:180]}"
                for n in sorted(skills)
            )
            return ToolResult(True, body, f"skill: {len(skills)} available")
        if a == "view":
            s = skills.get(name)
            if not s:
                close = [n for n in skills if name.lower() in n.lower()][:8]
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                return ToolResult(False, f"No skill named {name!r}.{hint}", "skill: not found")
            return ToolResult(True, f"# Skill: {s.name}\n\n{s.body()}", f"skill: loaded {s.name}")
        return ToolResult(False, f"Unknown action {action!r}.", "skill: bad action")
