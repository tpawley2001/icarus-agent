"""Icarus configuration.

Everything lives under ~/.icarus. There is no cloud component and no account:
the only network destination Icarus ever contacts is the llama-swap base_url
below, which defaults to loopback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

HOME = Path(os.environ.get("ICARUS_HOME", Path.home() / ".icarus"))
CONFIG_PATH = HOME / "config.yaml"
SESSIONS_DIR = HOME / "sessions"
SKILLS_DIR = HOME / "skills"
MEMORY_DIR = HOME / "memory"
CAPS_PATH = HOME / "capabilities.json"
LOG_PATH = HOME / "icarus.log"

# llama-swap's own config is the authoritative source for per-model context
# windows (it bakes --ctx-size into each model's launch command), so we read it
# rather than guessing. Optional: absence just means we fall back to defaults.
DEFAULT_SWAP_CONFIG = str(Path.home() / "llama-swap" / "config.yaml")

DEFAULTS: Dict[str, Any] = {
    "model": {
        # Empty default -> resolved at startup from llama-swap's active model.
        "default": "",
        "base_url": "http://127.0.0.1:9292/v1",
        # llama-swap needs no key, but some proxies in front of it might.
        "api_key": "not-needed",
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 4096,
        # Reasoning models served with --jinja return EMPTY content unless
        # thinking is explicitly disabled. On by default; see llm.py.
        "disable_thinking": True,
    },
    "llama_swap": {
        "config_path": DEFAULT_SWAP_CONFIG,
        # Cold model loads are slow (llama-swap's own healthCheckTimeout is
        # 600s). A short client timeout here is the #1 cause of phantom
        # failures on the first request after a swap.
        "load_timeout": 600,
        "request_timeout": 900,
        # Unload the previous model before switching. llama-swap will evict on
        # its own, but doing it explicitly avoids a moment of double-resident
        # VRAM that can OOM the big models on a 2x3060.
        "unload_before_switch": True,
    },
    "agent": {
        # Task profiles. Different work wants a different shape of context:
        # a shell one-liner does not need 64K, and a refactor across a dozen
        # files does. `ctx` is the server-side window to request (an integer,
        # "max" for the largest that fits, or null to leave the model alone);
        # the rest are client-side budget knobs applied immediately.
        "profile": "balanced",
        "profiles": {
            "quick": {
                "ctx": 8192, "max_tokens": 1024, "max_iterations": 10,
                "context_threshold": 0.70, "max_output_chars": 8000,
                "note": "short questions and one-off commands; smallest KV, fastest load",
            },
            "balanced": {
                "ctx": 32768, "max_tokens": 4096, "max_iterations": 40,
                "context_threshold": 0.75, "max_output_chars": 30000,
                "note": "the default; general work across a few files",
            },
            "code": {
                "ctx": 65536, "max_tokens": 8192, "max_iterations": 60,
                "context_threshold": 0.80, "max_output_chars": 60000,
                "note": "multi-file edits; room for long diffs and big tool output",
            },
            "deep": {
                "ctx": "max", "max_tokens": 8192, "max_iterations": 100,
                "context_threshold": 0.85, "max_output_chars": 80000,
                "note": "long investigations; the largest window that fits",
            },
        },
        "max_iterations": 40,
        # Fraction of the model's context window we will fill before compacting.
        "context_threshold": 0.75,
        "system_prompt": "",  # empty -> built-in prompt
        "stream": True,
    },
    "tools": {
        "enabled": [],  # empty -> all registered tools
        "disabled": [],
        # Shell commands matching these need explicit approval each run.
        # Approval is remembered per-session once granted for a given pattern.
        "require_approval": [
            r"\brm\s+-[rf]",
            r"\bdd\b",
            r"\bmkfs",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bsystemctl\s+(stop|disable|mask)",
            r":\(\)\{.*\};:",
            r"\bchown\b",
            r"\bchmod\s+-R",
            r">\s*/dev/[sn]d",
            r"\bgit\s+push\b",
            r"\bcurl\b.*\|\s*(ba)?sh",
        ],
        "terminal_timeout": 120,
        "max_output_chars": 30000,
    },
    "usage": {
        # Mission Control reads this file for its Token Usage tab. Writing here
        # makes Icarus turns show up alongside every other local model caller.
        "stats_file": str(Path.home() / "mission-control" / "usage_stats.json"),
        "enabled": True,
    },
    "ui": {
        "color": True,
        "show_tool_output": True,
        "spinner": True,
        # Type-ahead steering + Esc interrupt during a turn.
        # Automatically inert when stdin/stdout is not a terminal.
        "interrupt": True,
    },
}


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; override wins on scalars."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dirs() -> None:
    for d in (HOME, SESSIONS_DIR, SKILLS_DIR, MEMORY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load() -> Dict[str, Any]:
    ensure_dirs()
    user: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            user = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except Exception as e:  # a broken config must not brick the CLI
            print(f"icarus: config.yaml unreadable ({e}); using defaults")
            user = {}
    cfg = _merge(DEFAULTS, user)

    # Env overrides, so a cron job or a one-off can retarget without editing.
    if os.environ.get("ICARUS_BASE_URL"):
        cfg["model"]["base_url"] = os.environ["ICARUS_BASE_URL"]
    if os.environ.get("ICARUS_MODEL"):
        cfg["model"]["default"] = os.environ["ICARUS_MODEL"]
    return cfg


def save(cfg: Dict[str, Any]) -> None:
    """Persist only what differs from the defaults, keeping config.yaml legible."""
    ensure_dirs()

    def diff(base: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in cur.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                sub = diff(base[k], v)
                if sub:
                    out[k] = sub
            elif base.get(k) != v:
                out[k] = v
        return out

    CONFIG_PATH.write_text(yaml.safe_dump(diff(DEFAULTS, cfg), sort_keys=False))


def write_default_config() -> Path:
    """Create a commented starter config if the user has none."""
    ensure_dirs()
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    CONFIG_PATH.write_text(
        "# Icarus — fully local CLI agent for llama-swap.\n"
        "# Only the keys you set here override the built-in defaults;\n"
        "# run `icarus config` to print the effective config.\n\n"
        "model:\n"
        "  # Leave blank to follow llama-swap's currently loaded model.\n"
        "  default: ''\n"
        "  base_url: http://127.0.0.1:9292/v1\n"
        "  temperature: 0.2\n\n"
        "agent:\n"
        "  max_iterations: 40\n"
    )
    return CONFIG_PATH
