"""Autodetect the local AI stack.

Icarus is built for llama-swap, but "point it at whatever OpenAI-compatible
server you already run" is a much better first-run experience than "edit this
YAML". So on install (and on demand via `icarus --detect`) we go looking.

Detection is signature-based, not port-based: several of these servers share
default ports, and people move them. We probe a candidate, then ask it
questions only one product answers a particular way.

Everything here is loopback-first and read-only. Nothing is contacted except
hosts you already run.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TIMEOUT = 2.0

# (label, port). Ordered so the most specific/likely land first. Several
# products collide on a port, which is exactly why the signature check exists.
CANDIDATE_PORTS: List[Tuple[str, int]] = [
    ("llama-swap", 9292),
    ("ollama", 11434),
    ("lm-studio", 1234),
    ("vllm", 8000),
    ("llama.cpp", 8080),
    ("koboldcpp", 5001),
    ("text-generation-webui", 5000),
    ("tabbyapi", 5000),
    ("jan", 1337),
    ("localai", 8080),
    ("llamafile", 8081),
    ("openai-compatible", 8001),
    ("openai-compatible", 9000),
]

# Servers Icarus can drive. llama-swap gets the full feature set; the rest work
# as plain OpenAI-compatible endpoints (no /ctx resize, no VRAM planning).
FULL_SUPPORT = {"llama-swap"}

# Something listening on loopback is not necessarily running locally. Cloud
# routers (OpenRouter proxies, LiteLLM, Cloudflare Workers AI gateways) present
# an ordinary OpenAI endpoint on 127.0.0.1 and forward every token off the
# machine. Icarus' entire premise is that it doesn't do that, so these are
# detected, labelled, and never auto-selected.
CLOUD_MARKERS = (
    ":free",          # OpenRouter free tier
    "@cf/",           # Cloudflare Workers AI
    "openai/", "anthropic/", "google/", "x-ai/", "perplexity/", "cohere/",
)
CLOUD_MODEL_NAMES = (
    "gpt-3", "gpt-4", "gpt-5", "o1-", "o3-", "claude-", "gemini", "grok",
    "sonar", "command-r",
)


def looks_like_cloud_proxy(models: List[str]) -> bool:
    """True when a loopback endpoint is really forwarding to hosted providers."""
    if not models:
        return False
    low = [m.lower() for m in models]
    hits = sum(
        1 for m in low
        if any(k in m for k in CLOUD_MARKERS) or any(m.startswith(n) for n in CLOUD_MODEL_NAMES)
    )
    # A local server might legitimately carry one oddly-named model; a router
    # carries dozens. Require either a strong share or a decent absolute count.
    return hits >= 3 or (hits / len(low)) > 0.25


@dataclass
class Endpoint:
    kind: str                  # llama-swap | ollama | lm-studio | ...
    host: str
    port: int
    base_url: str              # includes /v1 where applicable
    models: List[str] = field(default_factory=list)
    version: str = ""
    config_path: str = ""      # llama-swap only
    notes: List[str] = field(default_factory=list)
    cloud_proxy: bool = False  # loopback address, hosted models behind it

    @property
    def full_support(self) -> bool:
        return self.kind in FULL_SUPPORT

    @property
    def label(self) -> str:
        return f"{self.kind} at {self.host}:{self.port}"


@dataclass
class Stack:
    endpoints: List[Endpoint] = field(default_factory=list)
    gpus: List[str] = field(default_factory=list)
    accelerator: str = ""      # cuda | rocm | metal | cpu
    python: str = ""
    has_yaml: bool = False
    usage_stats: str = ""      # Mission Control integration, if present
    warnings: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[Endpoint]:
        """Prefer a server we fully support, then one that actually has models."""
        if not self.endpoints:
            return None
        # Cloud proxies sort last and are never chosen implicitly.
        local = [e for e in self.endpoints if not e.cloud_proxy]
        pool = local or []
        if not pool:
            return None
        return sorted(pool, key=lambda e: (not e.full_support, not e.models, e.port))[0]


# ---------------------------------------------------------------------------
# low-level probes
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _get(url: str, timeout: float = TIMEOUT) -> Optional[Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "icarus-detect"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _models_from_openai(root: str) -> List[str]:
    data = _get(f"{root}/v1/models")
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return sorted(str(m.get("id", "")) for m in data["data"] if m.get("id"))
    return []


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------

def identify(host: str, port: int) -> Optional[Endpoint]:
    """Work out which product is listening, by asking it something distinctive."""
    root = f"http://{host}:{port}"

    # llama-swap: /running is its own management endpoint and returns a
    # {"running": [...]} object. Nothing else on this list serves that shape.
    running = _get(f"{root}/running")
    if isinstance(running, dict) and "running" in running:
        ep = Endpoint("llama-swap", host, port, f"{root}/v1",
                      models=_models_from_openai(root))
        ep.cloud_proxy = looks_like_cloud_proxy(ep.models)
        ep.config_path = find_llama_swap_config()
        if not ep.config_path:
            ep.notes.append(
                "config.yaml not found — /ctx resizing and exact context "
                "windows need it; set llama_swap.config_path"
            )
        return ep

    # Ollama: /api/tags is unique to it.
    tags = _get(f"{root}/api/tags")
    if isinstance(tags, dict) and "models" in tags:
        ver = _get(f"{root}/api/version")
        ep = Endpoint(
            "ollama", host, port, f"{root}/v1",
            models=sorted(str(m.get("name", "")) for m in tags.get("models") or []),
            version=(ver or {}).get("version", "") if isinstance(ver, dict) else "",
        )
        ep.notes.append("OpenAI-compatible endpoint; no VRAM planning or /ctx resize")
        return ep

    # llama.cpp's own server exposes /props with a default_generation_settings
    # block. LocalAI and llamafile also answer /v1/models, so check /props first.
    props = _get(f"{root}/props")
    if isinstance(props, dict) and ("default_generation_settings" in props
                                    or "model_path" in props):
        ep = Endpoint("llama.cpp", host, port, f"{root}/v1",
                      models=_models_from_openai(root))
        n_ctx = ""
        try:
            n_ctx = str(props["default_generation_settings"]["n_ctx"])
        except Exception:
            pass
        if n_ctx:
            ep.notes.append(f"serving a fixed {n_ctx}-token context")
        ep.notes.append("single model; use llama-swap if you want model switching")
        return ep

    # KoboldCpp
    kobold = _get(f"{root}/api/v1/model")
    if isinstance(kobold, dict) and "result" in kobold:
        return Endpoint("koboldcpp", host, port, f"{root}/v1",
                        models=[str(kobold["result"])],
                        notes=["OpenAI-compatible endpoint; no VRAM planning"])

    # Anything else that speaks /v1/models is usable as a generic endpoint.
    models = _models_from_openai(root)
    if models:
        kind = _guess_generic(root, port)
        ep = Endpoint(kind, host, port, f"{root}/v1", models=models,
                      notes=["generic OpenAI-compatible endpoint"])
        if looks_like_cloud_proxy(models):
            ep.kind = "cloud-router"
            ep.cloud_proxy = True
            ep.notes = ["forwards to hosted providers — NOT local inference; "
                        "Icarus will not select this automatically"]
        return ep
    return None


def _guess_generic(root: str, port: int) -> str:
    """Name a generic OpenAI endpoint from weak hints. Cosmetic only."""
    if _get(f"{root}/version") is not None and port == 8000:
        return "vllm"
    if port == 1234:
        return "lm-studio"
    if port == 1337:
        return "jan"
    if _get(f"{root}/readyz") is not None:
        return "localai"
    return "openai-compatible"


# ---------------------------------------------------------------------------
# llama-swap config discovery
# ---------------------------------------------------------------------------

def find_llama_swap_config() -> str:
    """Locate llama-swap's config.yaml.

    The running process is the most reliable source — it was started with
    --config. Fall back to the systemd unit, then to conventional locations.
    """
    # 1. The live process's own arguments.
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            if "llama-swap" in line and "--config" in line:
                m = re.search(r"--config[= ]+(\S+)", line)
                if m and Path(m.group(1)).is_file():
                    return m.group(1)
    except Exception:
        pass

    # 2. The systemd unit (system or user scope).
    for cmd in (["systemctl", "cat", "llama-swap.service"],
                ["systemctl", "--user", "cat", "llama-swap.service"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"--config[= ]+(\S+)", out)
            if m and Path(m.group(1)).is_file():
                return m.group(1)
        except Exception:
            continue

    # 3. Conventional locations.
    for p in (
        Path.home() / "llama-swap" / "config.yaml",
        Path.home() / ".config" / "llama-swap" / "config.yaml",
        Path("/etc/llama-swap/config.yaml"),
        Path("/opt/llama-swap/config.yaml"),
        Path.home() / "llama-swap" / "config.yml",
    ):
        if p.is_file():
            return str(p)
    return ""


# ---------------------------------------------------------------------------
# hardware + environment
# ---------------------------------------------------------------------------

def detect_accelerator() -> Tuple[str, List[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=8).stdout.strip()
        if out:
            return "cuda", [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        pass
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=8).stdout
        names = [l.strip() for l in out.splitlines() if "Card series" in l]
        if names:
            return "rocm", names
    except Exception:
        pass
    if os.uname().sysname == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if "Apple" in out:
                return "metal", [out]
        except Exception:
            pass
    return "cpu", []


def find_usage_stats() -> str:
    """Optional Mission Control integration — only if it already exists."""
    for p in (Path.home() / "mission-control" / "usage_stats.json",
              Path.home() / ".config" / "mission-control" / "usage_stats.json"):
        if p.is_file():
            return str(p)
    return ""


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def scan(hosts: Optional[List[str]] = None,
         extra_ports: Optional[List[int]] = None) -> Stack:
    """Probe the usual suspects and report everything found."""
    stack = Stack()
    stack.python = ".".join(map(str, __import__("sys").version_info[:3]))
    try:
        import yaml  # noqa: F401
        stack.has_yaml = True
    except ImportError:
        stack.has_yaml = False
        stack.warnings.append("PyYAML is not installed (pip install --user pyyaml)")

    stack.accelerator, stack.gpus = detect_accelerator()
    stack.usage_stats = find_usage_stats()

    targets = hosts or ["127.0.0.1"]
    ports: List[Tuple[str, int]] = list(CANDIDATE_PORTS)
    for p in extra_ports or []:
        ports.append(("openai-compatible", p))

    seen: set[Tuple[str, int]] = set()
    for host in targets:
        for _label, port in ports:
            if (host, port) in seen:
                continue
            seen.add((host, port))
            if not _port_open(host, port):
                continue
            ep = identify(host, port)
            if ep and not any(e.port == ep.port and e.host == ep.host
                              for e in stack.endpoints):
                stack.endpoints.append(ep)

    if not stack.endpoints:
        stack.warnings.append(
            "no local inference server found — start llama-swap, Ollama, "
            "LM Studio, or any OpenAI-compatible server and re-run"
        )
    return stack


def config_from(stack: Stack, endpoint: Optional[Endpoint] = None) -> Dict[str, Any]:
    """Turn a detection result into the config Icarus should write."""
    ep = endpoint or stack.best
    cfg: Dict[str, Any] = {}
    if ep:
        cfg["model"] = {"base_url": ep.base_url, "default": ""}
        if ep.kind == "llama-swap" and ep.config_path:
            cfg["llama_swap"] = {"config_path": ep.config_path}
        elif ep.kind != "llama-swap":
            # Without llama-swap there is no per-model --ctx-size to read, so
            # pin a conservative window rather than guess high and get
            # truncated server-side.
            cfg["llama_swap"] = {"config_path": ""}
            cfg["agent"] = {"profile": "balanced"}
    if stack.usage_stats:
        cfg["usage"] = {"stats_file": stack.usage_stats, "enabled": True}
    else:
        cfg["usage"] = {"enabled": False}
    return cfg


def render(stack: Stack, color=None) -> str:
    """Human-readable summary for the installer and `icarus --detect`."""
    c = color or (lambda s, _k=None: s)
    lines: List[str] = []
    lines.append("Inference servers")
    if not stack.endpoints:
        lines.append("  none found")
    for ep in stack.endpoints:
        star = " ← will use" if ep is stack.best else ""
        if ep.cloud_proxy:
            tier = "CLOUD ROUTER — not local"
        else:
            tier = "full support" if ep.full_support else "basic (OpenAI-compatible)"
        lines.append(f"  {ep.label:<34} {len(ep.models):>3} models   {tier}{star}")
        if ep.config_path:
            lines.append(f"      config: {ep.config_path}")
        for n in ep.notes:
            lines.append(f"      note:   {n}")
    lines.append("")
    lines.append("Hardware")
    if stack.gpus:
        for g in stack.gpus:
            lines.append(f"  {stack.accelerator}: {g}")
    else:
        lines.append(f"  {stack.accelerator} (no GPU detected — expect slow inference)")
    lines.append("")
    lines.append("Environment")
    lines.append(f"  python {stack.python}   PyYAML {'yes' if stack.has_yaml else 'MISSING'}")
    if stack.usage_stats:
        lines.append(f"  Mission Control usage stats: {stack.usage_stats}")
    for w in stack.warnings:
        lines.append(f"  ! {w}")
    return "\n".join(lines)
