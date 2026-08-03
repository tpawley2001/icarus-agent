"""Plan and apply a context-window change for a llama-swap model.

llama.cpp fixes the context window at server launch (`--ctx-size`), so a
"flexible context window" means relaunching the model with a different value.
On a two-GPU box that is not a free choice: the KV cache is allocated up front
and its cost per token varies enormously between models on this machine —

    qwen3.6:35b-a3b      80 KB/token   ->  32K ctx =  2.5 GB
    gemma4:12b        1,536 KB/token   ->  32K ctx = 48.0 GB

a 19x spread. Bumping the second one blindly is what OOM-thrashes the box, so
every change here is costed against the model's real geometry and the free VRAM
on the cards it is placed on, and refused if it will not fit.

Changes are recorded in ctx_overrides.json next to llama-swap's config so they
survive a gen_config.py regeneration, and written into config.yaml so
llama-swap's --watch-config picks them up.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .gguf import ModelShape, shape_of

GB = 1024 ** 3
MIB = 1024 ** 2

# Leave room for the compute buffers, CUDA context, and whatever else shares
# the card. Without this, a plan that "just fits" aborts at load.
HEADROOM_MIB = 900

# Context sizes offered by the presets. Powers of two keep llama.cpp happy.
LADDER = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


@dataclass
class Gpu:
    index: int
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass
class Placement:
    gpus: List[int] = field(default_factory=list)
    tensor_split: str = ""

    @property
    def dual(self) -> bool:
        return len(self.gpus) > 1 or bool(self.tensor_split)


@dataclass
class Plan:
    model: str
    current_ctx: int
    target_ctx: int
    feasible: bool = False
    kv_bytes: int = 0
    weight_bytes: int = 0
    free_bytes: int = 0
    max_safe_ctx: int = 0
    trained_ctx: int = 0
    warnings: List[str] = field(default_factory=list)
    reason: str = ""
    calibrated: bool = False

    @property
    def kv_gb(self) -> float: return self.kv_bytes / GB

    @property
    def weight_gb(self) -> float: return self.weight_bytes / GB

    @property
    def free_gb(self) -> float: return self.free_bytes / GB


def foreign_vram_mib() -> Dict[int, int]:
    """VRAM held by processes llama-swap will NOT evict, per GPU index.

    Instantaneous free memory is the wrong planning baseline: llama-swap
    unloads models on demand and on a TTL, so another llama-server's allocation
    is reclaimable. What is *not* reclaimable is everything else sharing the
    card — ComfyUI, the Kokoro TTS server, a Plex transcode. Budget against
    those, and the answer stops changing minute to minute.
    """
    try:
        uuid_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        apps_out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return {}

    by_uuid: Dict[str, int] = {}
    for line in uuid_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            by_uuid[parts[1]] = int(parts[0])

    out: Dict[int, int] = {}
    for line in apps_out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        pid, uuid, mem = parts
        idx = by_uuid.get(uuid)
        if idx is None or not mem.isdigit():
            continue
        try:
            comm = subprocess.run(["ps", "-p", pid, "-o", "comm="],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            comm = ""
        if comm == "llama-server":
            continue  # llama-swap can evict this
        if not comm:
            # PID already gone but nvidia-smi still lists its memory — a
            # process mid-teardown, which is exactly what an unload looks like.
            # Counting it as non-evictable zeroes the budget and blocks a
            # perfectly valid resize, so treat it as reclaimable.
            continue
        out[idx] = out.get(idx, 0) + int(mem)
    return out


def gpus() -> List[Gpu]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    found = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                found.append(Gpu(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
            except ValueError:
                continue
    return found


class SwapConfig:
    """Read/modify llama-swap's generated config without disturbing the rest."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def raw(self) -> str:
        return self.path.read_text()

    def parsed(self) -> Dict:
        return yaml.safe_load(self.raw()) or {}

    def model_cmd(self, model: str) -> str:
        spec = (self.parsed().get("models") or {}).get(model) or {}
        return str(spec.get("cmd", "")) if isinstance(spec, dict) else ""

    def gguf_path(self, model: str) -> str:
        m = re.search(r"--model\s+(\S+)", self.model_cmd(model))
        return m.group(1) if m else ""

    def ctx(self, model: str) -> int:
        m = re.search(r"--ctx-size\s+(\d+)", self.model_cmd(model))
        return int(m.group(1)) if m else 0

    def placement(self, model: str) -> Placement:
        spec = (self.parsed().get("models") or {}).get(model) or {}
        env = spec.get("env") or [] if isinstance(spec, dict) else []
        p = Placement()
        for e in env:
            g = re.match(r"CUDA_VISIBLE_DEVICES=(.+)", str(e))
            if g:
                p.gpus = [int(x) for x in g.group(1).split(",") if x.strip().isdigit()]
        ts = re.search(r"--tensor-split\s+(\S+)", self.model_cmd(model))
        if ts:
            p.tensor_split = ts.group(1)
            if not p.gpus:
                p.gpus = list(range(len(p.tensor_split.split(","))))
        if not p.gpus:
            p.gpus = [0]
        return p

    def set_ctx(self, model: str, ctx: int) -> bool:
        """Rewrite just this model's --ctx-size, textually.

        Deliberately not a YAML round-trip: the file is generated with comments
        and block scalars that pyyaml would reformat wholesale, producing a
        huge diff and losing the notes gen_config.py writes.
        """
        text = self.raw()
        # Find the model's block, then the first --ctx-size inside it.
        start = text.find(f'"{model}":')
        if start == -1:
            return False
        nxt = re.search(r'\n  "', text[start + 1:])
        end = start + 1 + nxt.start() if nxt else len(text)
        block = text[start:end]
        new_block, n = re.subn(r"--ctx-size\s+\d+", f"--ctx-size {ctx}", block, count=1)
        if n == 0:
            return False
        self.path.write_text(text[:start] + new_block + text[end:])
        return True


# --- weight calibration ----------------------------------------------------
# The GGUF file size overestimates the GPU weight footprint: llama.cpp keeps
# some tensors (notably Gemma's very large token-embedding table) on the host.
# Measured on this box, gemma4:e4b occupies ~3.7 GB of VRAM for a 5.0 GB file.
# So: measure once, remember, and plan against the real number. KV *scaling*
# needs no calibration — it was measured exact (384 MiB predicted / 384 MiB
# actual across an 8K->32K change).

CALIBRATION = Path.home() / ".icarus" / "vram_calibration.json"


def read_calibration() -> Dict[str, int]:
    if not CALIBRATION.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(CALIBRATION.read_text()).items()}
    except Exception:
        return {}


def record_measurement(model: str, ctx: int, measured_mib: int, shape: ModelShape) -> None:
    """Back out the true weight footprint from an observed load."""
    kv_mib = shape.kv_bytes(ctx) / MIB
    weights = int(max(0, measured_mib - kv_mib))
    if weights <= 0:
        return
    data = read_calibration()
    data[model] = weights
    try:
        CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CALIBRATION.write_text(json.dumps(data, indent=1, sort_keys=True))
    except Exception:
        pass


def llama_server_vram_mib() -> int:
    """Total VRAM currently held by llama-server processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return 0
    total = 0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        try:
            comm = subprocess.run(["ps", "-p", parts[0], "-o", "comm="],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            continue
        if comm == "llama-server":
            total += int(parts[1])
    return total


def overrides_path(config_path: str | Path) -> Path:
    return Path(config_path).parent / "ctx_overrides.json"


def read_overrides(config_path: str | Path) -> Dict[str, int]:
    p = overrides_path(config_path)
    if not p.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(p.read_text()).items()}
    except Exception:
        return {}


# The value gen_config.py would emit, captured the first time we override a
# model. Kept on Icarus' side so the sidecar stays a plain {model: int} map that
# gen_config.py can consume directly.
ORIGINALS = Path.home() / ".icarus" / "ctx_originals.json"


def _read_originals() -> Dict[str, int]:
    if not ORIGINALS.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(ORIGINALS.read_text()).items()}
    except Exception:
        return {}


def remember_original(model: str, ctx: int) -> None:
    data = _read_originals()
    if model in data or ctx <= 0:
        return  # only the first, pre-Icarus value is the real default
    data[model] = ctx
    try:
        ORIGINALS.parent.mkdir(parents=True, exist_ok=True)
        ORIGINALS.write_text(json.dumps(data, indent=1, sort_keys=True))
    except Exception:
        pass


def original_ctx(model: str) -> int:
    return _read_originals().get(model, 0)


def forget_original(model: str) -> None:
    data = _read_originals()
    if data.pop(model, None) is not None:
        try:
            ORIGINALS.write_text(json.dumps(data, indent=1, sort_keys=True))
        except Exception:
            pass


def write_override(config_path: str | Path, model: str, ctx: Optional[int]) -> None:
    """Record (or clear) a persistent override. gen_config.py reads this file."""
    p = overrides_path(config_path)
    data = read_overrides(config_path)
    if ctx is None:
        data.pop(model, None)
    else:
        data[model] = int(ctx)
    try:
        p.write_text(json.dumps(data, indent=1, sort_keys=True))
    except Exception:
        pass


def plan(
    cfg: SwapConfig,
    model: str,
    target_ctx: int,
    shape: Optional[ModelShape] = None,
    loaded_elsewhere_bytes: int = 0,
) -> Plan:
    """Cost a proposed context size against the model's geometry and free VRAM."""
    current = cfg.ctx(model)
    gguf = cfg.gguf_path(model)
    shape = shape or (shape_of(gguf) if gguf else None)
    p = Plan(model=model, current_ctx=current, target_ctx=target_ctx)

    if shape is None:
        p.reason = "could not read the model's GGUF metadata; cannot size safely"
        return p

    p.trained_ctx = shape.trained_ctx
    # Prefer a measured weight footprint; fall back to file size, which is a
    # safe over-estimate.
    measured = read_calibration().get(model, 0)
    p.weight_bytes = measured * MIB if measured else shape.file_bytes
    p.calibrated = bool(measured)
    p.kv_bytes = shape.kv_bytes(target_ctx)

    place = cfg.placement(model)
    cards = {g.index: g for g in gpus()}
    if not cards:
        p.reason = "nvidia-smi unavailable; cannot verify VRAM"
        return p

    # Budget = total VRAM on the placement cards, minus what llama-swap cannot
    # evict, minus headroom for compute buffers and the CUDA context.
    foreign = foreign_vram_mib()
    usable = 0
    for idx in place.gpus:
        g = cards.get(idx)
        if g:
            usable += max(0, g.total_mib - foreign.get(idx, 0) - HEADROOM_MIB) * MIB
    p.free_bytes = usable

    per_token = shape.kv_bytes_per_token()
    if per_token <= 0:
        p.reason = "unknown KV geometry; cannot size safely"
        return p

    # The weights have to live there too, whatever the context size.
    budget_for_kv = usable - p.weight_bytes
    p.max_safe_ctx = _floor_to_ladder(max(0, budget_for_kv) // per_token)
    # Never propose more than the model was trained for.
    if p.trained_ctx:
        p.max_safe_ctx = min(p.max_safe_ctx, _floor_to_ladder(p.trained_ctx))
    if not p.calibrated and p.max_safe_ctx < current:
        # File size over-estimates the GPU footprint (Gemma keeps its embedding
        # table on the host), so an uncalibrated model can look over budget at
        # a size it already runs at. Say so rather than imply it is broken.
        p.warnings.append(
            f"estimate is uncalibrated and conservative — this model already "
            f"runs at {current:,}; load it once and the figure will be measured"
        )
        p.max_safe_ctx = max(p.max_safe_ctx, current)

    if p.trained_ctx and target_ctx > p.trained_ctx:
        p.warnings.append(
            f"{target_ctx:,} exceeds the {p.trained_ctx:,} this model was trained "
            "for; quality degrades well before it errors"
        )
    if place.dual:
        p.warnings.append(
            "this model spans both GPUs — it must run alone, and a larger KV "
            "cache makes that stricter"
        )
    if shape.arch.startswith("gemma"):
        # Gemma uses sliding-window attention on most layers, so llama.cpp
        # allocates less than the full-KV figure. Being conservative is the
        # safe direction, but say so rather than look wrong.
        p.warnings.append(
            "gemma uses sliding-window attention, so real KV use is typically "
            "below this estimate — the figure is deliberately conservative"
        )

    if target_ctx <= p.max_safe_ctx:
        p.feasible = True
    else:
        p.reason = (
            f"needs {p.kv_gb:.1f} GB of KV cache but only {max(0, budget_for_kv) / GB:.1f} GB "
            f"is free on GPU {','.join(map(str, place.gpus))} "
            f"(largest that fits: {p.max_safe_ctx:,})"
        )
    return p


def _floor_to_ladder(n: int) -> int:
    best = 0
    for step in LADDER:
        if step <= n:
            best = step
    return best


def apply(
    cfg: SwapConfig,
    model: str,
    ctx: int,
    persist: bool = True,
    backup: bool = True,
) -> Tuple[bool, str]:
    """Write the new ctx, leaving llama-swap's --watch-config to reload it."""
    if backup:
        try:
            shutil.copy2(cfg.path, cfg.path.with_suffix(".yaml.icarus-bak"))
        except Exception:
            pass
    if persist:
        # Capture what gen_config.py had here before we touch it, so /ctx reset
        # can put it back without waiting for a regeneration.
        remember_original(model, cfg.ctx(model))
    if not cfg.set_ctx(model, ctx):
        return False, f"could not find --ctx-size for {model} in {cfg.path}"
    if persist:
        write_override(cfg.path, model, ctx)
    return True, f"{model} set to {ctx:,} context"


def suggest_for_usage(used_tokens: int, current: int, max_safe: int) -> int:
    """Right-size to the conversation: roughly double what is in use."""
    want = max(4096, used_tokens * 2)
    for step in LADDER:
        if step >= want:
            return min(step, max_safe or step)
    return min(LADDER[-1], max_safe or LADDER[-1])
