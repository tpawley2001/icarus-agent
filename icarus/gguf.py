"""Minimal GGUF metadata reader — enough to size a context window safely.

We need three things before offering to relaunch a model with a bigger window:

  * the context length it was actually trained for (going past it degrades
    output long before it errors),
  * the KV-cache geometry, so we can predict VRAM cost per token, and
  * the file size, which is roughly the weight footprint.

Reads only the header, so it is fast even on a 20 GB file. stdlib only.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# GGUF value type enum.
(U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64) = range(13)

_FIXED = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}


class GGUFError(RuntimeError):
    pass


def _read_value(f, vtype: int) -> Any:
    if vtype in _FIXED:
        fmt, size = _FIXED[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if vtype == ARRAY:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        # Arrays are usually the tokenizer vocab — hundreds of thousands of
        # entries we never look at. Skip past instead of materialising them.
        if etype in _FIXED:
            f.seek(_FIXED[etype][1] * n, 1)
            return f"<array:{n}>"
        if etype == STRING:
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)
            return f"<array:{n}>"
        raise GGUFError(f"unsupported array element type {etype}")
    raise GGUFError(f"unsupported value type {vtype}")


def read_metadata(path: str | Path, max_keys: int = 200) -> Dict[str, Any]:
    p = Path(path)
    with p.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError(f"{p} is not a GGUF file")
        struct.unpack("<I", f.read(4))[0]        # version
        struct.unpack("<Q", f.read(8))[0]        # tensor count
        n_kv = struct.unpack("<Q", f.read(8))[0]

        meta: Dict[str, Any] = {}
        for _ in range(min(n_kv, max_keys)):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8", "replace")
            vtype = struct.unpack("<I", f.read(4))[0]
            try:
                meta[key] = _read_value(f, vtype)
            except GGUFError:
                break
    return meta


@dataclass
class ModelShape:
    arch: str = ""
    trained_ctx: int = 0
    n_layers: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    embedding_length: int = 0
    key_length: int = 0
    value_length: int = 0
    file_bytes: int = 0
    # Sliding-window attention (Gemma 3/4 and friends). When present, most
    # layers only keep a fixed window of KV instead of the whole context, so
    # the naive all-layers-times-context figure overstates cost by ~5x.
    sliding_window: int = 0
    key_length_swa: int = 0
    value_length_swa: int = 0
    shared_kv_layers: int = 0

    # Gemma interleaves 5 local layers per 1 global; llama.cpp's gemma3 path
    # treats every 6th layer as full attention.
    SWA_INTERLEAVE = 6

    @property
    def head_dim(self) -> int:
        if self.key_length:
            return self.key_length
        if self.n_head and self.embedding_length:
            return self.embedding_length // self.n_head
        return 0

    @property
    def uses_swa(self) -> bool:
        return bool(self.sliding_window and self.key_length_swa)

    @property
    def kv_layers(self) -> int:
        """Layers that allocate their own KV (some models share across layers)."""
        return max(1, self.n_layers - max(0, self.shared_kv_layers))

    def kv_bytes_per_token(self, bits: int = 16) -> int:
        """Marginal KV cost of one more token of context.

        For an SWA model this counts only the global layers — the local ones
        are capped by the window and do not grow with the context.
        """
        kd = self.key_length or self.head_dim
        vd = self.value_length or self.head_dim
        if not (self.n_layers and self.n_head_kv and kd and vd):
            return 0
        b = bits / 8
        if self.uses_swa:
            n_global = max(1, self.kv_layers // self.SWA_INTERLEAVE)
            return int(n_global * self.n_head_kv * (kd + vd) * b)
        return int(self.kv_layers * self.n_head_kv * (kd + vd) * b)

    def kv_fixed_bytes(self, ctx: int, bits: int = 16) -> int:
        """KV held by windowed layers — bounded by the window, not the context."""
        if not self.uses_swa:
            return 0
        b = bits / 8
        n_global = max(1, self.kv_layers // self.SWA_INTERLEAVE)
        n_local = max(0, self.kv_layers - n_global)
        span = min(max(0, ctx), self.sliding_window)
        return int(n_local * self.n_head_kv *
                   (self.key_length_swa + self.value_length_swa) * span * b)

    def kv_bytes(self, ctx: int, bits: int = 16) -> int:
        return self.kv_bytes_per_token(bits) * max(0, ctx) + self.kv_fixed_bytes(ctx, bits)


def shape_of(path: str | Path) -> Optional[ModelShape]:
    """Best-effort architecture summary. Returns None if unreadable."""
    try:
        meta = read_metadata(path)
    except Exception:
        return None
    arch = str(meta.get("general.architecture") or "")

    def g(*suffixes, default=0):
        # Keys are namespaced by architecture, e.g. "gemma3.block_count".
        for s in suffixes:
            for key in (f"{arch}.{s}", s):
                v = meta.get(key)
                if isinstance(v, (int, float)):
                    return int(v)
        return default

    try:
        size = Path(path).stat().st_size
    except OSError:
        size = 0

    shape = ModelShape(
        arch=arch,
        trained_ctx=g("context_length"),
        n_layers=g("block_count"),
        n_head=g("attention.head_count"),
        n_head_kv=g("attention.head_count_kv"),
        embedding_length=g("embedding_length"),
        key_length=g("attention.key_length"),
        value_length=g("attention.value_length"),
        file_bytes=size,
        sliding_window=g("attention.sliding_window"),
        key_length_swa=g("attention.key_length_swa"),
        value_length_swa=g("attention.value_length_swa"),
        shared_kv_layers=g("attention.shared_kv_layers"),
    )
    # Multi-query models omit head_count_kv; it defaults to head_count.
    if not shape.n_head_kv:
        shape.n_head_kv = shape.n_head
    return shape
