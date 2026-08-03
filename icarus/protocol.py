"""Text tool-call protocol — the fallback for models with no tool template.

Roughly a third of the models on a typical llama-swap box are merges or base
models whose chat template has no ``tools`` support. Handing them an OpenAI
``tools`` array produces either a 500 or a confident hallucination in prose.

For those, Icarus describes the tools in the system prompt and asks for a
fenced JSON block instead, then parses it back into the same shape the native
path produces. From the agent loop's perspective the two are identical.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Tuple

# Fenced ```icarus blocks are the documented form. We also accept a bare
# ```json fence and a raw {"tool": ...} object, because small models drift.
FENCE = re.compile(
    r"```(?:icarus|json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)

# A model with a tool-calling template that is NOT being given a `tools` array
# will often fall back to emitting its trained special-token format as literal
# text, because llama.cpp only activates the matching parser when tools are
# passed. Each family leaks a different shape, so the fallback has to read all
# of them or the text protocol is useless on exactly the models that need it.
QWEN_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
LLAMA_FN = re.compile(r"<function\s*=\s*([\w.-]+)\s*>\s*(\{.*?\})\s*</function>", re.DOTALL)
MISTRAL = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\]|\{.*?\})", re.DOTALL)
# gemma: <|tool_call>call:read_file{path:<|"|>notes.txt<|"|>}<tool_call|>
GEMMA_TAG = re.compile(
    r"<\|?tool_call\|?>\s*(?:call:)?\s*([\w.-]+)\s*(\{.*?\})\s*<\/?\|?tool_call\|?>",
    re.DOTALL,
)
GEMMA_QUOTE = re.compile(r'<\|"\|>')
# Bare `name{...}` pseudo-JSON with unquoted keys, after quote markers are fixed.
LOOSE_PAIR = re.compile(r'([\w.-]+)\s*:\s*("(?:[^"\\]|\\.)*"|[^,{}]+)')


def render_instructions(tools: List[Dict[str, Any]]) -> str:
    """System-prompt section teaching the text protocol."""
    lines = [
        "## Calling tools",
        "",
        "You cannot call tools natively. To call one, reply with ONLY a fenced",
        "block in exactly this form and nothing else:",
        "",
        "```icarus",
        '{"tool": "<name>", "args": {"<key>": "<value>"}}',
        "```",
        "",
        "Rules:",
        "- One tool call per reply. Stop immediately after the closing fence.",
        "- Do not explain the call first; just emit the block.",
        "- After the result comes back you may call another tool or answer.",
        "- When you have the answer, reply in plain prose with no fenced block.",
        "",
        "### Available tools",
        "",
    ]
    for t in tools:
        fn = t.get("function", t)
        params = (fn.get("parameters") or {}).get("properties") or {}
        required = set((fn.get("parameters") or {}).get("required") or [])
        lines.append(f"**{fn.get('name')}** — {fn.get('description', '').strip()}")
        for pname, pspec in params.items():
            req = "required" if pname in required else "optional"
            desc = (pspec or {}).get("description", "")
            lines.append(f"  - `{pname}` ({(pspec or {}).get('type','any')}, {req}) {desc}")
        lines.append("")
    return "\n".join(lines)


def _loose_object(blob: str) -> Dict[str, Any]:
    """Parse pseudo-JSON that small models emit: unquoted keys, odd quoting."""
    cleaned = GEMMA_QUOTE.sub('"', blob).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    out: Dict[str, Any] = {}
    for k, v in LOOSE_PAIR.findall(cleaned.strip("{} \n")):
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] == '"':
            try:
                out[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                v = v[1:-1]
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        elif re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _mk(name: str, args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = _loose_object(args)
    if not isinstance(args, dict):
        args = {}
    return {
        "id": f"txt_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def parse(content: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Split a text reply into (prose, tool_calls).

    Returns tool_calls in native OpenAI shape so the loop needs no special case.
    Recognises the documented fenced form plus the special-token formats the
    major local families leak when their tool template fires without a `tools`
    array. First match wins — one call per turn keeps small models tractable.
    """
    if not content:
        return "", []

    def strip(span: str, call: Dict[str, Any]):
        return content.replace(span, "").strip(), [call]

    # 1. The documented fenced form.
    for m in FENCE.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("tool") or obj.get("name") or obj.get("function")
        if isinstance(name, str) and name:
            args = obj.get("args", obj.get("arguments", obj.get("parameters", {})))
            return strip(m.group(0), _mk(name, args))

    # 2. Qwen / Hermes: <tool_call>{"name":..., "arguments":{...}}</tool_call>
    m = QWEN_TAG.search(content)
    if m:
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("tool")
            if isinstance(name, str) and name:
                return strip(m.group(0), _mk(name, obj.get("arguments", obj.get("args", {}))))
        except json.JSONDecodeError:
            pass

    # 3. Gemma: <|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>
    m = GEMMA_TAG.search(content)
    if m:
        return strip(m.group(0), _mk(m.group(1), _loose_object(m.group(2))))

    # 4. Llama 3.1: <function=name>{...}</function>
    m = LLAMA_FN.search(content)
    if m:
        return strip(m.group(0), _mk(m.group(1), m.group(2)))

    # 5. Mistral: [TOOL_CALLS] [{"name":..., "arguments":{...}}]
    m = MISTRAL.search(content)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list) and obj:
                obj = obj[0]
            name = obj.get("name") or obj.get("tool")
            if isinstance(name, str) and name:
                return strip(m.group(0), _mk(name, obj.get("arguments", obj.get("args", {}))))
        except (json.JSONDecodeError, AttributeError):
            pass

    return content.strip(), []
