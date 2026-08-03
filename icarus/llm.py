"""Chat-completions client — stdlib only, no `openai` package.

Written against llama.cpp-behind-llama-swap specifically:

* The first request after a swap pays the full model-load cost, so timeouts are
  measured in minutes, not seconds.
* Models served with ``--jinja`` that have a reasoning template return an EMPTY
  ``content`` unless thinking is turned off. We send every known kill-switch and
  strip any ``<think>`` block that leaks through anyway.
* Some builds emit tool calls as bare text even when handed a ``tools`` array;
  the caller (protocol.py) handles that, we just surface content faithfully.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
OPEN_THINK = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


# llama-swap returns 5xx while a model is being torn down or swapped in. These
# are transient by definition — one short retry turns a hard failure into a
# pause, which matters because /ctx and /model both trigger a teardown.
TRANSIENT = ("shutting down", "matrix is", "no healthy upstream",
             "connection refused", "loading", "502", "503")


def _is_transient(err: str) -> bool:
    low = err.lower()
    return any(t in low for t in TRANSIENT)


@dataclass
class Reply:
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    # Set when the user pressed Esc mid-stream; the content is whatever
    # arrived before the abort and is still worth keeping in history.
    interrupted: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def strip_thinking(text: str) -> str:
    """Remove reasoning spans that a template emitted despite being told not to."""
    if not text:
        return text
    out = THINK_BLOCK.sub("", text)
    # An unterminated <think> means the model was cut off mid-reasoning; there
    # is no answer after it, so drop the remainder rather than show the guts.
    out = OPEN_THINK.sub("", out)
    return out.strip()


@dataclass
class LLMClient:
    base_url: str
    api_key: str = "not-needed"
    request_timeout: int = 900
    disable_thinking: bool = True

    def _post(self, payload: Dict[str, Any], timeout: int, stream: bool):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream" if stream else "application/json",
            },
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            raise LLMError(f"HTTP {e.code} from model server: {detail}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"cannot reach model server at {self.base_url}: {e.reason}"
            ) from e
        except TimeoutError as e:
            raise LLMError(
                f"model server timed out after {timeout}s — a cold load of a "
                f"large model can exceed this; raise llama_swap.request_timeout"
            ) from e

    def _payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        stream: bool,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            p["tools"] = tools
            p["tool_choice"] = "auto"
        if self.disable_thinking:
            # Different builds honour different spellings; sending all of them
            # is harmless (unknown keys are ignored) and covers gemma/qwen/glm.
            p["enable_thinking"] = False
            p["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            p["reasoning_effort"] = "none"
        if stream:
            p["stream_options"] = {"include_usage": True}
        return p

    def complete(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: Optional[int] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Reply:
        payload = self._payload(
            model, messages, tools, temperature, top_p, max_tokens, stream
        )
        t = timeout or self.request_timeout
        for attempt in (0, 1):
            try:
                if stream:
                    return self._stream(payload, t, on_token, should_abort)
                with self._post(payload, t, stream=False) as r:
                    data = json.loads(r.read().decode("utf-8", "replace"))
                return self._to_reply(data)
            except LLMError as e:
                if attempt == 0 and _is_transient(str(e)):
                    time.sleep(3.0)
                    continue
                raise
        raise LLMError("unreachable")

    def _to_reply(self, data: Dict[str, Any]) -> Reply:
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        return Reply(
            content=strip_thinking(msg.get("content") or ""),
            tool_calls=list(msg.get("tool_calls") or []),
            finish_reason=choices[0].get("finish_reason") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            model=data.get("model") or "",
        )

    def _stream(
        self,
        payload: Dict[str, Any],
        timeout: int,
        on_token: Optional[Callable[[str], None]],
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Reply:
        reply = Reply()
        # Tool-call deltas arrive fragmented across chunks and must be
        # reassembled by index, not appended blindly.
        acc: Dict[int, Dict[str, Any]] = {}
        buf = ""
        in_think = False

        with self._post(payload, timeout, stream=True) as r:
            for raw in r:
                # Checked per chunk so Esc lands within one token of being
                # pressed; closing the response drops the upstream request.
                if should_abort is not None and should_abort():
                    reply.interrupted = True
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    d = json.loads(chunk)
                except json.JSONDecodeError:
                    continue

                if d.get("usage"):
                    reply.prompt_tokens = int(d["usage"].get("prompt_tokens") or 0)
                    reply.completion_tokens = int(d["usage"].get("completion_tokens") or 0)
                if d.get("model"):
                    reply.model = d["model"]

                for ch in d.get("choices") or []:
                    if ch.get("finish_reason"):
                        reply.finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or {}

                    for tc in delta.get("tool_calls") or []:
                        i = tc.get("index", 0)
                        slot = acc.setdefault(
                            i, {"id": "", "type": "function",
                                 "function": {"name": "", "arguments": ""}}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]

                    piece = delta.get("content") or ""
                    if not piece:
                        continue
                    buf += piece
                    # Suppress reasoning live rather than after the fact, so the
                    # user never watches a <think> block scroll past.
                    if on_token:
                        low = buf.lower()
                        if not in_think and "<think>" in low:
                            in_think = True
                            head = buf[: low.index("<think>")]
                            if head:
                                on_token(head)
                        elif in_think and "</think>" in low:
                            in_think = False
                            buf = buf[low.index("</think>") + 8 :]
                        elif not in_think:
                            on_token(piece)

        reply.content = strip_thinking(buf)
        reply.tool_calls = [acc[i] for i in sorted(acc)]
        return reply
