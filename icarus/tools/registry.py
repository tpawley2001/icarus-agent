"""Tool registry.

A tool is a python callable plus a JSON-schema description. The same schema
feeds both the native ``tools`` array and protocol.py's text fallback, so a
tool is written once and works on every model.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[..., str]
    # Tools that change the world are surfaced differently in the UI and can be
    # gated behind approval.
    mutates: bool = False


@dataclass
class ToolResult:
    ok: bool
    content: str
    display: str = ""  # short one-liner for the transcript


class Registry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, parameters: Dict[str, Any],
             mutates: bool = False):
        """Decorator form."""
        def deco(fn: Callable[..., str]) -> Callable[..., str]:
            self.add(Tool(name, description, parameters, fn, mutates))
            return fn
        return deco

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def all(self) -> List[Tool]:
        return [self._tools[n] for n in self.names()]

    def enabled(self, include: List[str], exclude: List[str]) -> List[Tool]:
        out = []
        for t in self.all():
            if include and t.name not in include:
                continue
            if t.name in (exclude or []):
                continue
            out.append(t)
        return out

    def schemas(self, tools: Optional[List[Tool]] = None) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in (tools if tools is not None else self.all())
        ]

    def dispatch(self, name: str, arguments: str | Dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                False,
                f"No such tool: {name!r}. Available: {', '.join(self.names())}",
                f"unknown tool {name}",
            )
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult(
                    False,
                    f"Arguments for {name} were not valid JSON ({e}). "
                    f"Received: {arguments[:300]}",
                    f"{name}: bad arguments",
                )
        else:
            args = arguments or {}
        if not isinstance(args, dict):
            return ToolResult(False, f"Arguments for {name} must be an object.", f"{name}: bad arguments")

        try:
            out = tool.run(**args)
        except TypeError as e:
            # Wrong/missing parameters — tell the model precisely what it sent
            # so it can correct on the next iteration instead of looping.
            return ToolResult(
                False,
                f"{name} rejected these arguments: {e}. "
                f"Schema: {json.dumps(tool.parameters)}",
                f"{name}: bad arguments",
            )
        except Exception as e:
            return ToolResult(
                False,
                f"{name} raised {type(e).__name__}: {e}\n"
                + "".join(traceback.format_exc(limit=3)),
                f"{name}: {type(e).__name__}",
            )

        if isinstance(out, ToolResult):
            return out
        return ToolResult(True, str(out), f"{name}: ok")
