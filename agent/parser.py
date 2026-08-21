"""Pulls an Action out of the model's raw output.

We don't constrain generation with a grammar. Tested on Qwen: forcing a strict
tool-call schema gave 100% valid JSON but roughly halved the quality of the
decisions inside it. Better to let it write freely and deal with the occasional
parse failure here.

TODO(day4): repair pass — feed the parser error back and retry once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from contracts import Action

_ACTION_RE = re.compile(r"^\s*Action\s*:\s*([A-Za-z_][\w]*)", re.M | re.I)
_INPUT_RE = re.compile(r"^\s*Input\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
                       re.M | re.I | re.S)
_THOUGHT_RE = re.compile(r"^\s*Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Input|Observation)\s*:|\Z)",
                         re.M | re.I | re.S)
_FENCE_RE = re.compile(r"```(?:json|python|sql)?\s*(.*?)```", re.S)


@dataclass
class ParseResult:
    thought: str
    action: Action | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.action is not None


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text


def _extract_thought(text: str) -> str:
    """Get the reasoning, with or without the 'Thought:' label.

    Qwen usually includes the label now that the prompt shows an example, but
    it dropped it entirely before that, so handle both.
    """
    m = _THOUGHT_RE.search(text)
    if m:
        return " ".join(m.group(1).split())

    head = re.split(r"^\s*(?:Action|Input|Observation)\s*:", text, maxsplit=1,
                    flags=re.M | re.I)[0]
    head = re.sub(r"^\s*Thought\b\s*", "", head, flags=re.I)
    head = _FENCE_RE.sub("", head)
    return " ".join(head.split())


def _last_json_object(text: str) -> dict | None:
    """Find the last balanced {...} and parse it.

    Last, not first, because models often write prose containing braces before
    the real payload. Depth matching rather than regex so nested objects work.
    """
    for end in range(len(text) - 1, -1, -1):
        if text[end] != "}":
            continue
        depth, in_str, esc = 0, False, False
        for start in range(end, -1, -1):
            ch = text[start]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:end + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def parse_action(raw: str) -> ParseResult:
    """Extract Thought and Action. Returns action=None if it can't."""
    text = raw.strip()
    if not text:
        return ParseResult("", None, "empty model output")

    thought = _extract_thought(text)

    # normal case: Action: <tool> / Input: <json>
    a = _ACTION_RE.search(text)
    if a:
        tool = a.group(1).strip()
        i = _INPUT_RE.search(text[a.end():])
        args: dict = {}
        if i:
            payload = _strip_fences(i.group(1).strip())
            obj = _last_json_object(payload)
            if obj is None:
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    return ParseResult(
                        thought, None,
                        f"Action '{tool}' found but Input was not valid JSON: "
                        f"{payload[:120]!r}")
            args = obj if isinstance(obj, dict) else {}

        if tool == "final_answer":
            answer = args.get("answer") or args.get("final_answer") or ""
            return ParseResult(thought, Action(tool="final_answer", args=args,
                                               is_final=True,
                                               final_answer=str(answer)))
        return ParseResult(thought, Action(tool=tool, args=args))

    # fallback: bare {"tool": ..., "args": {...}}
    obj = _last_json_object(_strip_fences(text))
    if isinstance(obj, dict) and "tool" in obj:
        thought = thought or ""
        tool = str(obj["tool"])
        args = obj.get("args") or obj.get("input") or {}
        if not isinstance(args, dict):
            args = {}
        if tool == "final_answer":
            return ParseResult(thought, Action(
                tool="final_answer", args=args, is_final=True,
                final_answer=str(args.get("answer", ""))))
        return ParseResult(thought, Action(tool=tool, args=args))

    return ParseResult(thought, None,
                       "no Action found — expected 'Action: <tool>' followed by "
                       "'Input: {json}'")
