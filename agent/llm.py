"""Model wrapper, plus a scripted fake for development.

ScriptedLLM replays fixed responses so the loop can be built and tested without
a GPU. It also makes the awkward cases — malformed output, hallucinated tool
names, loops — reproducible, which is hard to arrange with a real model.

    llm = scripted("recovers_from_bad_column")
    llm = OllamaLLM()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol


class LLM(Protocol):
    name: str

    def __call__(self, prompt: str, max_tokens: int = 150) -> str: ...


@dataclass
class OllamaLLM:
    """Local Ollama server. `ollama pull qwen2.5-coder:3b` first."""

    # 3B not 7B: the card is a 6GB 3050 Laptop. A 7B at Q4 is ~4.7GB of weights
    # and leaves nothing for the KV cache — runs died a few steps in. The 3B is
    # also about 2x faster, which adds up over 200 overnight runs.
    name: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", "qwen2.5-coder:3b"))
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    temperature: float = field(default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.7")))
    quantisation: str = "Q4_K_M"

    def __call__(self, prompt: str, max_tokens: int = 150) -> str:
        import requests          # lazy so the stubs don't need it

        r = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                    "stop": ["\nObservation:", "\nOBSERVATION:"],
                },
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json().get("response", "")


@dataclass
class ScriptedLLM:
    """Returns the next canned response on each call.

    Falls through to a final_answer once the script runs out so a loop can't
    hang waiting for more.
    """

    script: list[str]
    name: str = "scripted-fake"
    quantisation: str = "none"
    temperature: float = 0.0
    calls: int = 0

    def __call__(self, prompt: str, max_tokens: int = 150) -> str:
        if self.calls < len(self.script):
            out = self.script[self.calls]
        else:
            out = 'Thought: I have the answer.\nAction: final_answer\nInput: {"answer": "done"}'
        self.calls += 1
        return out

    def reset(self) -> None:
        self.calls = 0


def _step(thought: str, tool: str, args: dict) -> str:
    return (f"Thought: {thought}\n"
            f"Action: {tool}\n"
            f"Input: {json.dumps(args)}")


def _final(thought: str, answer: str) -> str:
    return _step(thought, "final_answer", {"answer": answer})


# Spider's chinook_1 uses singular capitalised table names (Album, Track), not
# the plural lowercase of the original Chinook schema. test_stubs checks that
# the success scenarios still execute, so this can't silently rot.
SCENARIOS: dict[str, list[str]] = {

    "clean_success": [
        _step("Let me see what tables exist.", "get_schema", {}),
        _step("I need the Album table.", "get_schema", {"table": "Album"}),
        _step("Count the albums.", "run_sql", {"query": "SELECT COUNT(*) AS n FROM Album"}),
        _final("The count came back.", "347 albums"),
    ],

    "recovers_from_bad_column": [
        _step("I will guess the column name.", "run_sql",
              {"query": "SELECT album_title FROM Album LIMIT 5"}),
        _step("Wrong column. The hint lists the real ones.", "get_schema", {"table": "Album"}),
        _step("It is called Title.", "run_sql", {"query": "SELECT Title FROM Album LIMIT 5"}),
        _final("Got the titles.", "For Those About To Rock We Salute You, ..."),
    ],

    "malformed_then_ok": [
        "I think we should probably look at the Album table first, honestly.",
        "```json\n{\"tool\": \"get_schema\", \"args\": {\"table\": \"Album\"}}\n```",
        _step("Now query it.", "run_sql", {"query": "SELECT COUNT(*) AS n FROM Album"}),
        _final("Done.", "347"),
    ],

    # third entry differs only in whitespace — action_hash has to normalise
    "repeats_itself": [
        _step("Try this.", "run_sql", {"query": "SELECT * FROM nonexistent"}),
        _step("Try again.", "run_sql", {"query": "SELECT * FROM nonexistent"}),
        _step("And again.", "run_sql", {"query": "SELECT  *  FROM   nonexistent"}),
        _step("Once more.", "run_sql", {"query": "SELECT * FROM nonexistent"}),
    ],

    "hallucinates_tool": [
        _step("I will use a tool that does not exist.", "query_database", {"sql": "SELECT 1"}),
        _step("Fine, the real one.", "run_sql", {"query": "SELECT COUNT(*) AS n FROM Album"}),
        _final("Done.", "347"),
    ],

    "keeps_failing": [
        _step("Attempt 1.", "run_sql", {"query": "SELECT bad FROM Album"}),
        _step("Attempt 2.", "run_sql", {"query": "SELECT worse FROM Album"}),
        _step("Attempt 3.", "run_sql", {"query": "SELECT terrible FROM Album"}),
        _step("Attempt 4.", "run_sql", {"query": "SELECT awful FROM Album"}),
    ],

    "never_finishes": [
        _step(f"Step {i}.", "get_schema", {"table": t})
        for i, t in enumerate(
            ["Album", "Artist", "Customer", "Employee", "Genre",
             "Invoice", "MediaType", "Playlist", "Track",
             "InvoiceLine", "Album", "Artist"], 1)
    ],

    # 3503 rows — truncation should kick in
    "large_result": [
        _step("Fetch every track.", "run_sql", {"query": "SELECT * FROM Track"}),
        _final("That was a lot of rows.", "3503 tracks"),
    ],
}


def scripted(name: str) -> ScriptedLLM:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}. Available: {sorted(SCENARIOS)}")
    return ScriptedLLM(list(SCENARIOS[name]))
