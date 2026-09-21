"""A scripted stand-in for the model, so the loop can be tested end to end
without spending money or depending on the network."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from browser_agent.llm import Usage


@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    thinking: str = ""


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeResponse:
    content: list[Block]
    stop_reason: str = "tool_use"
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeLLM:
    """`script` is a list of callables: (last_tool_results) -> list[Block]."""

    def __init__(self, script: list[Callable[[list[Any]], list[Block]]]) -> None:
        self.script = script
        self.calls = 0
        self.usage = Usage()
        self._ids = itertools.count(1)
        self.text_calls: list[str] = []

    def next_id(self) -> str:
        return f"tu_{next(self._ids)}"

    def create(self, *, system, messages, tools=None, model=None, max_tokens=None,
               effort=None, tool_choice=None) -> FakeResponse:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        blocks = step(messages)
        # Real tool_use ids are unique per call; the scripts do not care about the
        # value, and reusing one would hide bugs in id-keyed bookkeeping.
        for block in blocks:
            if block.type == "tool_use":
                block.id = self.next_id()
        self.usage.add(model or "fake", FakeUsage())
        stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
        return FakeResponse(blocks, stop)

    def count_tokens(self, **_kwargs) -> int | None:
        return None

    def text(self, *, system, prompt, model=None, max_tokens=1500, effort=None) -> str:
        self.text_calls.append(prompt)
        return "GOAL: test\nDONE: 1. something\nFACTS: none\nSTATE: page\nPENDING: finish\nPITFALLS: none"
