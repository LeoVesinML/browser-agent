"""Context management: the difference between an agent that runs 8 steps and one
that runs 80.

Three independent mechanisms, in increasing order of aggression:

1. **Compression at the source** - a page enters the transcript as a semantic
   outline or a diff, never as HTML (see ``browser/snapshot.py``).
2. **Observation pruning** - a page observation is only useful while it is
   current. Older ones are replaced by a one-line stub, which keeps the
   transcript flat no matter how many steps the run takes, while leaving the
   tool_use/tool_result pairing intact so the API stays happy.
3. **Compaction** - when the budget is still exceeded, the older half of the
   transcript is replaced by a written handoff note produced by the model
   itself, cut at a turn boundary so no tool_result is orphaned.

On top of that, an explicit scratchpad (``notes``) lets the agent park facts
outside the transcript, so pruning and compaction can never lose them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..config import ContextConfig
from ..llm import LLM

PRUNED_TEMPLATE = (
    "[observation from step {step} was pruned to save context: {label}. "
    "It is no longer current - call browser_snapshot if you need to see the page again.]"
)

COMPACTION_SYSTEM = """You compact the working memory of an autonomous browser agent.

Write a handoff note for the same agent to continue from. It must let the agent
carry on without the transcript you are summarising. Be specific and factual;
never invent progress that is not in the transcript.

Use exactly these sections:
GOAL: the user's task, restated precisely, including any constraints.
DONE: numbered list of steps already completed, with their outcomes.
FACTS: every concrete datum gathered so far (names, prices, ids, counts,
  addresses, findings). This is the part that must survive - be exhaustive.
STATE: the current page/tab and what is visible there.
PENDING: what still has to happen, in order.
PITFALLS: what has already failed and must not be retried the same way.
"""


@dataclass
class Conversation:
    cfg: ContextConfig
    llm: LLM
    messages: list[dict[str, Any]] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    compactions: int = 0
    pruned_count: int = 0

    _observations: list[tuple[str, str, int]] = field(default_factory=list, init=False)
    _pruned: set[str] = field(default_factory=set, init=False)
    _chars_per_token: float = field(default=3.1, init=False)
    _steps_since_verify: int = field(default=0, init=False)
    _last_exact: int | None = field(default=None, init=False)

    # ------------------------------------------------------------------ writing

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def add_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict[str, Any]]) -> None:
        self.messages.append({"role": "user", "content": results})

    def mark_observation(self, tool_use_id: str, label: str, step: int) -> None:
        """Flag a tool result as a page observation - i.e. safe to prune later."""
        self._observations.append((tool_use_id, label, step))

    # ------------------------------------------------------------------ pruning

    def prune(self) -> int:
        """Collapse all but the most recent page observations."""
        if len(self._observations) <= self.cfg.keep_observations:
            return 0
        stale = self._observations[: -self.cfg.keep_observations]
        targets = {tid: (label, step) for tid, label, step in stale if tid not in self._pruned}
        if not targets:
            return 0
        pruned = 0
        for message in self.messages:
            if message["role"] != "user" or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tid = block.get("tool_use_id")
                if tid in targets:
                    label, step = targets[tid]
                    block["content"] = PRUNED_TEMPLATE.format(step=step, label=label)
                    self._pruned.add(tid)
                    pruned += 1
        self.pruned_count += pruned
        return pruned

    # ------------------------------------------------------------------ sizing

    def _serialised(self) -> str:
        def default(obj: Any) -> Any:
            for attr in ("model_dump", "to_dict", "dict"):
                if hasattr(obj, attr):
                    try:
                        return getattr(obj, attr)()
                    except Exception:  # noqa: BLE001
                        continue
            return str(obj)

        return json.dumps(self.messages, default=default, ensure_ascii=False)

    def estimate_tokens(self) -> int:
        return int(len(self._serialised()) / self._chars_per_token)

    def tokens(self, system: Any, tools: list[dict[str, Any]] | None) -> int:
        """Estimated prompt size, re-calibrated against the API now and then.

        We do not use tiktoken: it is the wrong tokenizer for Claude and
        undercounts Cyrillic badly. Instead we keep a cheap char-based estimate
        honest by periodically checking it against ``messages.count_tokens``.
        """
        self._steps_since_verify += 1
        if self._steps_since_verify >= self.cfg.verify_tokens_every or self._last_exact is None:
            exact = self.llm.count_tokens(system=system, messages=self.messages, tools=tools)
            self._steps_since_verify = 0
            if exact:
                chars = len(self._serialised())
                if exact > 0:
                    self._chars_per_token = max(1.5, min(6.0, chars / exact))
                self._last_exact = exact
                return exact
        return self.estimate_tokens()

    # --------------------------------------------------------------- compaction

    def needs_compaction(self, token_count: int) -> bool:
        return token_count > self.cfg.budget_tokens * self.cfg.compact_at

    def compact(self, task: str, current_url: str = "") -> str:
        """Replace the older transcript with a model-written handoff note."""
        cut = self._turn_boundary()
        if cut is None:
            return ""

        head, tail = self.messages[:cut], self.messages[cut:]
        transcript = self._render_for_summary(head)
        notes = (
            "\n".join(f"- {k}: {v}" for k, v in self.notes.items())
            or "(the agent saved no notes)"
        )
        summary = self.llm.text(
            system=COMPACTION_SYSTEM,
            prompt=(
                f"<task>{task}</task>\n"
                f"<current_url>{current_url}</current_url>\n"
                f"<agent_notes>\n{notes}\n</agent_notes>\n"
                f"<transcript>\n{transcript}\n</transcript>"
            ),
            max_tokens=2_000,
        )
        self.messages = [
            self.messages[0],
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<context_handoff>\nEarlier steps were compacted to stay inside the "
                            "context budget. This note replaces them; treat it as your own memory "
                            f"of what happened.\n\n{summary}\n</context_handoff>"
                        ),
                    }
                ],
            },
            *tail,
        ]
        self.compactions += 1
        self._steps_since_verify = self.cfg.verify_tokens_every  # force a re-measure
        return summary

    def _turn_boundary(self) -> int | None:
        """Index of an assistant message that starts a self-contained tail.

        Cutting at an assistant message guarantees every tool_result in the tail
        still has its tool_use, which the API requires.
        """
        assistant_idx = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        if len(assistant_idx) <= self.cfg.keep_recent_turns:
            return None
        cut = assistant_idx[-self.cfg.keep_recent_turns]
        return cut if cut > 1 else None

    @staticmethod
    def _render_for_summary(messages: list[dict[str, Any]], limit: int = 24_000) -> str:
        lines: list[str] = []
        for message in messages:
            role = message["role"]
            content = message.get("content")
            if isinstance(content, str):
                lines.append(f"[{role}] {content}")
                continue
            for block in content or []:
                btype = getattr(block, "type", None) or (
                    block.get("type") if isinstance(block, dict) else None
                )
                if btype == "text":
                    text = getattr(block, "text", None) or block.get("text", "")
                    lines.append(f"[{role}] {text}")
                elif btype == "tool_use":
                    name = getattr(block, "name", None) or block.get("name")
                    args = getattr(block, "input", None) or block.get("input")
                    lines.append(f"[tool_call] {name}({json.dumps(args, ensure_ascii=False)[:300]})")
                elif btype == "tool_result":
                    body = block.get("content") if isinstance(block, dict) else ""
                    if isinstance(body, list):
                        body = " ".join(
                            b.get("text", "") for b in body if isinstance(b, dict)
                        )
                    lines.append(f"[tool_result] {str(body)[:600]}")
        rendered = "\n".join(lines)
        if len(rendered) > limit:
            # Keep the beginning (the task) and the end (recent progress).
            rendered = rendered[: limit // 3] + "\n…[middle omitted]…\n" + rendered[-2 * limit // 3 :]
        return rendered
