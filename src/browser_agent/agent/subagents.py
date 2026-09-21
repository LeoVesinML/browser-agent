"""Sub-agent architecture.

The main agent plans and acts; reading a lot of material is delegated. The
sub-agent shares the same browser - it sees exactly the page the main agent left
it on - but runs in its own conversation, with its own context window and a
read-only tool set. Whatever it reads (a hundred search results, twenty message
bodies) is paid for once, in *its* context, and only the conclusion crosses back.

This is both a context strategy and a reliability one: the main agent's
transcript stays short enough to keep reasoning well over a long run.
"""

from __future__ import annotations

from typing import Any

from ..config import AgentConfig
from ..llm import LLM, text_of
from ..browser.session import BrowserSession
from .console import AgentConsole
from .prompts import SUBAGENT_SYSTEM
from .safety import SafetyPolicy
from .tools import Toolbox

PREFIX = "   [dim]│[/dim] "


def run_reading_subagent(
    llm: LLM,
    cfg: AgentConfig,
    session: BrowserSession,
    console: AgentConsole,
    instruction: str,
    max_steps: int,
    trace: Any = None,
) -> str:
    """Run a read-only sub-agent and return its answer as plain text."""
    safety = SafetyPolicy(cfg=cfg.safety, llm=None, confirm=None)
    box = Toolbox(session, cfg, safety)
    tools = box.schemas(readonly=True)

    page = session.active_page()
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"<current_page url=\"{page.url}\" title=\"{page.title()[:80]}\"/>\n"
                        f"<task>\n{instruction}\n</task>"
                    ),
                }
            ],
        }
    ]

    console.console.print(f"{PREFIX}[bold blue]sub-agent[/] {instruction[:110]}")
    if trace:
        trace.write("subagent_start", instruction=instruction)

    answer = ""
    for step in range(max_steps):
        response = llm.create(
            system=SUBAGENT_SYSTEM,
            messages=messages,
            tools=tools,
            max_tokens=4_000,
            effort="low",
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        answer = text_of(response.content) or answer
        if not tool_uses:
            break

        results = []
        for call in tool_uses:
            console.tool_call(call.name, dict(call.input), prefix=PREFIX)
            outcome = box.dispatch(call.name, dict(call.input))
            console.tool_result(
                outcome.content if isinstance(outcome.content, str) else "<image>",
                outcome.is_error,
                prefix=PREFIX,
                lines=2,
            )
            if trace:
                trace.write("subagent_tool", tool=call.name, args=dict(call.input), error=outcome.is_error)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            )
        messages.append({"role": "user", "content": results})

        # Hard stop: a reading sub-agent that has not concluded by its budget
        # gets one forced turn to answer with what it has.
        if step == max_steps - 2:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Step budget is nearly spent. Answer now with what you have, "
                            "and state explicitly what you could not check.",
                        }
                    ],
                }
            )

    if trace:
        trace.write("subagent_done", answer=answer[:2000])
    console.console.print(f"{PREFIX}[blue]↩ returned {len(answer)} chars[/blue]")
    return answer or "(the sub-agent returned nothing)"
