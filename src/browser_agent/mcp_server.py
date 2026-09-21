"""The same browser toolset, exposed over MCP.

The agent in this repo does *not* talk to the browser over MCP - it calls the
toolbox in-process, which is faster and lets the harness gate and render each
action. MCP is for the other direction: letting an external client (Claude
Desktop, Claude Code, another agent) drive the very same browser.

Run it with:  python -m browser_agent.mcp_server

Implementation note: Playwright's sync API must be used from the thread that
created it, while an MCP server is asyncio. Everything browser-related therefore
runs on one dedicated worker thread and calls are marshalled onto it.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp.server.mcpserver import MCPServer

from .agent.safety import SafetyPolicy
from .agent.tools import TOOL_SCHEMAS, Toolbox
from .browser.session import BrowserSession
from .config import AgentConfig

# Tools that only make sense inside our own agent loop.
EXCLUDED = {"delegate_reading", "ask_user", "finish", "note"}

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="browser")
_state: dict[str, Any] = {}


def _boot(cfg: AgentConfig) -> Toolbox:
    session = BrowserSession(
        user_data_dir=cfg.browser.profile_dir, headless=cfg.browser.headless
    )
    session.start(cfg.browser.start_url)
    # No terminal to confirm on: this process never auto-approves risky actions,
    # the connected client is responsible for its own approval UX.
    safety = SafetyPolicy(cfg=cfg.safety, llm=None, confirm=lambda *_: False)
    return Toolbox(session, cfg, safety)


def _call(name: str, args: dict[str, Any]) -> str:
    box: Toolbox = _state["box"]
    outcome = box.dispatch(name, args)
    if isinstance(outcome.content, str):
        return outcome.content
    return json.dumps(outcome.content, ensure_ascii=False)[:4000]


TYPE_MAP = {
    "string": "str", "integer": "int", "number": "float",
    "boolean": "bool", "array": "list[str]", "object": "dict",
}


def _describe(schema: dict[str, Any]) -> str:
    """Fold the parameter docs into the description: MCPServer derives the JSON
    schema from the Python signature, which carries types but not prose."""
    props: dict[str, Any] = schema["input_schema"]["properties"]
    if not props:
        return schema["description"]
    lines = [schema["description"], "", "Parameters:"]
    for name, spec in props.items():
        flag = "" if name in schema["input_schema"]["required"] else " (optional)"
        enum = f" one of {spec['enum']}." if "enum" in spec else ""
        lines.append(f"- {name}{flag}: {spec.get('description', '')}{enum}".rstrip())
    return "\n".join(lines)


def _make_handler(schema: dict[str, Any], invoke: Any) -> Any:
    """Build a function whose real signature matches the tool's JSON schema."""
    props: dict[str, Any] = schema["input_schema"]["properties"]
    required: list[str] = schema["input_schema"]["required"]
    params = []
    for name, spec in props.items():
        annotation = TYPE_MAP.get(spec.get("type", "string"), "str")
        params.append(
            f"{name}: {annotation}" if name in required
            else f"{name}: {annotation} | None = None"
        )
    src = (
        f"async def {schema['name']}({', '.join(params)}) -> str:\n"
        f"    return await _invoke({schema['name']!r}, dict(locals()))\n"
    )
    namespace: dict[str, Any] = {"_invoke": invoke}
    exec(compile(src, f"<mcp:{schema['name']}>", "exec"), namespace)  # noqa: S102
    return namespace[schema["name"]]


def build_server(cfg: AgentConfig | None = None) -> MCPServer:
    cfg = cfg or AgentConfig()
    server = MCPServer(
        name="browser-agent",
        instructions=(
            "Drive a real, visible Chromium. Call browser_snapshot to see the page as a "
            "semantic outline with [eNN] handles, then act on those handles. Handles go "
            "stale after navigation - re-snapshot."
        ),
    )

    async def invoke(name: str, kwargs: dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        if "box" not in _state:
            _state["box"] = await loop.run_in_executor(_pool, _boot, cfg)
        args = {k: v for k, v in kwargs.items() if v is not None}
        return await loop.run_in_executor(_pool, _call, name, args)

    for schema in TOOL_SCHEMAS:
        if schema["name"] in EXCLUDED:
            continue
        server.add_tool(
            _make_handler(schema, invoke),
            name=schema["name"],
            description=_describe(schema),
            structured_output=False,
        )
    return server


def main() -> None:  # pragma: no cover - process entry point
    build_server().run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
