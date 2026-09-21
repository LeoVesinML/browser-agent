"""The MCP surface must drive the same browser correctly from an asyncio host."""

from __future__ import annotations

import asyncio

import pytest

from browser_agent.config import AgentConfig

pytest.importorskip("mcp")


def test_mcp_tools_drive_the_browser(tmp_path, demo_url):
    from browser_agent import mcp_server

    cfg = AgentConfig()
    cfg.browser.profile_dir = tmp_path / "profile"
    cfg.browser.headless = True
    cfg.browser.start_url = f"{demo_url}/mail.html"
    server = mcp_server.build_server(cfg)

    async def drive():
        names = {t.name for t in await server.list_tools()}
        assert "browser_snapshot" in names and "finish" not in names
        await server.call_tool("browser_wait", {"condition": "text", "value": "Входящие"})
        result = await server.call_tool("browser_snapshot", {})
        return str(result)

    try:
        out = asyncio.run(drive())
        assert "Входящие" in out and "[e" in out
    finally:
        box = mcp_server._state.pop("box", None)
        if box:
            mcp_server._pool.submit(box.session.close).result(timeout=30)
