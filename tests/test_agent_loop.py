"""End-to-end test of the agent loop against the demo site, with a scripted
model. Exercises the real browser, the real tools, the real safety gate - only
the LLM is replaced, so the run is deterministic and free."""

from __future__ import annotations

from browser_agent.agent.console import AgentConsole
from browser_agent.agent.loop import BrowserAgent
from browser_agent.config import AgentConfig
from tests.fake_llm import Block, FakeLLM


def _find_ref(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line and "[e" in line:
            return line[line.rindex("[e") + 1 : line.rindex("]")]
    raise AssertionError(f"no handle for {needle!r} in:\n{text}")


def _last_result(messages) -> str:
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block["content"]
                    return content if isinstance(content, str) else str(content)
    return ""


def build_agent(session, monkeypatch, script, safety="ask", confirm="y"):
    cfg = AgentConfig()
    cfg.safety.use_llm_judge = False
    cfg.safety.mode = safety
    cfg.max_steps = 12
    console = AgentConsole(quiet=True)
    monkeypatch.setattr(console, "confirm_action", lambda *a, **k: confirm)
    agent = BrowserAgent(cfg, session, console)
    agent.llm = FakeLLM(script)
    agent.convo.llm = agent.llm
    agent.safety.llm = None
    return agent


def test_agent_completes_a_delete_task_through_the_gate(mail, monkeypatch):
    """Open a spam email, move it to trash through the site's confirm dialog,
    and report - all driven by tool calls, with the gate approving."""
    state = {}

    def step1(_messages):
        return [Block("tool_use", id="1", name="browser_snapshot", input={})]

    def step2(messages):
        state["ref"] = _find_ref(_last_result(messages), "Крипто-Доход")
        return [Block("tool_use", id="2", name="browser_click", input={"ref": state["ref"]})]

    def step3(messages):
        state["spam"] = _find_ref(_last_result(messages), "В корзину")
        return [Block("tool_use", id="3", name="browser_click", input={"ref": state["spam"]})]

    def step4(messages):
        state["modal"] = _last_result(messages)
        ref = _find_ref(state["modal"], "Переместить")
        return [Block("tool_use", id="4", name="browser_click", input={"ref": ref})]

    def step5(messages):
        state["after"] = _last_result(messages)
        return [
            Block("tool_use", id="5", name="note",
                  input={"action": "write", "key": "deleted", "value": "1 spam email"}),
        ]

    def step6(_messages):
        return [Block("tool_use", id="6", name="finish",
                      input={"report": "Удалил 1 спам-письмо.", "status": "completed"})]

    agent = build_agent(mail, monkeypatch, [step1, step2, step3, step4, step5, step6])
    result = agent.run("Удали спам")

    assert result.status == "completed"
    assert "MODAL DIALOG IS OPEN" in state["modal"]
    assert result.notes["deleted"] == "1 spam email"
    assert "Крипто-Доход" not in mail.active_page().inner_text("body")


def test_a_refused_gate_stops_the_action_and_tells_the_agent_why(mail, monkeypatch):
    def step1(_m):
        return [Block("tool_use", id="1", name="browser_snapshot", input={})]

    def step2(messages):
        ref = _find_ref(_last_result(messages), "Крипто-Доход")
        return [Block("tool_use", id="2", name="browser_click", input={"ref": ref})]

    def step3(messages):
        ref = _find_ref(_last_result(messages), "В корзину")
        return [Block("tool_use", id="3", name="browser_click", input={"ref": ref})]

    def step4(messages):
        assert "BLOCKED by the safety layer" in _last_result(messages)
        return [Block("tool_use", id="4", name="finish",
                      input={"report": "Пользователь не подтвердил удаление.", "status": "partial"})]

    agent = build_agent(mail, monkeypatch, [step1, step2, step3, step4], confirm="n")
    result = agent.run("Удали спам")
    assert result.status == "partial"
    # nothing was deleted
    assert "Это спам" in mail.active_page().inner_text("body")


def test_repeated_identical_calls_trigger_a_replan_nudge(mail, monkeypatch):
    def loopy(_messages):
        return [Block("tool_use", id="x", name="browser_find", input={"query": "не существует"})]

    def done(_messages):
        return [Block("tool_use", id="y", name="finish", input={"report": "ok", "status": "partial"})]

    agent = build_agent(mail, monkeypatch, [loopy, loopy, loopy, done])
    agent.run("что-то невозможное")
    text = str(agent.convo.messages)
    assert "Stop and" in text and "re-plan" in text


def test_stale_handles_produce_a_recoverable_error_not_a_crash(mail, monkeypatch):
    def step1(_m):
        return [Block("tool_use", id="1", name="browser_click", input={"ref": "e9999"})]

    def step2(messages):
        assert "fresh browser_snapshot" in _last_result(messages)
        return [Block("tool_use", id="2", name="finish", input={"report": "ok", "status": "partial"})]

    agent = build_agent(mail, monkeypatch, [step1, step2])
    assert agent.run("клик по несуществующему").status == "partial"


def test_context_stays_flat_over_a_long_run(mail, monkeypatch):
    """Twelve snapshots of the same page must not grow the transcript twelvefold."""
    sizes = []

    def snap(_m):
        sizes.append(len(agent.convo._serialised()))
        return [Block("tool_use", id="s", name="browser_snapshot", input={})]

    def done(_m):
        return [Block("tool_use", id="d", name="finish", input={"report": "ok", "status": "completed"})]

    agent = build_agent(mail, monkeypatch, [snap] * 10 + [done])
    agent.run("смотри на страницу много раз")
    growth = sizes[-1] - sizes[3]
    assert agent.convo.pruned_count >= 5
    assert growth < sizes[3], f"transcript grew by {growth} chars over 7 extra observations"


def test_a_reading_subagent_cannot_act(mail):
    """The sub-agent is only offered read-only schemas; this checks it is also
    physically unable to act if it asks for something else anyway."""
    from browser_agent.agent.safety import SafetyPolicy
    from browser_agent.agent.tools import Toolbox
    from browser_agent.config import AgentConfig, SafetyConfig

    cfg = AgentConfig()
    box = Toolbox(mail, cfg, SafetyPolicy(cfg=SafetyConfig(use_llm_judge=False)), readonly=True)
    names = {t["name"] for t in box.schemas(readonly=True)}
    assert "browser_click" not in names and "finish" not in names

    outcome = box.dispatch("browser_click", {"ref": "e1"})
    assert outcome.is_error and "not available to you" in outcome.content
    assert box.dispatch("finish", {"report": "x", "status": "completed"}).is_error


def test_a_rate_limit_is_waited_out_not_fatal(mail, monkeypatch):
    """Losing a run that was going fine because the provider rate-limited one
    call is the wrong trade: waiting is almost always cheaper than starting over."""
    import anthropic
    import browser_agent.agent.loop as loop_mod

    calls = {"n": 0}

    def flaky(_m):
        calls["n"] += 1
        if calls["n"] == 1:
            raise anthropic.RateLimitError(
                "rate limited",
                response=httpx_response(429),
                body=None,
            )
        return [Block("tool_use", name="finish", input={"report": "ok", "status": "completed"})]

    def httpx_response(status):
        import httpx2 as httpx

        return httpx.Response(status, request=httpx.Request("POST", "https://x.test"))

    waited: list[float] = []
    monkeypatch.setattr(loop_mod.time, "sleep", lambda s: waited.append(s))

    agent = build_agent(mail, monkeypatch, [flaky, flaky])
    result = agent.run("что-нибудь")

    assert result.status == "completed"
    assert waited, "the loop must back off rather than give up"
