from __future__ import annotations

from browser_agent.agent.context import Conversation
from browser_agent.config import ContextConfig
from tests.fake_llm import FakeLLM


def _convo(**kwargs) -> Conversation:
    cfg = ContextConfig(**kwargs)
    return Conversation(cfg=cfg, llm=FakeLLM([]))


def _turn(convo: Conversation, i: int, payload: str) -> None:
    convo.add_assistant([{"type": "tool_use", "id": f"t{i}", "name": "browser_snapshot", "input": {}}])
    convo.add_tool_results(
        [{"type": "tool_result", "tool_use_id": f"t{i}", "content": payload, "is_error": False}]
    )
    convo.mark_observation(f"t{i}", f"snapshot {i}", i)


def test_pruning_collapses_old_observations_but_keeps_pairing():
    convo = _convo(keep_observations=2)
    convo.add_user_text("task")
    for i in range(5):
        _turn(convo, i, "PAGE " * 500)

    before = len(convo._serialised())
    assert convo.prune() == 3
    after = len(convo._serialised())
    assert after < before / 2  # 3 of 5 observations collapsed to one line each

    # every tool_result still answers a tool_use, which the API requires
    used = [b["id"] for m in convo.messages if m["role"] == "assistant" for b in m["content"]]
    got = [b["tool_use_id"] for m in convo.messages if m["role"] == "user"
           and isinstance(m["content"], list) for b in m["content"]
           if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert used == got
    assert "pruned to save context" in convo._serialised()


def test_pruning_is_idempotent():
    convo = _convo(keep_observations=1)
    convo.add_user_text("task")
    for i in range(3):
        _turn(convo, i, "PAGE " * 100)
    assert convo.prune() == 2
    assert convo.prune() == 0


def test_compaction_cuts_at_a_turn_boundary_and_keeps_notes():
    convo = _convo(keep_recent_turns=2)
    convo.add_user_text("original task")
    convo.notes["price"] = "590 ₽"
    for i in range(6):
        _turn(convo, i, f"page {i}")

    summary = convo.compact("original task", "http://example.test")
    assert summary
    assert convo.messages[0]["content"][0]["text"] == "original task"
    assert "context_handoff" in convo.messages[1]["content"][0]["text"]
    # the tail must start with an assistant turn, or tool_results would be orphaned
    assert convo.messages[2]["role"] == "assistant"
    # notes were handed to the summariser, so facts cannot be lost with the transcript
    assert "590 ₽" in convo.llm.text_calls[0]


def test_compaction_is_skipped_when_there_is_nothing_to_cut():
    convo = _convo(keep_recent_turns=6)
    convo.add_user_text("task")
    _turn(convo, 0, "page")
    assert convo.compact("task") == ""
