from __future__ import annotations

import pytest

from browser_agent.agent.safety import SafetyPolicy
from browser_agent.config import SafetyConfig


def policy(mode="ask", answers=None):
    seen = []

    def confirm(tool, args, verdict):
        seen.append((tool, verdict.risk))
        return (answers or {}).get(verdict.risk, False)

    p = SafetyPolicy(cfg=SafetyConfig(mode=mode, use_llm_judge=False), confirm=confirm)
    p.seen = seen  # type: ignore[attr-defined]
    return p


@pytest.mark.parametrize(
    "label,expected",
    [
        ("button \"Оплатить заказ\"", "high"),
        ("button \"Place order\"", "high"),
        ("button \"Удалить\"", "medium"),
        ("button \"Это спам\"", "medium"),
        ("button \"Откликнуться\"", "medium"),
        ("link \"Показать ещё\"", "low"),
        ("button \"Фильтры\"", "low"),
    ],
)
def test_risk_is_read_off_the_control_itself(label, expected):
    p = policy()
    assert p.assess("browser_click", {"ref": "e1"}, label, "https://shop.test/menu").risk == expected


def test_reading_is_never_gated():
    p = policy()
    assert p.assess("browser_snapshot", {}, "", "https://x.test").risk == "none"
    assert p.assess("browser_read_text", {}, "", "https://x.test").risk == "none"


def test_card_numbers_are_hard_blocked_even_if_the_user_would_approve():
    p = policy(answers={"high": True, "medium": True, "low": True})
    allowed, verdict = p.gate(
        "browser_type", {"ref": "e1", "text": "4276 1600 1234 5678"}, "textbox \"Карта\"", "u"
    )
    assert not allowed and verdict.source == "policy"
    assert "user's to enter" in verdict.reason


def test_yolo_never_asks_but_strict_asks_about_everything():
    assert policy("yolo").gate("browser_click", {}, "button \"Оплатить\"", "u")[0] is True
    p = policy("strict")
    p.gate("browser_click", {}, "link \"Каталог\"", "https://x.test")
    assert p.seen  # even a low-risk click was put to the user


def test_always_allow_short_circuits_repeat_questions():
    p = policy(answers={"medium": True})
    label = "button \"Удалить\""
    assert p.gate("browser_click", {"ref": "e1"}, label, "u")[0] is True
    p.always_allow.add(f"browser_click|{label}")
    p.confirm = None
    assert p.gate("browser_click", {"ref": "e2"}, label, "u")[0] is True
