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


def test_descriptive_text_does_not_trip_the_action_rules():
    """A link that *describes* a section is not an action. Matching command
    words inside long prose was a real false positive on the first live run:
    'Почта · Входящие, поиск, чтение, удаление с подтверждением'."""
    p = policy()
    long_link = 'link "Почта · Входящие, поиск, чтение, удаление с подтверждением"'
    assert p.assess("browser_click", {"ref": "e1"}, long_link, "https://x.test/").risk == "low"
    # the same word as an actual control name still stops the agent
    assert p.assess("browser_click", {"ref": "e1"}, 'button "Удаление"', "https://x.test/").risk == "medium"


def test_high_risk_words_are_caught_even_in_long_labels():
    p = policy()
    label = 'button "Оформить заказ и оплатить картой ···· 4417 прямо сейчас, доставка 30 минут"'
    assert p.assess("browser_click", {"ref": "e1"}, label, "https://shop.test/cart").risk == "high"


def test_text_the_agent_is_about_to_type_is_classified_too():
    p = policy()
    verdict = p.assess(
        "browser_type", {"ref": "e1", "text": "удалить аккаунт", "submit": True}, 'textbox "Команда"', "u"
    )
    assert verdict.risk == "high"


def test_wording_that_means_different_things_on_different_sites_is_escalated():
    """"В корзину" is "move to trash" in a mail client and "add to cart" in a
    shop. A regex cannot tell those apart, so with no judge to ask, the agent
    asks the user rather than guessing."""
    p = policy()  # judge disabled
    assert p.assess(
        "browser_click", {"ref": "e1"}, 'button "В корзину"', "https://shop.test/menu"
    ).risk == "medium"
    # the unambiguous phrasing never needed a judge
    assert p.assess(
        "browser_click", {"ref": "e1"}, 'button "Переместить в корзину"', "https://mail.test/"
    ).risk == "medium"


def test_the_judge_is_what_disambiguates_when_it_is_available():
    seen = {}

    class Judge:
        def text(self, *, system, prompt, model=None, max_tokens=1500, effort=None):
            seen["prompt"] = prompt
            return '{"risk": "low", "reason": "adding an item to a cart"}'

    p = SafetyPolicy(cfg=SafetyConfig(mode="ask", use_llm_judge=True), llm=Judge())
    verdict = p.assess("browser_click", {"ref": "e1"}, 'button "В корзину"', "https://shop.test/menu")
    assert verdict.risk == "low" and verdict.source == "judge"
    assert "shop.test" in seen["prompt"], "the judge must see the page to disambiguate"
