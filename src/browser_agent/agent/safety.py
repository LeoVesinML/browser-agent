"""Security layer: a gate in front of anything the user cannot undo.

The agent is autonomous, not unsupervised. Every state-changing action is
classified before it runs, and anything that looks irreversible stops the loop
and asks the human sitting in front of the browser.

Two-stage classification, cheap first:

1. a lexical pass over the element's own accessible name and the page URL, in
   both Russian and English, which catches the obvious cases for free;
2. for anything it is unsure about, a small-model judge that sees only the
   action, the element label and the URL.

Note what is *not* here: no site-specific rules, no "this is the delete button
on service X". The classifier reads whatever the page itself says the control
does, exactly like the agent does.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import SafetyConfig
from ..llm import LLM

# Actions that can change the world. Everything else (looking, scrolling,
# reading) is never gated.
MUTATING_TOOLS = {
    "browser_click", "browser_type", "browser_select", "browser_check",
    "browser_press_key", "browser_navigate",
}

HIGH_RISK = [
    # payment & money
    r"оплат", r"к оплате", r"купить", r"оформить заказ", r"подтвердить заказ",
    r"\bpay\b", r"checkout", r"place order", r"buy now", r"purchase", r"subscribe",
    r"перевести", r"перевод денег", r"вывести средства", r"withdraw", r"transfer",
    # destruction
    r"удалить навсегда", r"удалить аккаунт", r"delete account", r"delete forever",
    r"очистить корзину", r"empty trash", r"удалить безвозвратно",
]

MEDIUM_RISK = [
    r"удалить", r"удаление", r"в корзину(?!\s*·)", r"переместить в корзину", r"это спам",
    r"пометить как спам", r"\bdelete\b", r"\bremove\b", r"move to trash", r"\bspam\b",
    r"отправить", r"\bsend\b", r"\bsubmit\b", r"откликнуться", r"\bapply\b",
    r"опубликовать", r"\bpublish\b", r"\bpost\b", r"отписаться", r"unsubscribe",
    r"выйти из аккаунта", r"\blog ?out\b", r"\bsign ?out\b", r"отменить заказ",
    r"сохранить изменения", r"save changes", r"подтвердить", r"\bconfirm\b",
    r"согласен", r"\baccept\b", r"принять",
]

PAYMENT_CONTEXT = [r"checkout", r"payment", r"оплат", r"cart", r"корзин", r"заказ", r"order"]

# Data we will not hand to a page on the agent's initiative, no matter what the
# model decides. The agent must ask the user to enter these themselves.
FORBIDDEN_INPUT = [
    (r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "what looks like a bank card number"),
    (r"\bcvv\b|\bcvc\b", "a card security code"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "what looks like a government id number"),
]

JUDGE_SYSTEM = """You are the safety gate of a browser agent. Given one action, decide how
hard it would be for the user to undo it.

Reply with JSON only: {"risk": "none|low|medium|high", "reason": "<=15 words"}

- none/low: navigating, searching, opening, filtering, expanding, adding to a cart.
- medium: deleting or archiving one item, sending a message or application,
  submitting a form, changing a setting, accepting terms.
- high: paying, ordering, transferring money, deleting an account or a bulk of
  data, anything that spends money or destroys data irreversibly.

Judge only the action described. Do not invent context."""


@dataclass
class Verdict:
    risk: str  # none | low | medium | high
    reason: str
    source: str  # rules | judge | policy

    @property
    def rank(self) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3}[self.risk]


@dataclass
class SafetyPolicy:
    cfg: SafetyConfig
    llm: LLM | None = None
    confirm: Callable[[str, dict[str, Any], Verdict], bool] | None = None
    always_allow: set[str] = field(default_factory=set)
    _judge_cache: dict[str, Verdict] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------- classifying

    # A control's own name is short and imperative ("Удалить", "Оплатить заказ").
    # Anything longer is descriptive prose - a card, a row, a link that explains a
    # section - and matching command words inside it produces false alarms, e.g.
    # a link reading "Входящие, поиск, чтение, удаление с подтверждением".
    ACTION_LABEL_CHARS = 48

    @staticmethod
    def _split(label: str) -> tuple[str, str]:
        """`role "name"` -> (role, name). Falls back to ('', label)."""
        match = re.match(r'^(\w+)\s+"(.*)"$', label.strip(), re.S)
        return (match.group(1), match.group(2)) if match else ("", label)

    def assess(self, tool: str, args: dict[str, Any], label: str, url: str) -> Verdict:
        if tool not in MUTATING_TOOLS:
            return Verdict("none", "read-only action", "rules")

        role, name = self._split(label)
        page = url.lower()
        typed = json.dumps(args.get("text", ""), ensure_ascii=False).lower()
        # Lexical rules only look at what the control calls itself, plus text the
        # agent is about to type. Never at the arguments' bookkeeping fields.
        haystack = f"{name} {typed}".lower()
        command_like = len(name) <= self.ACTION_LABEL_CHARS

        for pattern in HIGH_RISK:
            if re.search(pattern, haystack):
                return Verdict("high", f"matches irreversible action pattern {pattern!r}", "rules")

        if command_like:
            for pattern in MEDIUM_RISK:
                if re.search(pattern, haystack):
                    risk = "high" if any(re.search(p, page) for p in PAYMENT_CONTEXT) and (
                        "подтверд" in haystack or "confirm" in haystack or "оформ" in haystack
                    ) else "medium"
                    return Verdict(risk, f"matches state-changing pattern {pattern!r}", "rules")

        if tool == "browser_navigate":
            return Verdict("low", "navigation", "rules")
        if tool == "browser_type" and not args.get("submit"):
            return Verdict("low", "typing into a field", "rules")
        if tool == "browser_click" and role == "link" and not args.get("modifiers"):
            # Following a link is navigation. Sites that destroy data behind a
            # bare <a> exist, which is what the judge below is for.
            return Verdict("low", "following a link", "rules")

        if self.cfg.use_llm_judge and self.llm is not None:
            return self._judge(tool, args, label, url)
        return Verdict("low", "no risky pattern matched", "rules")

    def _judge(self, tool: str, args: dict[str, Any], label: str, url: str) -> Verdict:
        key = f"{tool}|{label}|{url.split('?')[0]}"
        if key in self._judge_cache:
            return self._judge_cache[key]
        prompt = (
            f"action: {tool}\n"
            f"element: {label or '(unnamed)'}\n"
            f"arguments: {json.dumps(args, ensure_ascii=False)[:300]}\n"
            f"page: {url[:200]}"
        )
        try:
            raw = self.llm.text(system=JUDGE_SYSTEM, prompt=prompt, max_tokens=200)
            data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            verdict = Verdict(
                str(data.get("risk", "low")).lower(),
                str(data.get("reason", ""))[:120],
                "judge",
            )
            if verdict.risk not in {"none", "low", "medium", "high"}:
                verdict = Verdict("low", "judge returned an unknown level", "judge")
        except Exception as exc:  # noqa: BLE001 - the judge must never break a run
            verdict = Verdict("low", f"judge unavailable ({type(exc).__name__})", "judge")
        self._judge_cache[key] = verdict
        return verdict

    # ---------------------------------------------------------------- gating

    def hard_block(self, tool: str, args: dict[str, Any]) -> str | None:
        """Things the agent may never do, regardless of any confirmation."""
        if tool != "browser_type":
            return None
        text = str(args.get("text", ""))
        for pattern, what in FORBIDDEN_INPUT:
            if re.search(pattern, text, re.I):
                return (
                    f"Blocked: this would type {what} into a web page. Payment and identity "
                    f"details are the user's to enter. Ask the user to fill this field "
                    f"themselves in the open browser window, then continue."
                )
        return None

    def gate(self, tool: str, args: dict[str, Any], label: str, url: str) -> tuple[bool, Verdict]:
        """Returns (allowed, verdict). May block the loop on a terminal prompt."""
        blocked = self.hard_block(tool, args)
        if blocked:
            return False, Verdict("high", blocked, "policy")

        verdict = self.assess(tool, args, label, url)
        if self.cfg.mode == "yolo":
            return True, verdict

        threshold = 1 if self.cfg.mode == "strict" else 2  # strict gates 'low' too
        if verdict.rank < threshold:
            return True, verdict
        if f"{tool}|{label}" in self.always_allow:
            return True, verdict
        if self.confirm is None:
            return True, verdict
        return self.confirm(tool, args, verdict), verdict
