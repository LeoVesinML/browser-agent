"""Acting on the page through refs, never through hand-written selectors.

Every action follows the same contract: resolve the ref against the live DOM,
scroll it into view, do the thing a human would do, let the page settle, then
report *what changed* rather than what the page now looks like in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from playwright.sync_api import Error as PlaywrightError, TimeoutError as PlaywrightTimeout

from . import snapshot as snap_mod
from .session import BrowserSession
from .snapshot import Snapshot, StaleRefError

SENSITIVE_INPUT_TYPES = {"password"}


class ActionError(RuntimeError):
    """A failure the agent can recover from; the message tells it how."""


@dataclass
class ActionResult:
    ok: bool
    message: str
    snapshot: Snapshot | None = None


def describe(session: BrowserSession, element: Any) -> str:
    try:
        return element.evaluate(
            "el => { const t = (el.innerText || el.value || el.getAttribute('aria-label') "
            "|| el.getAttribute('placeholder') || '').replace(/\\s+/g,' ').trim(); "
            "return el.tagName.toLowerCase() + (t ? ' \"' + t.slice(0,80) + '\"' : ''); }"
        )
    except PlaywrightError:
        return "<element>"


def _prepare(session: BrowserSession, snap: Snapshot, ref: str):
    element = snap_mod.resolve(session, snap, ref)
    try:
        element.scroll_into_view_if_needed(timeout=3_000)
    except (PlaywrightTimeout, PlaywrightError):
        pass  # fixed-position or zero-size elements are still clickable
    return element


def click(
    session: BrowserSession,
    snap: Snapshot,
    ref: str,
    button: str = "left",
    modifiers: list[str] | None = None,
) -> str:
    element = _prepare(session, snap, ref)
    label = describe(session, element)
    try:
        element.click(button=button, modifiers=modifiers or [], timeout=6_000)
    except PlaywrightTimeout as exc:
        # Something is covering the target: retry through the DOM, which bypasses
        # hit-testing, and say so plainly if that fails too.
        try:
            element.evaluate("el => el.click()")
        except PlaywrightError:
            raise ActionError(
                f"could not click {label}: {str(exc).splitlines()[0]}. It may be covered by "
                f"an overlay or a cookie banner - handle that first, or try a different element."
            ) from exc
    except PlaywrightError as exc:
        raise ActionError(f"could not click {label}: {str(exc).splitlines()[0]}") from exc
    session.settle()
    return f"clicked {label}"


def type_text(
    session: BrowserSession,
    snap: Snapshot,
    ref: str,
    text: str,
    clear: bool = True,
    submit: bool = False,
) -> str:
    element = _prepare(session, snap, ref)
    label = describe(session, element)
    try:
        input_type = (element.get_attribute("type") or "").lower()
    except PlaywrightError:
        input_type = ""
    if input_type in SENSITIVE_INPUT_TYPES:
        raise ActionError(
            "refusing to type into a password field. Credentials are the user's to enter: "
            "ask the user to log in manually in the open browser window, then continue."
        )
    try:
        if clear:
            element.fill("", timeout=5_000)
        element.type(text, delay=18, timeout=15_000)
        if submit:
            element.press("Enter")
    except PlaywrightError as exc:
        raise ActionError(f"could not type into {label}: {str(exc).splitlines()[0]}") from exc
    session.settle()
    return f"typed {text[:60]!r} into {label}" + (" and pressed Enter" if submit else "")


def select_option(session: BrowserSession, snap: Snapshot, ref: str, values: list[str]) -> str:
    element = _prepare(session, snap, ref)
    label = describe(session, element)
    try:
        chosen = element.select_option(label=values, timeout=5_000)
    except PlaywrightError:
        try:
            chosen = element.select_option(value=values, timeout=5_000)
        except PlaywrightError as exc:
            raise ActionError(
                f"could not select {values!r} in {label}: {str(exc).splitlines()[0]}. "
                f"Check the exact option labels in the snapshot."
            ) from exc
    session.settle()
    return f"selected {chosen} in {label}"


def set_checked(session: BrowserSession, snap: Snapshot, ref: str, checked: bool) -> str:
    element = _prepare(session, snap, ref)
    label = describe(session, element)
    try:
        element.set_checked(checked, timeout=5_000)
    except PlaywrightError as exc:
        raise ActionError(f"could not toggle {label}: {str(exc).splitlines()[0]}") from exc
    session.settle()
    return f"set {label} to {'checked' if checked else 'unchecked'}"


def press_key(session: BrowserSession, key: str) -> str:
    page = session.active_page()
    try:
        page.keyboard.press(key)
    except PlaywrightError as exc:
        raise ActionError(f"could not press {key!r}: {exc}") from exc
    session.settle()
    return f"pressed {key}"


def scroll(
    session: BrowserSession,
    snap: Snapshot,
    direction: str = "down",
    amount: int = 1,
    ref: str | None = None,
) -> str:
    page = session.active_page()
    if ref:
        element = snap_mod.resolve(session, snap, ref)
        try:
            element.evaluate(
                "(el, d) => el.scrollBy({top: d * el.clientHeight * 0.85, behavior: 'instant'})",
                1 if direction == "down" else -1,
            )
        except PlaywrightError as exc:
            raise ActionError(f"could not scroll {ref}: {exc}") from exc
        session.settle(quiet_ms=200)
        return f"scrolled {direction} inside {ref}"

    if direction in ("top", "bottom"):
        page.evaluate(
            "d => window.scrollTo({top: d === 'top' ? 0 : document.body.scrollHeight, behavior: 'instant'})",
            direction,
        )
    else:
        sign = 1 if direction == "down" else -1
        page.evaluate(
            "n => window.scrollBy({top: n * window.innerHeight * 0.85, behavior: 'instant'})",
            sign * max(1, amount),
        )
    session.settle(quiet_ms=200)
    return f"scrolled {direction}"


def navigate(session: BrowserSession, action: str, url: str | None = None) -> str:
    page = session.active_page()
    try:
        if action == "goto":
            if not url:
                raise ActionError("navigate(goto) requires a url")
            session.goto(url)
            return f"navigated to {page.url}"
        if action == "back":
            page.go_back(wait_until="domcontentloaded")
        elif action == "forward":
            page.go_forward(wait_until="domcontentloaded")
        elif action == "reload":
            page.reload(wait_until="domcontentloaded")
        else:
            raise ActionError(f"unknown navigation action {action!r}")
    except PlaywrightTimeout as exc:
        raise ActionError(
            f"navigation timed out: {str(exc).splitlines()[0]}. The page may still be usable - "
            f"take a snapshot to check before retrying."
        ) from exc
    except PlaywrightError as exc:
        raise ActionError(f"navigation failed: {str(exc).splitlines()[0]}") from exc
    session.settle()
    return f"{action}: now at {session.active_page().url}"


def find(session: BrowserSession, query: str, limit: int = 15) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for frame in session.frames():
        try:
            res = frame.evaluate("([q, n]) => window.__agent.find(q, n)", [query, limit])
        except PlaywrightError:
            continue
        for match in res.get("matches", []):
            out.append(match)
        if len(out) >= limit:
            break
    return out[:limit]


def read_text(
    session: BrowserSession,
    ref: str | None = None,
    start: int = 0,
    max_chars: int = 4_000,
) -> dict[str, Any]:
    frame = session.active_page().main_frame
    if ref:
        for f in session.frames():
            try:
                if f.evaluate("r => !!window.__agent.get(r)", ref):
                    frame = f
                    break
            except PlaywrightError:
                continue
    try:
        return frame.evaluate(
            "([r, s, m]) => window.__agent.readText(r, s, m)", [ref, start, max_chars]
        )
    except PlaywrightError as exc:
        raise ActionError(f"could not read page text: {exc}") from exc


def wait_for(session: BrowserSession, condition: str, value: str | None, timeout_ms: int) -> str:
    page = session.active_page()
    try:
        if condition == "text":
            if not value:
                raise ActionError("wait(text) requires a value")
            page.wait_for_function(
                "t => document.body && document.body.innerText.includes(t)",
                arg=value,
                timeout=timeout_ms,
            )
            return f"text {value!r} appeared"
        if condition == "text_gone":
            page.wait_for_function(
                "t => !document.body || !document.body.innerText.includes(t)",
                arg=value,
                timeout=timeout_ms,
            )
            return f"text {value!r} disappeared"
        if condition == "idle":
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
            return "network went idle"
        if condition == "seconds":
            page.wait_for_timeout(min(timeout_ms, 10_000))
            return "waited"
        raise ActionError(f"unknown wait condition {condition!r}")
    except PlaywrightTimeout as exc:
        raise ActionError(
            f"timed out waiting for {condition} {value!r} after {timeout_ms}ms. "
            f"Take a snapshot to see the current state - the page may have changed differently "
            f"than expected."
        ) from exc


__all__ = [
    "ActionError",
    "ActionResult",
    "StaleRefError",
    "click",
    "find",
    "navigate",
    "press_key",
    "read_text",
    "scroll",
    "select_option",
    "set_checked",
    "type_text",
    "wait_for",
]
