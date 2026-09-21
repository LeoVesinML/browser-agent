"""The perception layer is where most agent failures come from, so it is the
part that gets the most tests."""

from __future__ import annotations

from browser_agent.browser import actions as A
from browser_agent.browser import snapshot as S


def test_snapshot_is_tiny_compared_to_the_html(mail):
    html_size = len(mail.active_page().content())
    snap = S.capture(mail)
    rendered = snap.render(20_000)
    assert snap.interactive_count > 10
    assert len(rendered) < html_size / 3
    assert "Мяу.Почта" in rendered


def test_delegated_click_handlers_still_get_handles(mail):
    """Rows here are plain <div>s with a delegated listener - only the pointer
    cursor betrays them. A snapshot that missed them would make the task
    impossible."""
    snap = S.capture(mail)
    rows = [n for n in snap.nodes if n.get("role") == "listitem" and n.get("ref")]
    assert len(rows) >= 10


def test_refs_are_stable_across_snapshots(mail):
    first = S.capture(mail)
    second = S.capture(mail)
    by_name = {n["name"]: n.get("ref") for n in first.nodes if n.get("ref")}
    for node in second.nodes:
        if node.get("ref") and node["name"] in by_name:
            assert node["ref"] == by_name[node["name"]], node["name"]


def test_diff_of_a_small_change_is_small(mail):
    before = S.capture(mail)
    ref = next(n["ref"] for n in before.nodes if n.get("name") == "Принять все")
    A.click(mail, before, ref)
    after = S.capture(mail)
    delta = S.diff(before, after)
    assert len(delta) < len(after.render(20_000)) / 3
    assert "Принять все" in delta


def test_modal_dominates_the_snapshot(mail):
    snap = S.capture(mail)
    row = next(n["ref"] for n in snap.nodes if "Крипто" in (n.get("name") or ""))
    A.click(mail, snap, row)
    snap = S.capture(mail)
    spam = next(n["ref"] for n in snap.nodes if n.get("name") == "Это спам")
    A.click(mail, snap, spam)
    snap = S.capture(mail)
    assert snap.modal
    assert "MODAL DIALOG IS OPEN" in snap.render()
    names = [n.get("name") for n in snap.nodes]
    assert "Отмена" in names and "Поиск по письмам" not in names


def test_stale_ref_is_reported_not_silently_rebound(mail):
    snap = S.capture(mail)
    row = next(n["ref"] for n in snap.nodes if "Крипто" in (n.get("name") or ""))
    A.click(mail, snap, row)  # the list is replaced by the reader view
    try:
        A.click(mail, snap, row)
    except (S.StaleRefError, A.ActionError) as exc:
        assert "snapshot" in str(exc).lower()
    else:  # pragma: no cover - the point of the test
        raise AssertionError("a stale ref must not resolve silently")


def test_find_returns_usable_handles(mail):
    matches = A.find(mail, "Спам", 10)
    assert matches
    snap = S.capture(mail)
    A.click(mail, snap, matches[0]["ref"])
    assert "Писем нет" in mail.active_page().inner_text("body")


def test_read_text_is_chunked(mail):
    snap = S.capture(mail)
    row = next(n["ref"] for n in snap.nodes if "Крипто" in (n.get("name") or ""))
    A.click(mail, snap, row)
    head = A.read_text(mail, max_chars=40)
    assert head["returned"] == 40 and head["total"] > 40
    tail = A.read_text(mail, start=40, max_chars=400)
    assert tail["start"] == 40


def test_typing_into_a_password_field_is_refused(session, demo_url):
    session.goto("data:text/html,<input type=password id=p>")
    session.settle(quiet_ms=200)
    snap = S.capture(session)
    ref = next(n["ref"] for n in snap.nodes if n.get("ref"))
    try:
        A.type_text(session, snap, ref, "hunter2")
    except A.ActionError as exc:
        assert "password" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("passwords must never be typed by the agent")


def test_back_in_a_fresh_tab_says_so_instead_of_doing_nothing(session, demo_url):
    """A tab opened by a link has no history. Silently doing nothing is what
    makes an agent retry the same dead action; an explicit error is what makes
    it try the tabs tool instead."""
    session.goto(f"{demo_url}/mail.html")
    session.new_tab(f"{demo_url}/shop.html")
    try:
        A.navigate(session, "back")
    except A.ActionError as exc:
        assert "browser_tabs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a no-op back must be reported")
