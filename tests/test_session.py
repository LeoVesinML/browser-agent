"""Persistent sessions are a stated requirement: the user logs in by hand, the
agent carries on. That rests entirely on the browser profile surviving a
restart, so it gets a test rather than a promise."""

from __future__ import annotations

from browser_agent.browser.session import BrowserSession


def _start(tmp_path, url):
    session = BrowserSession(user_data_dir=tmp_path / "profile", headless=True)
    session.start(url)
    session.settle(quiet_ms=300)
    return session


def test_a_login_survives_a_restart_of_the_agent(tmp_path, demo_url):
    """Stand in for a manual login: whatever the site stored about the user has
    to still be there after the agent process goes away and comes back."""
    session = _start(tmp_path, f"{demo_url}/mail.html")
    session.active_page().evaluate(
        "() => { document.cookie = 'session_id=abc123; path=/; max-age=86400';"
        "  localStorage.setItem('signed_in_as', 'leo@example.test'); }"
    )
    session.close()

    again = _start(tmp_path, f"{demo_url}/mail.html")
    try:
        cookie = again.active_page().evaluate("() => document.cookie")
        stored = again.active_page().evaluate("() => localStorage.getItem('signed_in_as')")
        assert "session_id=abc123" in cookie, "cookies must outlive the process"
        assert stored == "leo@example.test", "site storage must outlive the process"
    finally:
        again.close()


def test_a_fresh_profile_starts_signed_out(tmp_path, demo_url):
    """The flip side: a different profile must not inherit the first one's state,
    or 'persistent' would just mean 'leaky'."""
    session = _start(tmp_path / "a", f"{demo_url}/mail.html")
    session.active_page().evaluate("() => localStorage.setItem('signed_in_as', 'leo')")
    session.close()

    other = _start(tmp_path / "b", f"{demo_url}/mail.html")
    try:
        assert other.active_page().evaluate("() => localStorage.getItem('signed_in_as')") is None
    finally:
        other.close()


def test_the_browser_is_visible_by_default(tmp_path):
    """The brief asks for a visible browser; headless must stay opt-in."""
    assert BrowserSession(user_data_dir=tmp_path).headless is False


def test_a_tab_the_site_opens_becomes_the_active_one(session, demo_url):
    """A human follows the tab the site just opened; so does the agent, and its
    next snapshot has to describe that tab rather than the one left behind.

    Driven through the context event the browser itself fires for a popup, which
    is the path a target=_blank link takes.
    """
    from browser_agent.browser import snapshot as S

    session.goto(f"{demo_url}/mail.html")
    before = len(session.pages)

    opened = session.context.new_page()  # fires context "page", as a popup would
    opened.goto(f"{demo_url}/shop.html", wait_until="domcontentloaded")
    session.settle(quiet_ms=500)

    assert len(session.pages) == before + 1
    assert session.page is opened, "the new tab must take focus"
    assert "Вкусно и Быстро" in S.capture(session).render()

    tabs = session.tab_list()
    assert sum(1 for t in tabs if t["active"]) == 1


def test_a_site_that_redirects_on_arrival_is_not_a_failure(session, demo_url):
    """Real sites bounce you to a city subdomain or a locale the moment you land.
    Playwright calls that an interrupted navigation; treating it as an error made
    every such site unreachable."""
    session.goto(f"{demo_url}/redirect.html")
    session.settle(quiet_ms=400)
    assert session.active_page().url.endswith("/index.html")
    assert "Демо-стенд" in session.active_page().title()
