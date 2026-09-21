"""Browser lifecycle: a visible, persistent Chromium the agent shares with the user.

The session owns everything stateful about the browser (profile, tabs, dialogs,
downloads) and exposes a small surface the tool layer can build on. It knows
nothing about any particular website.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Dialog,
    Error as PlaywrightError,
    Frame,
    Page,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)

log = logging.getLogger(__name__)

_INIT_SCRIPT = Path(__file__).parent / "js" / "extract.js"


@dataclass
class DialogEvent:
    kind: str
    message: str
    handled_as: str


@dataclass
class BrowserSession:
    """A running browser the agent drives."""

    user_data_dir: Path
    headless: bool = False
    slow_mo_ms: int = 0
    video_dir: Path | None = None
    dialog_policy: str = "accept"  # accept | dismiss
    default_timeout_ms: int = 12_000

    _pw: Any = field(default=None, init=False, repr=False)
    _context: BrowserContext | None = field(default=None, init=False, repr=False)
    _browser: Browser | None = field(default=None, init=False, repr=False)
    page: Page | None = field(default=None, init=False, repr=False)
    pending_dialogs: list[DialogEvent] = field(default_factory=list, init=False)
    last_console_errors: list[str] = field(default_factory=list, init=False)

    # ---------------------------------------------------------------- lifecycle

    def start(self, start_url: str | None = None) -> None:
        self._pw = sync_playwright().start()
        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        launch_kwargs: dict[str, Any] = {
            "user_data_dir": str(self.user_data_dir),
            "headless": self.headless,
            "slow_mo": self.slow_mo_ms,
            "no_viewport": True,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ],
            "ignore_default_args": ["--enable-automation"],
        }
        if self.video_dir:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            launch_kwargs["record_video_dir"] = str(self.video_dir)
            launch_kwargs["record_video_size"] = {"width": 1440, "height": 900}
            # A recorded context needs a fixed viewport.
            launch_kwargs["no_viewport"] = False
            launch_kwargs["viewport"] = {"width": 1440, "height": 900}

        self._context = self._pw.chromium.launch_persistent_context(**launch_kwargs)
        self._context.set_default_timeout(self.default_timeout_ms)
        # Injected before any page script runs, in every frame, on every navigation.
        self._context.add_init_script(path=str(_INIT_SCRIPT))
        self._context.on("page", self._on_new_page)

        pages = self._context.pages
        self.page = pages[0] if pages else self._context.new_page()
        self._wire_page(self.page)
        if start_url:
            self.goto(start_url)

    def close(self) -> None:
        for closer in (self._context, self._browser):
            try:
                if closer:
                    closer.close()
            except Exception:  # noqa: BLE001 - best effort teardown
                pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------- events

    def _on_new_page(self, page: Page) -> None:
        self._wire_page(page)
        # Popups and target=_blank links become the active tab, like for a human.
        self.page = page
        try:
            page.wait_for_load_state("domcontentloaded", timeout=5_000)
        except PlaywrightError:
            pass

    def _wire_page(self, page: Page) -> None:
        page.on("dialog", self._on_dialog)
        page.on("console", self._on_console)
        page.on("close", lambda _: self._on_page_closed())

    def _on_dialog(self, dialog: Dialog) -> None:
        action = self.dialog_policy
        try:
            if action == "accept":
                dialog.accept()
            else:
                dialog.dismiss()
        except PlaywrightError:
            action = "already-handled"
        self.pending_dialogs.append(
            DialogEvent(kind=dialog.type, message=dialog.message[:300], handled_as=action)
        )

    def _on_console(self, msg: Any) -> None:
        if msg.type == "error":
            self.last_console_errors.append(msg.text[:200])
            del self.last_console_errors[:-10]

    def _on_page_closed(self) -> None:
        live = [p for p in self.pages if not p.is_closed()]
        if live and (self.page is None or self.page.is_closed()):
            self.page = live[-1]

    # -------------------------------------------------------------------- tabs

    @property
    def context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("browser session not started")
        return self._context

    @property
    def pages(self) -> list[Page]:
        return [p for p in self.context.pages if not p.is_closed()]

    def active_page(self) -> Page:
        if self.page is None or self.page.is_closed():
            self.page = self.pages[-1] if self.pages else self.context.new_page()
            self._wire_page(self.page)
        return self.page

    def select_tab(self, index: int) -> Page:
        pages = self.pages
        if not 0 <= index < len(pages):
            raise IndexError(f"tab {index} does not exist (open tabs: {len(pages)})")
        self.page = pages[index]
        self.page.bring_to_front()
        return self.page

    def new_tab(self, url: str | None = None) -> Page:
        page = self.context.new_page()
        self._wire_page(page)
        self.page = page
        if url:
            self.goto(url)
        return page

    def close_tab(self, index: int) -> None:
        pages = self.pages
        if not 0 <= index < len(pages):
            raise IndexError(f"tab {index} does not exist")
        pages[index].close()
        self._on_page_closed()

    def tab_list(self) -> list[dict[str, Any]]:
        active = self.active_page()
        out = []
        for i, p in enumerate(self.pages):
            try:
                out.append({"index": i, "title": p.title()[:80], "url": p.url, "active": p is active})
            except PlaywrightError:
                out.append({"index": i, "title": "<unavailable>", "url": "", "active": p is active})
        return out

    # --------------------------------------------------------------- navigation

    def goto(self, url: str, timeout_ms: int | None = None) -> None:
        page = self.active_page()
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:", url):
            url = "https://" + url
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms or self.default_timeout_ms)
        self.settle()

    def settle(self, quiet_ms: int = 350, timeout_ms: int = 4_000) -> None:
        """Best-effort wait for the page to stop changing.

        Never raises: a page that keeps polling (chat widgets, analytics) must not
        stall the agent, it just means we snapshot a slightly livelier page.
        """
        page = self.active_page()
        try:
            page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
        except (PlaywrightTimeout, PlaywrightError):
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except (PlaywrightTimeout, PlaywrightError):
            pass
        time.sleep(quiet_ms / 1000)

    # ------------------------------------------------------------------- frames

    def frames(self) -> list[Frame]:
        page = self.active_page()
        out = []
        for f in page.frames:
            try:
                if f.is_detached():
                    continue
                out.append(f)
            except PlaywrightError:
                continue
        return out

    def drain_dialogs(self) -> list[DialogEvent]:
        events, self.pending_dialogs = self.pending_dialogs, []
        return events
