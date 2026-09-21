from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "demo" / "site"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def demo_url() -> str:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "-d", str(SITE)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    proc.terminate()


@pytest.fixture
def session(tmp_path):
    from browser_agent.browser.session import BrowserSession

    s = BrowserSession(user_data_dir=tmp_path / "profile", headless=True)
    s.start()
    yield s
    s.close()


@pytest.fixture
def mail(session, demo_url):
    session.goto(f"{demo_url}/mail.html")
    session.settle(quiet_ms=700)
    return session
