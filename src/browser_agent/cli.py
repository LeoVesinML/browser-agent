"""Command line entry point: a terminal next to a browser."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from rich.prompt import Prompt

from .agent.console import AgentConsole
from .agent.loop import BrowserAgent
from .agent.trace import Trace
from .browser.session import BrowserSession
from .config import PROJECT_ROOT, AgentConfig

DEMO_PORT = 8765


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="browser-agent",
        description="An autonomous agent that drives a real browser to complete open-ended tasks.",
    )
    p.add_argument("task", nargs="*", help="Task text. Omit it to get an interactive prompt.")
    p.add_argument("--url", help="Page to open before the agent starts.")
    p.add_argument("--demo", action="store_true", help="Serve the bundled demo site and start there.")
    p.add_argument("--profile-dir", help="Chromium profile directory (persists logins).")
    p.add_argument("--model", help="Main model id (default claude-opus-5).")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-steps", type=int)
    p.add_argument("--context-budget", type=int, help="Token budget the agent manages against.")
    p.add_argument(
        "--safety",
        choices=["ask", "strict", "yolo"],
        help="ask (default): confirm risky actions; strict: confirm anything that mutates; "
        "yolo: never ask.",
    )
    p.add_argument("--no-judge", action="store_true", help="Disable the LLM risk judge (rules only).")
    p.add_argument("--headless", action="store_true", help="Hide the browser (not recommended).")
    p.add_argument("--slow-mo", type=int, help="Milliseconds to slow each browser action, for demos.")
    p.add_argument("--record-video", action="store_true", help="Record the browser to runs/<ts>/video.")
    return p


def apply_args(cfg: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    if args.model:
        cfg.model.main = args.model
    if args.effort:
        cfg.model.effort = args.effort
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.context_budget:
        cfg.context.budget_tokens = args.context_budget
    if args.safety:
        cfg.safety.mode = args.safety
    if args.no_judge:
        cfg.safety.use_llm_judge = False
    if args.headless:
        cfg.browser.headless = True
    if args.slow_mo:
        cfg.browser.slow_mo_ms = args.slow_mo
    if args.profile_dir:
        cfg.browser.profile_dir = Path(args.profile_dir)
    if args.record_video:
        cfg.browser.record_video = True
    if args.url:
        cfg.browser.start_url = args.url
    return cfg


def start_demo_server(console: AgentConsole) -> subprocess.Popen | None:
    site = PROJECT_ROOT / "demo" / "site"
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(DEMO_PORT), "-d", str(site)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.6)
    console.note(f"demo site served at http://localhost:{DEMO_PORT}")
    return proc


def check_credentials(console: AgentConsole) -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    console.error(
        "No API credentials found. Set ANTHROPIC_API_KEY, or point ANTHROPIC_BASE_URL + "
        "ANTHROPIC_AUTH_TOKEN at an Anthropic-compatible endpoint (see .env.example)."
    )
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = AgentConsole()
    cfg = apply_args(AgentConfig(), args)

    if not check_credentials(console):
        return 2

    demo_proc = None
    if args.demo:
        demo_proc = start_demo_server(console)
        if not args.url:
            cfg.browser.start_url = f"http://localhost:{DEMO_PORT}/"

    trace = Trace(cfg.runs_dir)
    session = BrowserSession(
        user_data_dir=cfg.browser.profile_dir,
        headless=cfg.browser.headless,
        slow_mo_ms=cfg.browser.slow_mo_ms,
        video_dir=(trace.dir / "video") if cfg.browser.record_video else None,
    )

    console.rule("browser agent")
    try:
        session.start(cfg.browser.start_url)
    except Exception as exc:  # noqa: BLE001
        console.error(f"could not start the browser: {exc}")
        return 3

    # Marks the point the video recording starts, so a composed demo video can
    # line the terminal log up with the browser footage.
    trace.write("browser_started", url=session.active_page().url)
    console.note(
        f"profile {cfg.browser.profile_dir} · safety={cfg.safety.mode} · trace {trace.dir}"
    )
    console.note(
        "The browser window is yours too: log in manually if a site needs it, then give the "
        "agent its task - the session persists."
    )

    try:
        tasks = [" ".join(args.task)] if args.task else []
        while True:
            if not tasks:
                try:
                    text = Prompt.ask("\n[bold cyan]task[/bold cyan]")
                except (EOFError, KeyboardInterrupt):
                    break
                if not text.strip() or text.strip() in {"exit", "quit", "q"}:
                    break
                tasks = [text]

            task = tasks.pop(0)
            agent = BrowserAgent(cfg, session, console, trace)
            console.task(task, cfg.model.main, session.active_page().url)
            try:
                result = agent.run(task)
            except KeyboardInterrupt:
                console.warn("interrupted by the user")
                continue
            except Exception as exc:  # noqa: BLE001
                console.error(f"run failed: {type(exc).__name__}: {exc}")
                trace.write("run_error", error=str(exc))
                continue

            console.report(result.report, result.status)
            console.usage(result.usage, result.steps, agent.convo.compactions, agent.convo.pruned_count)
            trace.write("run_done", status=result.status, report=result.report,
                        steps=result.steps, cost_usd=round(result.usage.cost_usd, 4))
            if args.task:
                break
    finally:
        session.close()
        if demo_proc:
            demo_proc.terminate()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
