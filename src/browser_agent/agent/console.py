"""Terminal UI.

The point of the run is watching the agent think: every tool call and its
arguments are printed as they happen, next to the browser window doing it.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

RISK_STYLE = {"none": "green", "low": "green", "medium": "yellow", "high": "bold red"}


class AgentConsole:
    def __init__(self, quiet: bool = False) -> None:
        self.console = Console()
        self.quiet = quiet

    # ------------------------------------------------------------------ output

    def rule(self, text: str) -> None:
        self.console.rule(f"[bold]{text}[/]")

    def task(self, task: str, model: str, url: str) -> None:
        self.console.print(
            Panel(
                Text(task, style="bold white"),
                title="task",
                subtitle=f"{model} · {url}",
                border_style="cyan",
            )
        )

    def step(self, n: int, tokens: int, budget: int) -> None:
        pct = int(100 * tokens / max(budget, 1))
        colour = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
        self.console.print(
            f"[dim]── step {n} · context [{colour}]{tokens:,}[/{colour}]/{budget:,} tok ──[/dim]"
        )

    def thinking(self, text: str) -> None:
        if text.strip():
            self.console.print(Text(text.strip(), style="italic grey62"))

    def assistant(self, text: str) -> None:
        if text.strip():
            self.console.print(Markdown(text.strip()))

    def tool_call(self, name: str, args: dict[str, Any], prefix: str = "") -> None:
        rendered = ", ".join(
            f"[cyan]{k}[/cyan]=[white]{self._short(v)}[/white]" for k, v in args.items()
        )
        self.console.print(f"{prefix}[bold yellow]▸ {name}[/]([dim]{rendered}[/dim])")

    def tool_result(self, text: str, is_error: bool, prefix: str = "", lines: int = 6) -> None:
        if self.quiet:
            return
        body = text if isinstance(text, str) else "<non-text result>"
        shown = body.strip().splitlines()[:lines]
        style = "red" if is_error else "grey58"
        for line in shown:
            self.console.print(f"{prefix}  [{style}]{line[:180]}[/{style}]")
        total = len(body.strip().splitlines())
        if total > lines:
            self.console.print(f"{prefix}  [dim]… +{total - lines} more lines[/dim]")

    def note(self, text: str) -> None:
        self.console.print(f"[magenta]• {text}[/magenta]")

    def warn(self, text: str) -> None:
        self.console.print(f"[yellow]! {text}[/yellow]")

    def error(self, text: str) -> None:
        self.console.print(f"[red]✗ {text}[/red]")

    def report(self, report: str, status: str) -> None:
        colour = {"completed": "green", "partial": "yellow", "failed": "red"}.get(status, "cyan")
        self.console.print(
            Panel(Markdown(report or "(no report)"), title=f"result: {status}", border_style=colour)
        )

    def usage(self, usage: Any, steps: int, compactions: int, pruned: int) -> None:
        self.console.print(
            f"[dim]{steps} steps · {usage.calls} model calls · "
            f"{usage.input_tokens:,} in / {usage.output_tokens:,} out · "
            f"cache read {usage.cache_read:,} · "
            f"compactions {compactions} · pruned observations {pruned} · "
            f"~${usage.cost_usd:.3f}[/dim]"
        )

    # ------------------------------------------------------------------- input

    def confirm_action(self, tool: str, args: dict[str, Any], verdict: Any) -> str:
        """Returns 'y' (allow once), 'n' (refuse) or 'a' (allow this kind from now on)."""
        style = RISK_STYLE.get(verdict.risk, "yellow")
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold]{tool}[/bold]({self._short(json.dumps(args, ensure_ascii=False))})\n"
                    f"risk: [{style}]{verdict.risk}[/{style}] — {verdict.reason} "
                    f"[dim]({verdict.source})[/dim]"
                ),
                title="confirm before the agent continues",
                border_style=style,
            )
        )
        self.console.print(
            "  [dim]y = allow once · n = refuse · a = allow this kind for the rest of the run[/dim]"
        )
        return Prompt.ask("  [bold]allow?[/bold]", choices=["y", "n", "a"], default="y")

    def ask_user(self, question: str) -> str:
        self.console.print(Panel(Text(question, style="bold"), title="the agent needs you", border_style="magenta"))
        return Prompt.ask("  [bold magenta]you[/bold magenta]")

    # ------------------------------------------------------------------ helper

    @staticmethod
    def _short(value: Any, limit: int = 120) -> str:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        text = text.replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 1] + "…"
