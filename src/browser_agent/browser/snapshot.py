"""Turning a live page into the smallest text that still supports good decisions.

A raw HTML page is 100k-1M characters. The model needs roles, names, state and
stable handles - that is ~1-3% of the bytes. This module produces that view,
renders it as an indented outline, and can express the *difference* between two
views so that an action only costs a few lines of context instead of a fresh
page dump.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from playwright.sync_api import ElementHandle, Error as PlaywrightError, Frame

from .session import BrowserSession

MAX_DEPTH = 8


@dataclass
class Snapshot:
    url: str = ""
    title: str = ""
    modal: bool = False
    scroll_y: int = 0
    scroll_max: int = 0
    nodes: list[dict[str, Any]] = field(default_factory=list)
    ref_frames: dict[str, Frame] = field(default_factory=dict)
    omitted: int = 0
    scope: str = "viewport"

    @property
    def interactive_count(self) -> int:
        return sum(1 for n in self.nodes if n.get("ref"))

    def header(self) -> str:
        pos = f"{self.scroll_y}/{self.scroll_max}px" if self.scroll_max else "no scroll"
        bits = [
            f'page "{self.title[:70]}"',
            self.url[:160],
            f"scroll {pos}",
            f"{self.interactive_count} actionable",
        ]
        if self.modal:
            bits.append("MODAL DIALOG IS OPEN - the rest of the page is blocked")
        if self.omitted:
            bits.append(f"{self.omitted} nodes omitted")
        return " | ".join(bits)

    def lines(self) -> list[str]:
        out = []
        for n in self.nodes:
            indent = "  " * min(n.get("depth", 0), MAX_DEPTH)
            parts = [n["role"]]
            name = n.get("name") or ""
            if name:
                parts.append(f'"{name}"')
            state = n.get("state") or {}
            flags = []
            for key in ("value", "checked", "selected", "expanded", "disabled", "required",
                        "current", "level", "inputType"):
                if key in state:
                    v = state[key]
                    flags.append(key if v is True else f"{key}={v}")
            if "options" in state:
                flags.append("options=[" + ", ".join(state["options"][:12]) + "]")
            if n.get("offscreen"):
                flags.append("offscreen")
            if n.get("scrollable"):
                flags.append("scrollable")
            if flags:
                parts.append("(" + ", ".join(str(f) for f in flags) + ")")
            if n.get("ref"):
                parts.append(f"[{n['ref']}]")
            out.append(indent + "- " + " ".join(parts))
        return out

    def render(self, max_chars: int = 6_000) -> str:
        body = "\n".join(self.lines())
        note = ""
        if len(body) > max_chars:
            body = body[:max_chars]
            body = body[: body.rfind("\n")] if "\n" in body else body
            note = (
                "\n… snapshot truncated to fit the context budget. Use browser_find to "
                "locate a specific control, or browser_scroll to move the viewport."
            )
        return f"{self.header()}\n{body}{note}"


def capture(
    session: BrowserSession,
    scope: str = "viewport",
    interactive_only: bool = False,
    max_nodes: int = 600,
) -> Snapshot:
    """Snapshot every reachable frame of the active tab into one outline."""
    snap = Snapshot(scope=scope)
    ref_start = 0
    for frame_index, frame in enumerate(session.frames()):
        try:
            data = frame.evaluate(
                "opts => window.__agent.collect(opts)",
                {
                    "scope": scope,
                    "interactiveOnly": interactive_only,
                    "maxNodes": max_nodes,
                    "refStart": ref_start,
                },
            )
        except PlaywrightError:
            # Cross-origin or mid-navigation frame: skip it rather than fail the turn.
            continue
        if not data:
            continue

        is_main = frame_index == 0 or not snap.url
        depth_offset = 0
        if is_main:
            snap.url, snap.title = data["url"], data["title"]
            snap.modal = data["modal"]
            snap.scroll_y = data["scroll"]["y"]
            snap.scroll_max = data["scroll"]["max"]
        else:
            if not data["nodes"]:
                continue
            # Nested frames are announced and indented so the model can see the boundary.
            snap.nodes.append({"role": "iframe", "name": data["url"][:80], "depth": 0})
            depth_offset = 1

        for node in data["nodes"]:
            node["depth"] = node.get("depth", 0) + depth_offset
            if node.get("ref"):
                snap.ref_frames[node["ref"]] = frame
            snap.nodes.append(node)
        snap.omitted += data["stats"]["omitted"]
        ref_start = data["stats"]["refEnd"]
    return snap


def resolve(session: BrowserSession, snap: Snapshot, ref: str) -> ElementHandle:
    """Map an opaque ref back to a live element. Raises with a recoverable message."""
    frame = snap.ref_frames.get(ref)
    if frame is None:
        # The ref may have come from browser_find, which registers refs lazily.
        for f in session.frames():
            try:
                if f.evaluate("r => !!window.__agent.get(r)", ref):
                    frame = f
                    break
            except PlaywrightError:
                continue
    if frame is None:
        raise StaleRefError(
            f"ref {ref!r} is not known on this page. Take a fresh browser_snapshot "
            f"(refs are invalidated by navigation) and use a ref from it."
        )
    try:
        handle = frame.evaluate_handle("r => window.__agent.get(r)", ref)
        element = handle.as_element()
    except PlaywrightError as exc:
        raise StaleRefError(f"ref {ref!r} could not be resolved: {exc}") from exc
    try:
        if not element.evaluate("el => el.isConnected"):
            raise StaleRefError(
                f"ref {ref!r} points at an element that has been removed from the page. "
                f"Take a fresh browser_snapshot and use a current ref."
            )
    except PlaywrightError:
        pass
    if element is None:
        raise StaleRefError(
            f"ref {ref!r} no longer points at a live element (the page re-rendered). "
            f"Take a fresh browser_snapshot and retry with the new ref."
        )
    return element


class StaleRefError(RuntimeError):
    """A ref outlived the DOM it pointed at - always recoverable by re-snapshotting."""


def diff(before: Snapshot | None, after: Snapshot, max_lines: int = 40) -> str:
    """A compact description of what the last action changed.

    This is the main context-saving trick: after an action we usually send ~10
    lines of delta instead of a ~300-line snapshot.
    """
    if before is None:
        return after.render()

    head = []
    if before.url != after.url:
        head.append(f"url: {before.url[:90]} -> {after.url[:120]}")
    if before.title != after.title:
        head.append(f'title: "{before.title[:50]}" -> "{after.title[:70]}"')
    if after.modal and not before.modal:
        head.append("a modal dialog opened and now dominates the page")
    if before.modal and not after.modal:
        head.append("the modal dialog closed")
    if before.scroll_y != after.scroll_y:
        head.append(f"scrolled {before.scroll_y} -> {after.scroll_y} of {after.scroll_max}px")

    old, new = before.lines(), after.lines()
    added, removed = [], []
    for tag, _i1, _i2, j1, j2 in _opcodes(old, new):
        if tag in ("insert", "replace"):
            added.extend(new[j1:j2])
    for tag, i1, i2, _j1, _j2 in _opcodes(old, new):
        if tag in ("delete", "replace"):
            removed.extend(old[i1:i2])

    churn = len(added) + len(removed)
    if new and churn > 0.6 * len(new):
        # More than half the page turned over: a fresh outline is both smaller
        # and easier to reason about than a delta against a page that is gone.
        return after.render()

    if not added and not removed and not head:
        return (
            f"{after.header()}\nNo visible change in the page outline. The action may "
            f"have had no effect, or the result is outside the current viewport."
        )

    parts = [after.header()]
    if head:
        parts.append("changes: " + "; ".join(head))
    if added:
        parts.append(f"appeared ({len(added)}):")
        parts.extend("  " + line.strip() for line in added[:max_lines])
        if len(added) > max_lines:
            parts.append(f"  … {len(added) - max_lines} more new lines - browser_snapshot to see all")
    if removed:
        shown = [line.split(" [")[0] for line in removed[:8]]
        parts.append(f"disappeared ({len(removed)}):")
        parts.extend("  " + line.strip() for line in shown)
        if len(removed) > len(shown):
            parts.append(f"  … {len(removed) - len(shown)} more")
    return "\n".join(parts)


def _opcodes(old: list[str], new: list[str]):
    return difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
