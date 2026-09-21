#!/usr/bin/env python3
"""Compose a demo video: the terminal log next to the browser doing the work.

Playwright records the browser itself; `runs/<ts>/trace.jsonl` carries every tool
call with a timestamp. This script replays the trace as a terminal panel, frame
by frame, and stacks it beside the browser recording.

    python scripts/make_video.py runs/20260921-181500 -o demo.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
BG = (14, 16, 22)
COLOURS = {
    "task": (120, 210, 255),
    "tool": (240, 200, 90),
    "result": (135, 142, 160),
    "error": (240, 110, 110),
    "note": (200, 130, 240),
    "gate": (255, 160, 60),
    "sub": (110, 170, 255),
    "report": (120, 230, 150),
    "plain": (215, 220, 230),
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines() or [""]:
        while len(raw) > width:
            out.append(raw[:width])
            raw = raw[width:]
        out.append(raw)
    return out


def trace_to_lines(path: Path, cols: int) -> list[tuple[float, str, str]]:
    """Flatten the trace into (timestamp, colour-key, text) terminal lines."""
    lines: list[tuple[float, str, str]] = []

    def push(t: float, kind: str, text: str, indent: str = "") -> None:
        for chunk in wrap(text, cols - len(indent)):
            lines.append((t, kind, indent + chunk))

    for record in (json.loads(l) for l in path.read_text().splitlines() if l.strip()):
        t, kind = record["t"], record["kind"]
        if kind == "task":
            push(t, "task", f"task> {record['task']}")
            push(t, "result", f"model {record.get('model', '')} · {record.get('url', '')}")
        elif kind == "tool_call":
            args = json.dumps(record.get("args", {}), ensure_ascii=False)
            push(t, "tool", f"▸ {record['tool']}({args[:200]})")
        elif kind == "tool_result":
            body = (record.get("content") or "").strip().splitlines()
            for line in body[:3]:
                push(t, "error" if record.get("error") else "result", "  " + line[:cols - 4])
        elif kind == "gate":
            verdict = "allowed" if record.get("allowed") else "REFUSED"
            push(t, "gate", f"⚠ safety gate [{record.get('risk')}] {record.get('reason', '')} → {verdict}")
        elif kind == "compaction":
            push(t, "note", f"• context compacted {record['before']} → {record['after']} tokens")
        elif kind == "subagent_start":
            push(t, "sub", f"│ sub-agent: {record['instruction'][:180]}")
        elif kind == "subagent_tool":
            push(t, "sub", f"│   ▸ {record['tool']}")
        elif kind == "subagent_done":
            push(t, "sub", f"│ ↩ returned {len(record.get('answer', ''))} chars")
        elif kind == "ask_user":
            push(t, "note", f"? {record['question']}")
            push(t, "note", f"  you> {record['answer']}")
        elif kind == "run_done":
            push(t, "report", f"── result: {record['status']} ──")
            push(t, "report", record.get("report", ""))
    return lines


def probe(path: Path) -> tuple[float, int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        text=True,
    )
    data = json.loads(out)
    stream = data["streams"][0]
    return float(data["format"]["duration"]), int(stream["width"]), int(stream["height"])


def render_panel(
    lines: list[tuple[float, str, str]],
    now: float,
    size: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    line_h: int,
    rows: int,
) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], 26], fill=(26, 30, 40))
    draw.text((12, 6), "browser-agent — terminal", font=font, fill=(150, 160, 180))

    visible = [l for l in lines if l[0] <= now][-rows:]
    y = 34
    for _, kind, text in visible:
        draw.text((12, y), text, font=font, fill=COLOURS.get(kind, COLOURS["plain"]))
        y += line_h
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--panel-width", type=int, default=760)
    ap.add_argument("--font-size", type=int, default=13)
    ap.add_argument(
        "--hold", type=float, default=6.0,
        help="Freeze on the last frame this long, so the final report is readable.",
    )
    ap.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed both streams up by this factor (4 turns a 12-minute run into 3 minutes).",
    )
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe are required (brew install ffmpeg)")
        return 2

    trace_path = args.run_dir / "trace.jsonl"
    videos = sorted((args.run_dir / "video").glob("*.webm"))
    if not trace_path.exists() or not videos:
        print(f"need {trace_path} and a recording in {args.run_dir / 'video'}")
        return 2
    video = max(videos, key=lambda p: p.stat().st_size)

    duration, vw, vh = probe(video)
    font = load_font(args.font_size)
    line_h = args.font_size + 5
    cols = max(40, int((args.panel_width - 24) / (args.font_size * 0.6)))
    rows = max(10, (vh - 44) // line_h)

    # Align the trace clock with the recording: the browser starts recording when
    # the context is created, which the trace marks with `browser_started`.
    records = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
    offset = next((r["t"] for r in records if r["kind"] == "browser_started"), 0.0)
    lines = trace_to_lines(trace_path, cols)

    tmp = Path(tempfile.mkdtemp(prefix="agentvid-"))
    # The terminal panel is rendered at the *output* rate, so speeding the video
    # up keeps the log in step with the browser instead of drifting away from it.
    frames = int((duration / args.speed + args.hold) * args.fps)
    for i in range(frames):
        now = i / args.fps * args.speed + offset
        panel = render_panel(lines, now, (args.panel_width, vh), font, line_h, rows)
        panel.save(tmp / f"f{i:05d}.png")
    print(f"rendered {frames} terminal frames at {args.panel_width}x{vh}")

    out = args.output or (args.run_dir / "demo.mp4")
    speed_filter = "" if args.speed == 1.0 else f",setpts=PTS/{args.speed}"
    hold_filter = "" if args.hold <= 0 else f",tpad=stop_mode=clone:stop_duration={args.hold}"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png"),
        "-i", str(video),
        "-filter_complex",
        f"[1:v]fps={args.fps}{speed_filter}{hold_filter}[b];[0:v][b]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(out),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out}  ({vw + args.panel_width}x{vh})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
