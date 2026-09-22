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


_REMUXED: dict[str, Path] = {}


def probe(path: Path) -> tuple[float, int, int]:
    """Duration and size of a recording.

    A webm whose writer was interrupted carries no duration in its header, and
    Playwright leaves one behind whenever the browser goes away abruptly. Remuxing
    rebuilds the header without re-encoding, which is fast and lossless.
    """
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        text=True,
    )
    data = json.loads(out)
    stream = data["streams"][0]
    duration = data.get("format", {}).get("duration")
    if duration in (None, "N/A"):
        fixed = _REMUXED.get(str(path))
        if fixed is None:
            fixed = Path(tempfile.mkdtemp(prefix="agentvid-fix-")) / (path.stem + ".webm")
            print(f"  {path.name}: header has no duration, remuxing")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c", "copy", str(fixed)],
                           check=True)
            _REMUXED[str(path)] = fixed
        return probe(fixed)
    return float(duration), int(stream["width"]), int(stream["height"])


def build_timeline(records: list[dict], run_dir: Path) -> list[tuple[float, float, Path]]:
    """Which tab was on screen when.

    Playwright records one track per page, so a run that opens tabs leaves
    several files, each starting when its page was created. The trace stamps
    every tool result with the track that was visible, which is enough to cut
    between them instead of showing a tab where nothing was happening.
    """
    tracks: dict[str, float] = {}      # path -> when its recording started
    for rec in records:
        if rec["kind"] == "page_opened" and rec.get("video"):
            tracks.setdefault(rec["video"], rec["t"])

    marks: list[tuple[float, str]] = []
    for rec in records:
        video = rec.get("video")
        if not video:
            continue
        tracks.setdefault(video, rec["t"])
        if not marks or marks[-1][1] != video:
            marks.append((rec["t"], video))

    if not marks:
        return []

    end = max(r["t"] for r in records) + 1.0
    spans: list[tuple[float, float, Path]] = []
    for i, (start, video) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else end
        path = Path(video)
        if not path.exists():
            path = run_dir / "video" / path.name
        if not path.exists() or stop - start < 0.4:
            continue
        spans.append((start - tracks[video], stop - tracks[video], path))
    return spans


def browser_stream(
    spans: list[tuple[float, float, Path]], fps: int, speed: float, hold: float
) -> tuple[list[str], str, float, str]:
    """ffmpeg inputs and a filter that stitches the visible tabs into one track."""
    # Probe every track first: that is what repairs a header-less recording, and
    # the repaired path is the one ffmpeg must be given as an input.
    durations: dict[Path, float] = {}
    for _s, _e, path in spans:
        if path not in durations:
            durations[path] = probe(path)[0]

    files: list[Path] = []
    for _s, _e, path in spans:
        actual = _REMUXED.get(str(path), path)
        if actual not in files:
            files.append(actual)

    parts, labels, total = [], [], 0.0
    for idx, (start, stop, path) in enumerate(spans):
        duration = durations[path]
        path = _REMUXED.get(str(path), path)
        start, stop = max(0.0, start), min(stop, duration)
        if stop - start < 0.4:
            continue
        stream = files.index(path) + 1  # input 0 is the terminal panel
        parts.append(
            f"[{stream}:v]trim=start={start:.2f}:end={stop:.2f},"
            f"setpts=PTS-STARTPTS,fps={fps},scale=1440:900:force_original_aspect_ratio=decrease,"
            f"pad=1440:900:(ow-iw)/2:(oh-ih)/2[t{idx}]"
        )
        labels.append(f"[t{idx}]")
        total += stop - start

    joined = "".join(labels)
    speed_filter = "" if speed == 1.0 else f",setpts=PTS/{speed}"
    hold_filter = "" if hold <= 0 else f",tpad=stop_mode=clone:stop_duration={hold}"
    filt = ";".join(parts) + f";{joined}concat=n={len(labels)}:v=1[cat]"
    filt += f";[cat]fps={fps}{speed_filter}{hold_filter}[b]"
    return [str(f) for f in files], filt, total, speed_filter


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
    ap.add_argument("--panel-width", type=int, default=900)
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument(
        "--hold", type=float, default=8.0,
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

    records = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
    # Align the log clock with the footage: recording starts when the context does.
    offset = next((r["t"] for r in records if r["kind"] == "browser_started"), 0.0)
    spans = build_timeline(records, args.run_dir)
    if not spans:
        # A trace from before tab tracking: fall back to the longest single track.
        biggest = max(videos, key=lambda p: p.stat().st_size)
        spans = [(0.0, probe(biggest)[0], biggest)]

    vh = 900
    font = load_font(args.font_size)
    line_h = args.font_size + 5
    cols = max(40, int((args.panel_width - 24) / (args.font_size * 0.6)))
    rows = max(10, (vh - 44) // line_h)
    lines = trace_to_lines(trace_path, cols)

    # The recording can end before the run does - a browser that goes away while
    # the agent is still finishing leaves a shorter track than the trace. Freeze
    # on the last frame long enough for the log to reach its own end, or the
    # video would cut away before the result the whole run was for.
    trace_end = max(r["t"] for r in records) - offset
    hold = args.hold
    spans_seconds = sum(min(stop, probe(path)[0]) - max(0.0, start) for start, stop, path in spans)
    missing = trace_end - spans_seconds
    if missing > 1:
        hold += missing / args.speed
        print(f"recording is {missing:.0f}s shorter than the run; holding {hold:.0f}s at the end")

    inputs, browser_filter, browser_seconds, _ = browser_stream(
        spans, args.fps, args.speed, hold
    )
    print(f"{len(inputs)} tab recording(s), {len(spans)} visible segment(s), "
          f"{browser_seconds:.0f}s of footage")

    tmp = Path(tempfile.mkdtemp(prefix="agentvid-"))
    # The terminal panel is rendered at the *output* rate, so speeding the video
    # up keeps the log in step with the browser instead of drifting away from it.
    frames = int((browser_seconds / args.speed + hold) * args.fps)
    for i in range(frames):
        now = i / args.fps * args.speed + offset
        panel = render_panel(lines, now, (args.panel_width, vh), font, line_h, rows)
        panel.save(tmp / f"f{i:05d}.png")
    print(f"rendered {frames} terminal frames at {args.panel_width}x{vh}")

    out = args.output or (args.run_dir / "demo.mp4")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png")]
    for path in inputs:
        cmd += ["-i", path]
    cmd += [
        "-filter_complex", f"{browser_filter};[0:v][b]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2500:])
        return 1
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out}  ({1440 + args.panel_width}x{vh})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
