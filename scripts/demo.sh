#!/usr/bin/env bash
# One command: run a task on the bundled demo site, record it, and compose the
# side-by-side video (terminal + browser).
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-Прочитай письма во входящих, определи какие из них спам, и удали их. В конце дай короткий отчёт.}"
PY=.venv/bin/python

.venv/bin/browser-agent --demo --record-video --slow-mo 90 "$TASK"

RUN=$(ls -td runs/*/ | head -1)
echo "run: $RUN"
$PY scripts/make_video.py "${RUN%/}" -o "${RUN%/}/demo.mp4"
echo "video: ${RUN%/}/demo.mp4"
