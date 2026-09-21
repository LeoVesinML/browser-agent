#!/usr/bin/env bash
# The task forbids baked-in site knowledge. This is the mechanical part of that
# promise: no URLs, no selectors, no site names anywhere the agent reads from.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

check() {  # name, pattern, paths...
  local name="$1" pattern="$2"; shift 2
  local hits
  hits=$(grep -rnE "$pattern" "$@" 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "FAIL: $name"; echo "$hits" | sed 's/^/    /'; fail=1
  else
    echo "ok:   $name"
  fi
}

PROMPTS=src/browser_agent/agent/prompts.py
check "no urls in the system prompt"        'https?://'            "$PROMPTS"
check "no site names in the system prompt"  '(yandex|hh\.ru|ozon|google|gmail|habr|delivery|avito)' "$PROMPTS"
check "no css selectors in agent logic"     'querySelector|\[data-|\.class|css=|xpath=' \
      src/browser_agent/agent
check "no task recipes"                     '(def|function) *(delete_spam|order_food|apply_to_job|checkout)' \
      src/browser_agent

echo
if [ "$fail" = 0 ]; then echo "no site knowledge is baked in"; else echo "site knowledge leaked into the agent"; fi
exit $fail
