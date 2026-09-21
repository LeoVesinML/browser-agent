#!/usr/bin/env bash
# The brief forbids three things: canned action sequences, pre-written selectors,
# and hints about where things live on a site. This is the mechanical half of
# that promise - it runs in CI, over the whole agent, not just the prompt.
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

AGENT=src/browser_agent

# 1. No canned action sequences for particular tasks.
check "no task recipes" \
      '(def|function) *(delete_spam|order_food|apply_to_job|checkout|add_to_cart|login_to)' "$AGENT"

# 2. No pre-written selectors. Generic HTML/ARIA queries inside the page
#    extractor are how the accessibility model is built at all; what is banned is
#    a site's own hooks, which is what `a[data-qa='vacancy']` in the brief means.
check "no site test-id selectors anywhere" \
      "data-qa|data-test|data-testid|data-automation|data-marker|data-widget" "$AGENT"
check "no selectors in the agent's own logic" \
      'querySelector|css=|xpath=|\.getElementById' "$AGENT/agent" "$AGENT/browser/actions.py"

# 3. No hints about where things live or what they are called.
check "no site urls in the agent" \
      'https?://(www\.)?[a-z0-9-]+\.(ru|com|org|net)' "$AGENT"
check "no site names in the prompt" \
      '(yandex|hh\.ru|ozon|gmail|habr|avito|papajohns|chitai|wildberries|delivery)' "$AGENT/agent/prompts.py"
check "no url paths hinted in the prompt" \
      '/(vacancies|cart|checkout|inbox|orders|search)\b' "$AGENT/agent/prompts.py"

# The risk vocabulary in safety.py is deliberately exempt: it can only make the
# agent stop and ask, never help it find anything. Enforce that it stays that
# way - `assess` must be reachable only through the gate.
callers=$(grep -rn "\.assess(" "$AGENT" | grep -v "safety.py" || true)
if [ -n "$callers" ]; then
  echo "FAIL: the risk classifier is used outside the safety gate"
  echo "$callers" | sed 's/^/    /'; fail=1
else
  echo "ok:   risk vocabulary can only gate, never choose"
fi

echo
if [ "$fail" = 0 ]; then echo "no site knowledge is baked in"; else echo "site knowledge leaked into the agent"; fi
exit $fail
