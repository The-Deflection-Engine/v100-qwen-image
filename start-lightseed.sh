#!/usr/bin/env bash
# Start Lightseed Studio for remote users: the image studio on this PC, with the
# remote listener that Railway's gateway (https://lightseed.up.railway.app)
# reaches over Tailscale. For the local-only app, run ./start-web.sh instead.
#
#   ./start-lightseed.sh              start the Railway gateway, then this PC's app
#   ./start-lightseed.sh --no-railway start this PC's app only (gateway already up)
#
# Any other arguments are passed on to the server, as with start-web.sh.
# Stop the app with Ctrl+C. To take the gateway down as well:
#   railway down -y --project "$RAILWAY_PROJECT" --service "$RAILWAY_SERVICE" \
#     --environment "$RAILWAY_ENVIRONMENT"
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$PROJECT_DIR"

PUBLIC_ORIGIN="${QWEN_REMOTE_ORIGIN:-https://lightseed.up.railway.app}"
RAILWAY_PROJECT="${RAILWAY_PROJECT:-42e3ccb6-fc5e-45ce-b729-00d5ea97383b}"
RAILWAY_SERVICE="${RAILWAY_SERVICE:-a4105a7e-8dbd-48dc-9efa-be003071a62e}"
RAILWAY_ENVIRONMENT="${RAILWAY_ENVIRONMENT:-production}"

start_railway=1
if [[ ${1:-} == --no-railway ]]; then
  start_railway=0
  shift
fi

# The shared secret Railway's gateway sends on every request. Without it
# start-web.sh would silently start the local-only app.
token_file="${QWEN_REMOTE_TOKEN_FILE:-cache/remote-token}"
if [[ ! -f $token_file ]]; then
  echo "error: $token_file is missing; it must hold the same value as Railway's" \
       "LIGHTSEED_PROXY_TOKEN (see docs/railway.md)" >&2
  exit 1
fi
if ! tailscale ip -4 >/dev/null 2>&1; then
  echo "error: Tailscale is not up on this PC; Railway reaches the app through it." \
       "Run 'sudo tailscale up' first." >&2
  exit 1
fi

if [[ $start_railway == 1 ]]; then
  if command -v railway >/dev/null 2>&1; then
    # --from-source builds the latest commit on the service's branch (main), so
    # this works after `railway down` as well as when the gateway is running.
    echo "starting the Railway gateway (build takes about a minute)..."
    railway redeploy --from-source -y --project "$RAILWAY_PROJECT" \
      --service "$RAILWAY_SERVICE" --environment "$RAILWAY_ENVIRONMENT" \
      || echo "warning: could not start the Railway gateway; redeploy it from" \
              "the Railway dashboard. Starting this PC's app anyway." >&2
  else
    echo "warning: the railway CLI is not installed; if the gateway is down," \
         "redeploy it from the Railway dashboard." >&2
  fi
fi

# Anyone who gets through Google sign-in (Google's Test users list) gets a
# Q8-only account on first visit; people removed on the Admin tab stay out.
export QWEN_REMOTE_ORIGIN="$PUBLIC_ORIGIN"
export QWEN_REMOTE_AUTO_INVITE="${QWEN_REMOTE_AUTO_INVITE:-1}"
exec ./start-web.sh "$@"
