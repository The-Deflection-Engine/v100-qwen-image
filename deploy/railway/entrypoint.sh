#!/usr/bin/env bash
# Lightseed Studio — Railway entrypoint.
#
# Process tree (production):
#
#   tailscaled (userspace, ephemeral node, tag:railway)
#     └ outbound HTTP proxy 127.0.0.1:1055   <- the gateway dials the PC here
#   oauth2-proxy  0.0.0.0:$PORT              <- the only public listener
#     └ upstream http://127.0.0.1:18080
#   node gateway 127.0.0.1:18080             <- injects identity + proxy token
#
# Order matters: Tailscale must be up before the gateway can reach the PC, and
# oauth2-proxy starts last so no request can arrive at a gateway that has no
# proxy in front of it.
#
# Dev mode (LIGHTSEED_NO_TAILSCALE=1 and no oauth2 config) skips both layers:
# the gateway binds $PORT itself and every request is the dev identity. That
# mode exists for the local smoke test and is refused if oauth2 credentials are
# present, so it can never be reached on a deployed service by accident.

set -Eeuo pipefail

readonly LOG_PREFIX="entrypoint"
readonly GATEWAY_DIR="/app"

log() {
  printf '%s %s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$LOG_PREFIX" "$*" >&2
}

die() {
  log "FATAL: $*"
  exit 1
}

# ------------------------------------------------------------------- env ---

require() {
  local name="$1"
  local value="${!name:-}"
  if [ -z "$value" ]; then
    die "$name is required"
  fi
}

optional() {
  local name="$1"
  local fallback="$2"
  local value="${!name:-}"
  if [ -z "$value" ]; then
    printf '%s' "$fallback"
  else
    printf '%s' "$value"
  fi
}

is_truthy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

PORT="$(optional PORT "")"
[ -n "$PORT" ] || die "PORT is required (Railway sets it)"
case "$PORT" in
  ''|*[!0-9]*) die "PORT must be a number" ;;
esac
[ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || die "PORT is out of range"

LIGHTSEED_UPSTREAM="${LIGHTSEED_UPSTREAM:-}"
LIGHTSEED_PROXY_TOKEN="${LIGHTSEED_PROXY_TOKEN:-}"
LIGHTSEED_DEV_MODE="${LIGHTSEED_DEV_MODE:-}"
LIGHTSEED_NO_TAILSCALE="${LIGHTSEED_NO_TAILSCALE:-}"
LIGHTSEED_PUBLIC_URL="$(optional LIGHTSEED_PUBLIC_URL "")"

DEV_MODE=0
if is_truthy "$LIGHTSEED_DEV_MODE"; then DEV_MODE=1; fi
NO_TAILSCALE=0
if is_truthy "$LIGHTSEED_NO_TAILSCALE"; then NO_TAILSCALE=1; fi

OAUTH_CLIENT_ID="${OAUTH2_PROXY_CLIENT_ID:-}"
OAUTH_CLIENT_SECRET="${OAUTH2_PROXY_CLIENT_SECRET:-}"
OAUTH_COOKIE_SECRET="${OAUTH2_PROXY_COOKIE_SECRET:-}"

# The dev bypass is all-or-nothing: no oauth2 credentials, and the caller has to
# say so explicitly.
if [ "$DEV_MODE" -eq 0 ] && [ "$NO_TAILSCALE" -eq 1 ] &&
   [ -z "$OAUTH_CLIENT_ID" ] && [ -z "$OAUTH_CLIENT_SECRET" ]; then
  log "no oauth2 credentials and LIGHTSEED_NO_TAILSCALE=1: running in dev mode"
  DEV_MODE=1
fi

if [ "$DEV_MODE" -eq 1 ]; then
  if [ -n "$OAUTH_CLIENT_ID" ] || [ -n "$OAUTH_CLIENT_SECRET" ]; then
    die "dev mode refuses to run with oauth2 credentials set: unset LIGHTSEED_DEV_MODE/LIGHTSEED_NO_TAILSCALE"
  fi
  # Dev mode answers with a fixed identity and no sign-in, so it must never
  # run where real people can reach it.
  if [ -n "${RAILWAY_PROJECT_ID:-}" ] || [ -n "${RAILWAY_SERVICE_ID:-}" ]; then
    die "dev mode refuses to run on Railway"
  fi
  if [ -n "${TS_AUTHKEY:-}" ]; then
    die "dev mode refuses to run with TS_AUTHKEY set"
  fi
  if ! is_truthy "$LIGHTSEED_NO_TAILSCALE" && ! is_truthy "$LIGHTSEED_DEV_MODE"; then
    die "dev mode needs LIGHTSEED_NO_TAILSCALE=1 or LIGHTSEED_DEV_MODE=1"
  fi
  LIGHTSEED_DEV_EMAIL="${LIGHTSEED_DEV_EMAIL:-}"
  [ -n "$LIGHTSEED_DEV_EMAIL" ] || die "LIGHTSEED_DEV_EMAIL is required in dev mode"
fi

require LIGHTSEED_UPSTREAM
require LIGHTSEED_PROXY_TOKEN

if [ "$DEV_MODE" -eq 0 ]; then
  require OAUTH2_PROXY_CLIENT_ID
  require OAUTH2_PROXY_CLIENT_SECRET
  require OAUTH2_PROXY_COOKIE_SECRET
  [ -n "$LIGHTSEED_PUBLIC_URL" ] || die "LIGHTSEED_PUBLIC_URL is required (https://<railway domain>)"
  case "$LIGHTSEED_PUBLIC_URL" in
    https://*) ;;
    *) die "LIGHTSEED_PUBLIC_URL must start with https://" ;;
  esac
  if [ "${#LIGHTSEED_PROXY_TOKEN}" -lt 32 ]; then
    die "LIGHTSEED_PROXY_TOKEN must be at least 32 characters"
  fi

  # oauth2-proxy's own rule: decode the secret as URL-safe base64; if that works
  # and gives 16, 24 or 32 bytes, use it, otherwise use the raw text, which must
  # then be 16, 24 or 32 characters. Standard base64 ('+' or '/', as printed by
  # `openssl rand -base64`) is NOT URL-safe, so it falls through to the raw
  # length and fails. Checking here turns that into a clear boot message.
  cookie_hint="generate one with: openssl rand -hex 16"
  case "$OAUTH_COOKIE_SECRET" in
    *[+/]*) die "OAUTH2_PROXY_COOKIE_SECRET contains '+' or '/': oauth2-proxy needs URL-safe base64; $cookie_hint" ;;
  esac
  cookie_bytes=0
  if cookie_decoded="$(printf '%s' "$OAUTH_COOKIE_SECRET" | tr -- '-_' '+/' | base64 -d 2>/dev/null | wc -c)"; then
    cookie_bytes="$(printf '%s' "$cookie_decoded" | tr -d ' ')"
  fi
  case "$cookie_bytes:${#OAUTH_COOKIE_SECRET}" in
    16:*|24:*|32:*|*:16|*:24|*:32) ;;
    *) die "OAUTH2_PROXY_COOKIE_SECRET must be 16, 24 or 32 bytes (raw, or URL-safe base64); $cookie_hint" ;;
  esac
fi

LIGHTSEED_GATEWAY_PORT="$(optional LIGHTSEED_GATEWAY_PORT 18080)"
LIGHTSEED_TAILSCALE_PROXY="$(optional LIGHTSEED_TAILSCALE_PROXY 127.0.0.1:1055)"
LIGHTSEED_TAILSCALED_SOCKET="$(optional LIGHTSEED_TAILSCALED_SOCKET /tmp/lightseed-ts/ts.sock)"
LIGHTSEED_HOSTNAME="$(optional LIGHTSEED_HOSTNAME lightseed-railway)"
TAILSCALE_SOCKS5_PROXY="$(optional LIGHTSEED_TAILSCALE_SOCKS5_PROXY 127.0.0.1:1056)"

export PORT LIGHTSEED_UPSTREAM LIGHTSEED_PROXY_TOKEN LIGHTSEED_GATEWAY_PORT
export LIGHTSEED_TAILSCALE_PROXY LIGHTSEED_TAILSCALED_SOCKET

if [ "$DEV_MODE" -eq 1 ]; then
  export LIGHTSEED_MODE=dev
  export LIGHTSEED_NO_TAILSCALE=1
else
  export LIGHTSEED_MODE=proxy
  export LIGHTSEED_NO_TAILSCALE=0
fi

# ------------------------------------------------------------- lifecycle ---

CHILD_PIDS=()
SHUTTING_DOWN=0

stop_children() {
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done
  # Give each child a moment, then be firm.
  local waited=0
  while [ "$waited" -lt 10 ]; do
    local alive=0
    for pid in "${CHILD_PIDS[@]:-}"; do
      [ -n "$pid" ] || continue
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ "$alive" -eq 0 ] && return 0
    sleep 1
    waited=$((waited + 1))
  done
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}

on_signal() {
  local signal="$1"
  if [ "$SHUTTING_DOWN" -eq 1 ]; then
    return
  fi
  SHUTTING_DOWN=1
  log "received $signal, stopping children"
  stop_children
  exit 0
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal QUIT' QUIT
trap 'stop_children' EXIT

# Any child dying is fatal: a half-running stack serves errors or, worse, an
# unsigned gateway. Railway restarts the service on a non-zero exit.
wait_for_children() {
  local pid
  set +e
  wait -n
  local status=$?
  set -e
  if [ "$SHUTTING_DOWN" -eq 1 ]; then
    exit 0
  fi
  for pid in "${CHILD_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      log "child process $pid exited; shutting the service down (status $status)"
    fi
  done
  exit "$status"
}

# ------------------------------------------------------------- tailscale ---

start_tailscale() {
  local authkey="${TS_AUTHKEY:-}"
  [ -n "$authkey" ] || die "TS_AUTHKEY is required (an ephemeral, reusable, tagged auth key)"

  mkdir -p "$(dirname "$LIGHTSEED_TAILSCALED_SOCKET")"
  rm -f "$LIGHTSEED_TAILSCALED_SOCKET"

  log "starting tailscaled (userspace, ephemeral node, no TUN)"
  # --state=mem: the node is ephemeral: it re-registers on every boot and the
  #   old entry expires, so a restarted container cannot accumulate devices.
  # -port=0: no fixed WireGuard port; the ACL only needs outbound to the PC.
  tailscaled \
    --tun=userspace-networking \
    --state=mem: \
    --socket="$LIGHTSEED_TAILSCALED_SOCKET" \
    --outbound-http-proxy-listen="$LIGHTSEED_TAILSCALE_PROXY" \
    --socks5-server="$TAILSCALE_SOCKS5_PROXY" \
    -port=0 \
    --no-logs-no-support &
  CHILD_PIDS+=("$!")

  # tailscaled creates its socket early; `tailscale up` retries the control
  # plane itself, but it needs the socket to exist first.
  local waited=0
  while [ ! -S "$LIGHTSEED_TAILSCALED_SOCKET" ]; do
    if [ "$waited" -ge 30 ]; then
      die "tailscaled did not create $LIGHTSEED_TAILSCALED_SOCKET within 30s"
    fi
    sleep 1
    waited=$((waited + 1))
  done

  log "bringing Tailscale up as $LIGHTSEED_HOSTNAME"
  # --reset: accept the defaults for every setting not named here, so a
  #   Tailscale release that adds a setting cannot make boot fail.
  # The tag is not passed on this command line: a tagged auth key forces the
  #   node's tags, and asking for one the key lacks is a hard error.
  if ! tailscale --socket="$LIGHTSEED_TAILSCALED_SOCKET" up \
      --authkey="$authkey" \
      --hostname="$LIGHTSEED_HOSTNAME" \
      --reset >/dev/null 2>&1; then
    # `tailscale up` can fail while the control plane is still warming up; one
    # retry after a pause covers a Railway cold start without hiding a bad key.
    log "first 'tailscale up' failed, retrying once in 5s"
    sleep 5
    tailscale --socket="$LIGHTSEED_TAILSCALED_SOCKET" up \
      --authkey="$authkey" \
      --hostname="$LIGHTSEED_HOSTNAME" \
      --reset ||
      die "tailscale up failed: check TS_AUTHKEY, its tag and the ACL"
  fi

  # BackendState Running is the only proof the node is on the tailnet; a
  # socket alone proves nothing.
  waited=0
  local state=""
  while :; do
    state="$(batch_state 2>/dev/null || true)"
    if [ "$state" = "Running" ]; then
      break
    fi
    if [ "$waited" -ge 60 ]; then
      die "Tailscale did not reach Running within 60s (last state: ${state:-unknown})"
    fi
    sleep 2
    waited=$((waited + 2))
  done
  log "Tailscale is Running"
}

# Reads BackendState straight from the local API with node (curl is not in the
# slim base image and adding it for one call is not worth the surface).
batch_state() {
  node -e '
    const http = require("node:http");
    const socket = process.argv[1];
    const req = http.request({
      socketPath: socket,
      path: "/localapi/v0/status",
      method: "GET",
      headers: { host: "local-tailscaled.sock" },
      timeout: 5000,
    }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        if (res.statusCode !== 200) { process.exit(3); }
        try {
          process.stdout.write(String(JSON.parse(Buffer.concat(chunks)).BackendState || ""));
        } catch { process.exit(4); }
      });
    });
    req.on("timeout", () => { req.destroy(); process.exit(5); });
    req.on("error", () => process.exit(6));
    req.end();
  ' "$LIGHTSEED_TAILSCALED_SOCKET"
}

# --------------------------------------------------------------- gateway ---

start_gateway() {
  log "starting gateway (mode $LIGHTSEED_MODE)"
  node "$GATEWAY_DIR/gateway.js" &
  CHILD_PIDS+=("$!")

  # Probe the address the gateway actually binds, which differs by mode:
  # production is loopback-only, dev serves the public PORT. Probing the wrong
  # one makes the check fail 30s after a perfectly healthy start.
  local probe_host probe_port
  if [ "$LIGHTSEED_MODE" = "dev" ]; then
    probe_host="127.0.0.1"
    probe_port="$PORT"
  else
    probe_host="127.0.0.1"
    probe_port="$LIGHTSEED_GATEWAY_PORT"
  fi

  local waited=0
  while :; do
    if node -e '
      const net = require("node:net");
      const socket = net.connect(Number(process.argv[2]), process.argv[1]);
      socket.on("connect", () => { socket.end(); process.exit(0); });
      socket.on("error", () => process.exit(1));
    ' "$probe_host" "$probe_port" 2>/dev/null; then
      break
    fi
    if [ "$waited" -ge 30 ]; then
      die "gateway did not listen on $probe_host:$probe_port within 30s"
    fi
    sleep 1
    waited=$((waited + 1))
  done
  log "gateway listening on $probe_host:$probe_port"
}

# ----------------------------------------------------------- oauth2-proxy ---

start_oauth2_proxy() {
  log "starting oauth2-proxy on 0.0.0.0:$PORT"
  # Everything secret comes from the environment: Railway variables never touch
  # the filesystem or the process list. Everything else lives in
  # oauth2-proxy.args, which the smoke test also feeds to the real binary, so
  # the flags that ship are the flags that were exercised.
  export OAUTH2_PROXY_HTTP_ADDRESS="0.0.0.0:$PORT"
  export OAUTH2_PROXY_REDIRECT_URL="${LIGHTSEED_PUBLIC_URL}/oauth2/callback"
  # Note the plural: oauth2-proxy's config key is "upstreams", so the
  # environment variable that sets it is OAUTH2_PROXY_UPSTREAMS. The singular
  # spelling is accepted silently and ignored, which turns every authenticated
  # request into oauth2-proxy's own bare 404. Do not "fix" this to singular.
  export OAUTH2_PROXY_UPSTREAMS="http://127.0.0.1:${LIGHTSEED_GATEWAY_PORT}/"

  local args=()
  local line
  # One line is one argument, verbatim: no word splitting, no globbing. Splitting
  # would turn "--scope=openid email" into two arguments and silently drop the
  # email scope (Google then issues a token with no email claim), and a "*" would
  # be expanded against whatever happens to be in the working directory.
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in \#*) continue ;; esac
    args+=("$line")
  done < "$GATEWAY_DIR/oauth2-proxy.args"
  [ "${#args[@]}" -gt 0 ] ||
    die "oauth2-proxy.args is empty or missing; refusing to start without sign-in flags"

  oauth2-proxy "${args[@]}" &
  CHILD_PIDS+=("$!")
}

# ------------------------------------------------------------------- main ---

log "starting Lightseed Studio edge (dev=$DEV_MODE, node $(node -v))"

if [ "$NO_TAILSCALE" -eq 0 ]; then
  start_tailscale
fi

start_gateway

if [ "$DEV_MODE" -eq 0 ]; then
  start_oauth2_proxy
  log "ready: public listener on :$PORT, upstream $LIGHTSEED_UPSTREAM"
else
  log "ready: dev mode, gateway on :$PORT, upstream $LIGHTSEED_UPSTREAM"
fi

wait_for_children
