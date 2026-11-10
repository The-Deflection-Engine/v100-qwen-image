#!/usr/bin/env bash
# Local smoke test for the Lightseed Studio Railway image.
#
#   ./smoke.sh            # build + run + assert
#   IMAGE=lightseed-railway:smoke ./smoke.sh --no-build
#
# What it does:
#   1. builds deploy/railway into a local image (docker or podman);
#   2. starts a FAKE upstream ("the PC") on 127.0.0.1 -- a small Python HTTP
#      server that echoes the headers it received and can stream slowly;
#   3. runs the image with host networking in dev mode
#      (LIGHTSEED_NO_TAILSCALE=1), which skips Tailscale AND oauth2-proxy;
#   4. asserts header stripping/injection, streaming, the offline page and
#      /healthz;
#   5. stops the fake upstream and asserts the offline page comes back.
#
# Dev mode is the ONLY mode where sign-in is bypassed, and entrypoint.sh
# refuses dev mode if oauth2 credentials are present -- so this script cannot
# accidentally test a configuration that ships.
#
# Ports are fixed in the 8770-8779 test range so a stray run is easy to spot
# and cannot collide with the real app on 8765/8766.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE="${IMAGE:-lightseed-railway:smoke}"
readonly APP_PORT=8770
readonly UPSTREAM_PORT=8771
# 8772 is the production-chain listener, 8773 the gateway behind it, 8776/8777
# the fake tailscaled proxies. All inside the 8770-8779 test range.
readonly PROXY_TOKEN="smoke-proxy-token-0123456789abcdef"
readonly DEV_EMAIL="Invited.Person@Example.COM"
# Obviously fake and fixed: a 32-byte base64 cookie secret, the shape the
# entrypoint validates. Never a real secret.
# URL-safe on purpose ("-" and "_"): the form oauth2-proxy accepts.
readonly COOKIE_SECRET="c21va2-_b25seS1jb29raWUtc2VjcmV0LTMyYnl0ZXM="

BUILD=1
if [ "${1:-}" = "--no-build" ]; then
  BUILD=0
fi

CONTAINER_ID=""
UPSTREAM_PID=""
WORK_DIR=""

log() { printf '\n== %s\n' "$*" >&2; }
fail() { printf '\nFAIL: %s\n' "$*" >&2; exit 1; }
pass() { printf 'ok: %s\n' "$*" >&2; }

cleanup() {
  if [ -n "$CONTAINER_ID" ]; then
    "${RUNTIME}" rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
  fi
  if [ -n "$UPSTREAM_PID" ]; then
    kill "$UPSTREAM_PID" >/dev/null 2>&1 || true
    wait "$UPSTREAM_PID" 2>/dev/null || true
  fi
  if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------------ runtime ---

RUNTIME=""
if command -v docker >/dev/null 2>&1; then
  RUNTIME="docker"
elif command -v podman >/dev/null 2>&1; then
  RUNTIME="podman"
else
  cat >&2 <<'EOF'
Neither docker nor podman is installed, so the image cannot be built here.

The rest of the checks are still available without a container:
  node --test test/          # gateway unit/integration tests, loopback only
EOF
  exit 2
fi
log "using $RUNTIME"

WORK_DIR="$(mktemp -d)"
readonly WORK_DIR

# ------------------------------------------------------------ fake upstream ---

cat >"$WORK_DIR/upstream.py" <<'PY'
"""Fake PC: echoes request headers, streams slowly on /stream."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[2]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def _record(self, body=b""):
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "bytes": len(body),
            }) + "\n")

    def _send(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _drain(self):
        length = self.headers.get("content-length")
        if not length:
            return b""
        return self.rfile.read(int(length))

    def do_GET(self):
        self._record()
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for chunk in (b"first-chunk\n", b"second-chunk\n"):
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(0.6)
            self.wfile.write(b"0\r\n\r\n")
            return
        if self.path == "/slow":
            time.sleep(3)
            self._send(200, {"slow": True})
            return
        if self.path == "/very-slow":
            # Longer than the default connect timeout on purpose: the PC is up
            # and merely thinking. If the connect timer survives the connection,
            # this returns the 503 offline page at 10s and the smoke fails.
            time.sleep(12)
            self._send(200, {"slow": True})
            return
        if self.path == "/web-error":
            self._send(502, {
                "schema": "qwen_engine.web_error.v1",
                "code": "generation_failed",
                "message": "the diffusion process exited",
            })
            return
        self._send(200, {"ok": True, "path": self.path})

    def do_POST(self):
        body = self._drain()
        self._record(body)
        self._send(200, {"bytes": len(body)})

    def do_HEAD(self):
        self._record()
        self._send(200, b"")


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PY

start_upstream() {
  rm -f "$WORK_DIR/upstream.jsonl"
  python3 "$WORK_DIR/upstream.py" "$UPSTREAM_PORT" "$WORK_DIR/upstream.jsonl" &
  UPSTREAM_PID=$!
  local waited=0
  while ! curl -fsS -m 1 "http://127.0.0.1:${UPSTREAM_PORT}/ready" \
      >/dev/null 2>&1; do
    waited=$((waited + 1))
    if [ "$waited" -ge 50 ]; then
      fail "fake upstream did not start"
    fi
    sleep 0.1
  done
}

stop_upstream() {
  if [ -n "$UPSTREAM_PID" ]; then
    kill "$UPSTREAM_PID" >/dev/null 2>&1 || true
    wait "$UPSTREAM_PID" 2>/dev/null || true
    UPSTREAM_PID=""
  fi
}

last_upstream_record() {
  tail -n 1 "$WORK_DIR/upstream.jsonl"
}

# ------------------------------------------------------------------- build ---

if [ "$BUILD" -eq 1 ]; then
  log "building image $IMAGE"
  "$RUNTIME" build \
    --build-arg "LIGHTSEED_VERSION=smoke" \
    -t "$IMAGE" "$SCRIPT_DIR"
else
  "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1 ||
    fail "image $IMAGE does not exist; run without --no-build"
fi

# shellcheck disable=SC2016
"$RUNTIME" run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'test -x /usr/local/bin/oauth2-proxy && test -x /usr/local/bin/tailscaled &&
   test -x /usr/local/bin/tailscale && test -f /app/gateway.js &&
   test -f /app/templates/sign_in.html && test -f /app/templates/error.html &&
   test -f /app/templates/robots.txt && test -f /app/pages/offline.html' ||
  fail "image is missing one of oauth2-proxy, tailscaled, tailscale, gateway.js, templates, pages"
pass "image carries both pinned binaries, the gateway and the pages"

# shellcheck disable=SC2016
"$RUNTIME" run --rm --entrypoint /bin/sh "$IMAGE" -c 'id -u' |
  grep -qx 10001 || fail "image does not run as uid 10001"
pass "image runs unprivileged (uid 10001)"

# The Tailscale control socket lives in a directory the unprivileged user must
# own, or tailscaled cannot bind and every boot dies 30s later. The rest of this
# script runs in dev mode, which skips tailscaled, so assert it here.
"$RUNTIME" run --rm --entrypoint /bin/sh "$IMAGE" -c \
  'test -O /tmp/lightseed-ts && test -w /tmp/lightseed-ts' ||
  fail "the Tailscale socket directory /tmp/lightseed-ts is not writable by uid 10001"
pass "tailscaled can bind its socket as the unprivileged user"

# --------------------------------------------------------------------- run ---

start_upstream

log "starting the image in dev mode (no Tailscale, no oauth2-proxy)"
CONTAINER_ID="$("$RUNTIME" run -d --name "lightseed-smoke-$$" \
  --network=host \
  -e "PORT=$APP_PORT" \
  -e "LIGHTSEED_NO_TAILSCALE=1" \
  -e "LIGHTSEED_DEV_MODE=1" \
  -e "LIGHTSEED_DEV_EMAIL=$DEV_EMAIL" \
  -e "LIGHTSEED_DEV_SUB=smoke-sub" \
  -e "LIGHTSEED_UPSTREAM=127.0.0.1:${UPSTREAM_PORT}" \
  -e "LIGHTSEED_PROXY_TOKEN=$PROXY_TOKEN" \
  -e "LIGHTSEED_VERSION=smoke" \
  -e "LIGHTSEED_LOG_REQUESTS=1" \
  "$IMAGE")"

waited=0
until curl -fsS -m 1 "http://127.0.0.1:${APP_PORT}/healthz" >/dev/null 2>&1; do
  waited=$((waited + 1))
  if [ "$waited" -ge 120 ]; then
    "$RUNTIME" logs "$CONTAINER_ID" >&2 || true
    fail "container did not answer /healthz within 60s"
  fi
  sleep 0.5
done
pass "container is serving on :$APP_PORT"

# oauth2-proxy must not be in the dev-mode process tree: this run has no
# sign-in, and the assertions below rely on reaching the gateway directly.
# /proc is used rather than pgrep: procps is not in the slim base image, and a
# missing pgrep would look like a pass.
"$RUNTIME" exec "$CONTAINER_ID" /bin/sh -c '
  for proc in /proc/[0-9]*; do
    if [ "$(cat "$proc/comm" 2>/dev/null)" = "oauth2-proxy" ]; then exit 1; fi
  done
  exit 0' &&
  pass "oauth2-proxy is bypassed in dev mode" ||
  fail "oauth2-proxy is running in dev mode"

"$RUNTIME" exec "$CONTAINER_ID" /bin/sh -c '
  seen=""
  for proc in /proc/[0-9]*; do
    seen="$seen $(cat "$proc/comm" 2>/dev/null)"
  done
  case "$seen" in *node*) ;; *) echo "no node process"; exit 1 ;; esac' &&
  pass "the gateway process is running" ||
  fail "no gateway process in dev mode"

# -------------------------------------------------------- header rewriting ---

log "checking header stripping and injection"
HTTP_CODE="$(curl -sS -o "$WORK_DIR/echo.json" -w '%{http_code}' \
  -H 'X-Lightseed-Proxy-Token: forged' \
  -H 'X-Lightseed-User-Email: attacker@evil.example' \
  -H 'X-Lightseed-User-Sub: 666' \
  -H 'X-Lightseed-Smuggled: yes' \
  -H 'X-Forwarded-Email: attacker@evil.example' \
  -H 'X-Qwen-UI: 1' \
  -H 'Origin: https://lightseed.example' \
  "http://127.0.0.1:${APP_PORT}/echo")"
[ "$HTTP_CODE" = "200" ] || fail "proxied GET returned $HTTP_CODE"

python3 - "$WORK_DIR/upstream.jsonl" "$PROXY_TOKEN" <<'PY'
import json
import sys

path, token = sys.argv[1], sys.argv[2]
items = [json.loads(line) for line in open(path, encoding="utf-8")]
echo = [item for item in items if item["path"] == "/echo"]
assert echo, "the upstream never saw /echo"
headers = echo[-1]["headers"]

def check(condition, message):
    print(("ok: " if condition else "FAIL: ") + message, file=sys.stderr)
    if not condition:
        raise SystemExit(1)

check(headers.get("x-lightseed-proxy-token") == token,
      "proxy token was replaced with the server's own")
check(headers.get("x-lightseed-user-email") == "invited.person@example.com",
      "identity email was injected and lowercased")
check(headers.get("x-lightseed-user-sub") == "smoke-sub",
      "identity sub was injected")
check("x-lightseed-smuggled" not in headers,
      "a client-invented X-Lightseed-* header was stripped")
check("x-forwarded-email" not in headers,
      "a client X-Forwarded-Email was stripped")
check(headers.get("x-qwen-ui") == "1", "X-Qwen-UI passed through")
check(headers.get("origin") == "https://lightseed.example",
      "Origin passed through unchanged")
PY

# -------------------------------------------------------------- streaming ---

log "checking streaming (chunk 1 must arrive while the upstream is still open)"
python3 - "$APP_PORT" <<'PY'
import sys
import time
import urllib.request

port = sys.argv[1]
started = time.monotonic()
first = None
last = None
body = b""
with urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=30) as res:
    while True:
        chunk = res.read(1)
        if not chunk:
            break
        if first is None:
            first = time.monotonic() - started
        last = time.monotonic() - started
        body += chunk

ok = True
if body != b"first-chunk\nsecond-chunk\n":
    print(f"FAIL: streamed body was {body!r}", file=sys.stderr)
    ok = False
elif first is None or first > 0.5:
    print(f"FAIL: first byte after {first}s -- the proxy buffered", file=sys.stderr)
    ok = False
elif (last - first) < 0.4:
    print(f"FAIL: both chunks arrived together ({last - first}s apart)", file=sys.stderr)
    ok = False
if ok:
    print(f"ok: streamed incrementally (first byte {first:.2f}s, "
          f"total {last:.2f}s)", file=sys.stderr)
    raise SystemExit(0)
raise SystemExit(1)
PY

log "checking a 40 MiB upload passes without a body limit"
python3 - "$APP_PORT" <<'PY'
import json
import sys
import urllib.request

port = sys.argv[1]
payload = b"x" * (40 * 1024 * 1024)
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/api/v1/references",
    data=payload,
    method="POST",
    headers={"content-type": "application/octet-stream"},
)
with urllib.request.urlopen(request, timeout=120) as res:
    body = json.loads(res.read())
ok = res.status == 200 and body.get("bytes") == len(payload)
print(("ok: " if ok else "FAIL: ") +
      f"40 MiB upload arrived as {body.get('bytes')} bytes", file=sys.stderr)
raise SystemExit(0 if ok else 1)
PY

log "checking that a slow generation-length request is not cut off"
SLOW_CODE="$(curl -sS -o "$WORK_DIR/slow.json" -w '%{http_code}' -m 60 \
  "http://127.0.0.1:${APP_PORT}/slow")"
[ "$SLOW_CODE" = "200" ] || fail "3s upstream reply became $SLOW_CODE"
pass "a request holding the socket for 3s still succeeded"

log "checking that a PC slower than the connect timeout is not called offline"
VERY_SLOW_CODE="$(curl -sS -o "$WORK_DIR/very-slow.json" -w '%{http_code}' -m 90 \
  "http://127.0.0.1:${APP_PORT}/very-slow")"
[ "$VERY_SLOW_CODE" = "200" ] ||
  fail "a 12s upstream reply (longer than the 10s connect timeout) became $VERY_SLOW_CODE"
python3 - "$WORK_DIR/very-slow.json" <<'PY'
import json
import sys

with open(sys.argv[1], "rb") as handle:
    body = json.load(handle)
if body.get("slow") is not True:
    raise SystemExit("expected the upstream's own body, got %r" % (body,))
PY
pass "a 12s upstream reply is served, not replaced by the offline page"

# ------------------------------------------------------------- healthcheck ---

log "checking /healthz"
curl -fsS "http://127.0.0.1:${APP_PORT}/healthz" -o "$WORK_DIR/health.json" ||
  fail "/healthz did not answer 200 while the upstream was up"
python3 - "$WORK_DIR/health.json" "$UPSTREAM_PORT" "$APP_PORT" <<'PY'
import json
import sys

path, upstream_port, app_port = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(path, encoding="utf-8").read()
body = json.loads(raw)
ok = True
if body.get("status") != "ok":
    print(f"FAIL: health status {body.get('status')!r}", file=sys.stderr)
    ok = False
if body.get("upstream") != "reachable":
    print(f"FAIL: health upstream {body.get('upstream')!r}", file=sys.stderr)
    ok = False
for leaked in ("127.0.0.1", upstream_port, app_port, "100."):
    if leaked in raw:
        print(f"FAIL: health body leaks {leaked!r}", file=sys.stderr)
        ok = False
print("ok: /healthz reports ok/reachable without addresses", file=sys.stderr)
raise SystemExit(0 if ok else 1)
PY

# ------------------------------------------------------- real web error 502 ---

log "checking that a real upstream web_error.v1 passes through"
WEBERR_CODE="$(curl -sS -o "$WORK_DIR/weberr.json" -w '%{http_code}' \
  "http://127.0.0.1:${APP_PORT}/web-error")"
[ "$WEBERR_CODE" = "502" ] || fail "upstream 502 web error became $WEBERR_CODE"
python3 - "$WORK_DIR/weberr.json" <<'PY'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
ok = body.get("schema") == "qwen_engine.web_error.v1" and \
    body.get("code") == "generation_failed"
print(("ok: " if ok else "FAIL: ") +
      "the upstream's own 502 JSON was not replaced", file=sys.stderr)
raise SystemExit(0 if ok else 1)
PY

# ------------------------------------------------------------ offline page ---

log "stopping the fake PC and checking the offline page"
stop_upstream
# /healthz caches its snapshot for 5s on purpose, so a burst of probes cannot
# hammer the PC. Wait past that window, or this asserts against the pre-shutdown
# "ok" snapshot and fails for the right answer arriving a moment late.
sleep 6

OFFLINE_CODE="$(curl -sS -o "$WORK_DIR/offline.html" -w '%{http_code}' \
  "http://127.0.0.1:${APP_PORT}/api/v1/status")"
[ "$OFFLINE_CODE" = "503" ] || fail "offline reply was $OFFLINE_CODE, not 503"
grep -q "Lightseed Studio is offline" "$WORK_DIR/offline.html" ||
  fail "offline page has no headline"
grep -q "switched off" "$WORK_DIR/offline.html" ||
  fail "offline page does not say the server is switched off"
grep -q 'aria-label="Lightseed"' "$WORK_DIR/offline.html" ||
  fail "offline page has no brand logo"
grep -q "<code>smoke</code>" "$WORK_DIR/offline.html" ||
  fail "offline page has no version stamp"
if grep -Eq "ECONNREFUSED|socket hang up|AggregateError|proxy error" \
    "$WORK_DIR/offline.html"; then
  fail "offline page leaked a raw proxy error"
fi
pass "PC off: branded 503 page, no raw error"

HEALTH_CODE="$(curl -sS -o "$WORK_DIR/health-offline.json" -w '%{http_code}' \
  "http://127.0.0.1:${APP_PORT}/healthz")"
[ "$HEALTH_CODE" = "200" ] ||
  fail "/healthz returned $HEALTH_CODE with the PC off (Railway would restart-loop)"
python3 - "$WORK_DIR/health-offline.json" <<'PY'
import json
import sys

body = json.load(open(sys.argv[1], encoding="utf-8"))
ok = body.get("status") == "degraded" and body.get("upstream") == "unreachable"
print(("ok: " if ok else "FAIL: ") + f"/healthz says {body.get('status')!r}",
      file=sys.stderr)
raise SystemExit(0 if ok else 1)
PY

# ------------------------------------------------------- production chain ---

# Everything above proves the gateway. This proves the layer that actually sits
# in front of it on Railway: the real entrypoint's oauth2-proxy, started from the
# same oauth2-proxy.args, with the gateway behind it. Tailscale is replaced by a
# fake control socket that reports Running, so no tailnet and no credentials are
# needed -- but the sign-in boundary is genuine.
log "checking the production chain: oauth2-proxy in front of the gateway"

"$RUNTIME" stop -t 15 "$CONTAINER_ID" >/dev/null 2>&1 || true
"$RUNTIME" rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
CONTAINER_ID=""

PROD_PORT=8772
# The real entrypoint, unmodified, with tailscaled/tailscale replaced by the
# fakes in test/fake-tailscale (PATH only; the image is untouched). Everything
# else -- oauth2-proxy, its flags, the gateway, the port split -- is what ships.
CONTAINER_ID="$("$RUNTIME" run -d --name "lightseed-prodchain-$$" \
  --network=host \
  -e "PATH=/opt/fake-tailscale:/usr/local/bin:/usr/bin:/bin" \
  -e "PORT=$PROD_PORT" \
  -e "LIGHTSEED_GATEWAY_PORT=8773" \
  -e "LIGHTSEED_UPSTREAM=127.0.0.1:${UPSTREAM_PORT}" \
  -e "LIGHTSEED_PROXY_TOKEN=$PROXY_TOKEN" \
  -e "LIGHTSEED_PUBLIC_URL=https://lightseed.example" \
  -e "LIGHTSEED_VERSION=smoke" \
  -e "TS_AUTHKEY=fake-smoke-test-authkey" \
  -e "LIGHTSEED_TAILSCALE_PROXY=127.0.0.1:8776" \
  -e "LIGHTSEED_TAILSCALE_SOCKS5_PROXY=127.0.0.1:8777" \
  -e "OAUTH2_PROXY_CLIENT_ID=fake-client-id" \
  -e "OAUTH2_PROXY_CLIENT_SECRET=fake-client-secret" \
  -e "OAUTH2_PROXY_COOKIE_SECRET=$COOKIE_SECRET" \
  -v "$SCRIPT_DIR/test/fake-tailscale:/opt/fake-tailscale:ro" \
  "$IMAGE")"

waited=0
until curl -fsS -m 1 "http://127.0.0.1:${PROD_PORT}/healthz" >/dev/null 2>&1; do
  waited=$((waited + 1))
  if [ "$waited" -ge 60 ]; then
    "$RUNTIME" logs "$CONTAINER_ID" >&2 || true
    fail "the production chain did not answer /healthz within 30s"
  fi
  sleep 0.5
done

python3 - "$PROD_PORT" <<'PY'
import json
import sys
import urllib.error
import urllib.request

port = sys.argv[1]
base = f"http://127.0.0.1:{port}"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surfaces 3xx to the caller instead of chasing it.

    Chasing matters: /oauth2/start redirects to Google, and following that would
    report Google's 200 as if the redirect never happened.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


opener = urllib.request.build_opener(NoRedirect)


def get(path, headers=None):
    request = urllib.request.Request(base + path, headers=headers or {})
    try:
        with opener.open(request, timeout=15) as res:
            return res.status, res.headers, res.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read().decode("utf-8", "replace")


failures = []


def check(condition, message):
    print(("ok: " if condition else "FAIL: ") + message, file=sys.stderr)
    if not condition:
        failures.append(message)


# 1. An anonymous browser gets the branded sign-in page, not the app.
status, _, body = get("/", {"accept": "text/html"})
check(status == 403, f"anonymous / is the sign-in page (403, got {status})")
check("Sign in with Google" in body, "the sign-in page has the Google button")
check("What signing in stores" in body,
      "the sign-in page carries the privacy notice")
check('action="/oauth2/start"' in body, "the button posts to /oauth2/start")
check("<svg" in body and "viewBox" in body, "the brand logo is inlined")
check("oauth2-proxy" not in body.lower(), "no proxy internals leak into the page")

# 2. The sign-in redirect really goes to Google with email-only scope, for a
#    browser Accept header too: our page is the sign-in page, so the button
#    must leave in one click rather than re-render another button.
status, headers, _ = get("/oauth2/start?rd=/", {"accept": "text/html"})
location = headers.get("location", "")
check(status == 302, f"/oauth2/start redirects (302, got {status})")
check(location.startswith("https://accounts.google.com/"),
      "the redirect target is Google")
check("scope=openid+email" in location or "scope=openid%20email" in location,
      "the OAuth scope is openid email only")
check("redirect_uri=https%3A%2F%2Flightseed.example%2Foauth2%2Fcallback"
      in location, "the redirect URI is the public callback")

# 3. A client cannot smuggle its own identity: the gateway must still refuse.
status, _, body = get("/api/v1/status", {"x-forwarded-email": "attacker@evil.example"})
check(status != 200, f"spoofed X-Forwarded-Email does not authenticate (got {status})")
check("attacker@evil.example" not in body, "the spoofed address is not echoed")

# 4. oauth2-proxy itself is alive, so the checks above were not passing against
#    a dead proxy.
status, _, _ = get("/oauth2/ping")
check(status == 200, f"oauth2-proxy answers /oauth2/ping (200, got {status})")

# 5. The gateway is bound to loopback, so nothing outside the container can skip
#    oauth2-proxy by talking to it directly.
status, _, body = get("/api/v1/status")
check(status == 403,
      f"an anonymous API call is refused by sign-in (403, got {status})")

if failures:
    raise SystemExit(1)
PY

"$RUNTIME" logs "$CONTAINER_ID" >"$WORK_DIR/prodchain.log" 2>&1 || true
grep -q "OAuthProxy configured for Google" "$WORK_DIR/prodchain.log" ||
  fail "oauth2-proxy did not report a Google configuration"
if grep -q "cover\|unknown flag\|invalid value" "$WORK_DIR/prodchain.log"; then
  fail "oauth2-proxy rejected its arguments"
fi
pass "oauth2-proxy accepted the shipped flags and fronts the gateway"

# ------------------------------------------------------------- graceful stop ---

log "checking SIGTERM shutdown"
"$RUNTIME" stop -t 15 "$CONTAINER_ID" >/dev/null 2>&1 || true
# Read the logs once into a file: piping into `grep -q` under pipefail fails
# spuriously when grep exits early and the runtime gets SIGPIPE.
"$RUNTIME" logs "$CONTAINER_ID" >"$WORK_DIR/container.log" 2>&1 || true
grep -q "listening" "$WORK_DIR/container.log" ||
  fail "no gateway startup log"
grep -q "shutting down" "$WORK_DIR/container.log" ||
  fail "the gateway did not log a graceful shutdown"
if grep -q "FATAL" "$WORK_DIR/container.log"; then
  fail "the container logged a fatal error"
fi
pass "SIGTERM stopped the stack cleanly"

"$RUNTIME" rm -f "$CONTAINER_ID" >/dev/null 2>&1 || true
CONTAINER_ID=""

log "smoke test passed"
