# Host startup: LAN and Tailscale

2026-10-24: user requested both bindings and an executable startup script.

Run from the project directory:

```bash
./start-web.sh
```

The script selects the project virtual environment and starts real generation
on up to three explicit IPv4 addresses, sharing one serial job queue:

- Localhost: http://127.0.0.1:8765, always.
- LAN: the private IPv4 address of the default route (`ip -4 route get`), if
  the machine has one. On the development PC, http://192.168.68.103:8765.
- Tailscale: `tailscale ip -4`, if Tailscale is up. On the development PC,
  http://100.98.64.62:8765.

The script also works when invoked by absolute path from another directory.
Stop it with Ctrl+C. It runs in the foreground; no login/boot autostart or
systemd unit is installed. It uses `output/web` and `cache` beneath the project.
It applies the previously used direct-host memory attestation; set
`QWEN_ATTEST_UNBOUNDED_HOST=0` in a memory-limited environment.

Optional overrides:

```bash
QWEN_WEB_PORT=8766 ./start-web.sh
QWEN_LAN_HOST=192.168.68.104 ./start-web.sh
```

`QWEN_TAILSCALE_HOST` can override the Tailscale IP; set either variable to
`""` to leave that network out. Binding any requested address must succeed,
otherwise startup closes earlier listeners and refuses, which is why only
addresses found on this machine are passed by default. There is
no wildcard fallback. The executable bit is tracked in git.

The underlying server now accepts repeated `--host` options for explicit IPv4
loopback, RFC1918 LAN and Tailscale/CGNAT literals. IPv6, public IPs, hostnames
and wildcard binds remain refused. Host must match the particular listener IP
and actual port; `localhost` is an alias only on the 127.0.0.1 listener. Origin
must match that exact origin, including its port.

Browsers may also use this machine's own names, listed in
`cache/web-settings.json` and accepted on every listener:

```json
{"host_names": ["tiktak-ai", "tiktak-ai.local", "tiktak-ai.tail3e3fc8.ts.net"]}
```

The first `./start-web.sh` writes that file with the hostname,
`<hostname>.local` (mDNS) and the Tailscale MagicDNS name, so
`http://tiktak-ai.local:8765/` works. Edit it and restart to change the list;
`[]` accepts IP addresses only. A file that is not exactly that shape stops the
server at startup. `qwen-web --host-name NAME` adds names for one run. Names
are matched as text and never resolved; any other name is still refused, which
is what stops a DNS-rebinding page from reaching the server under its own
domain. Existing JSON, X-Qwen-UI,
body limits, response headers, image verification and path protections remain.
All listeners use the same worker and cancel/close it once at shutdown.

The app has no login layer. This is the requested exposure to peers able to
reach these LAN/Tailscale addresses. No firewall, Tailscale ACL, router, or
unrelated service configuration was changed.

Verified on this host: interface addresses, three actual port8765 listeners,
generation-enabled status and same-origin planning on each, and rejection of
foreign Host and Origin. This verifies local connections to those addresses;
an independent remote-client test was not performed. Bind-policy, partial-startup
cleanup and shared-worker/shutdown regressions are in `tests/test_web_bind.py`.
Real Chrome also recognized the generation-enabled workspace through both LAN
and Tailscale IPs on this host. No additional GPU generation was needed for the
bind change; the previously verified W3 engine path is unchanged.

Restart testing also fixed an existing server-construction issue: worker state
is initialized before the socket constructor, so a failed bind reports its
original error and closes cleanly. SO_REUSEADDR now permits clean restarts while
old connections are in TIME_WAIT; SO_REUSEPORT is not enabled.

Final validation: guarded CPU suite **2567 passed, 2 skipped** (23.17 s),
HTTP/real-Chrome planning regression **105/105 passed**. A separate live-socket
check confirmed a second server gets a clean bind error while the first runs,
then succeeds immediately on the same port after the first stops, even after
serving a request.

# Remote listener (C1): invited users through the Railway proxy

2026-10-27: added by D2. The three LAN/Tailscale listeners above remain the
**local** port family: admin, Settings, model downloads and "delete all
history" live there and nowhere else.

## Enabling it

`start-web.sh` adds the four `--remote-*` flags when **all** of these hold:

1. `${QWEN_REMOTE:-1}` is `1`;
2. `QWEN_REMOTE_ORIGIN` is set (`https://<railway-domain>`, no trailing slash);
3. the token file exists — `$QWEN_REMOTE_TOKEN_FILE`, else `cache/remote-token`;
4. a Tailscale IPv4 address is found (`tailscale ip -4`, overridable with
   `QWEN_REMOTE_HOST`).

Otherwise the listener is simply off, and the launcher says so when it wanted
to start it and could not (`notice: no Tailscale IPv4 address found`).

```bash
install -m 600 /dev/null cache/remote-token
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > cache/remote-token
chmod 600 cache/remote-token
QWEN_REMOTE_ORIGIN=https://studio.example.com ./start-web.sh
QWEN_REMOTE_PORT=8766   # first port tried; 0 asks the kernel
```

The server refuses to start (exit 2, before any bind) when the four flags are
not given as a set, when the origin is not a bare `http(s)` origin, when the
host is not an explicit IPv4 loopback/RFC1918/Tailscale literal, when the token
file is missing, a symlink, not a regular file, readable by group/other, or
shorter than 16 bytes, and when `--cache-dir` is absent (the allowlist and the
token live there). `--remote-port 0` is a legal ephemeral bind and counts as
*given*.

## What the remote listener refuses

| Condition | Answer |
|---|---|
| no/duplicate/wrong proxy token | `401 unauthorized`, flat body, no detail |
| malformed Host | `400 invalid_host` |
| POST without `X-Qwen-UI: 1` | `403 ui_header_required` |
| POST whose `Origin` is not the configured one | `403 foreign_origin_denied` |
| `Expect:` on a POST | `417` |
| malformed, unknown or duplicate identity | `401 unauthorized` |
| a valid account that is not invited (or whose bound id differs) | `403 not_invited` |
| a revoked feature | `403 feature_not_allowed`, naming the feature and the permission to ask for |
| someone else's job/image/reference id | `404` — byte-identical to an unknown target |
| `/api/v1/settings/*`, `/api/v1/models*`, `/api/v1/jobs/forget-all` | `404 not_found` |
| any `HEAD` | `405 method_not_allowed` + `Allow: GET` |

The token is checked **first**, so a caller without the secret cannot even
learn which routes exist here. The Host header is only shape-checked on this
listener (the browser's Host is the Railway domain, the socket is the tunnel);
the token and the Origin check are what identify the caller.

## Owner isolation (C3)

The owner of a row is the provider account id (`sub`), never the email: an
address can be recycled, an account id cannot. Consequences:

- history is per owner, capped at 5 finished jobs for a remote user; the 6th
  success reaps the oldest (its directory deleted through the controller's
  own FD-anchored safe-delete, never outside the root, never following a
  symlink) and a job that fails never evicts an older success;
- at most one live (pending/running) job per remote owner — a second submit is
  `409 queue_limit`;
- the reference store stays content-addressed and shared (identical bytes are
  one blob), but a per-owner index decides what a person may see and reference;
  removing a user drops their index entries and deletes only blobs no owner
  references any more;
- a remote draft is pinned to the local LLM rewriter unless the account has
  `official_enhancer`; an unconfigured local rewriter is `503
  rewrite_unavailable` rather than a silent fallback.

The local port is unchanged: an owner id there is always `local`, and spurious
identity headers sent to it are ignored. A `job.json` written before ownership
existed has no `owner` field; it loads as `local`, so upgrading does not empty
History and never hands an old job to a remote account.

## The status document

Additive and closed-schema; `WEB_STATUS_SCHEMA`
(`"qwen_engine.web_status.v1"`) is **not** bumped. Every
listener now sends two extra keys:

```json
{"server_version": "1a2b3c4", "admin": true}
```

`server_version` is the checkout's git short sha read once at startup
(7-40 lowercase hex), or `"unknown"` — never anything else. On the local port
`admin` is `true` and there is no `user` key.

The remote port sends `admin: false` and adds `user`, and nothing else:

```json
{
  "schema": "qwen_engine.web_status.v1",
  "server_version": "1a2b3c4",
  "admin": false,
  "user": {
    "email": "a@example.com",
    "display_name": "Alice",
    "allow": {"q8": true, "2k": false, "edit": false, "official_enhancer": false},
    "history": {"count": 2, "cap": 5, "will_replace_oldest": false}
  }
}
```

`allow` always carries all four flags (a missing key would make the page
guess). `history.count` counts this person's finished jobs; `cap` is 5;
`will_replace_oldest` says whether the next success would evict the oldest one.
`display_name` is always a string: the name the host set for this account
(bounded, printable, trimmed), or `""` when there is none — the page shows the
email in that case rather than inventing a name.

## Allowlist file

`<cache-dir>/users.json`, mode `0600`, rewritten atomically (temp file in the
same directory, `os.replace`) under one lock, so a concurrent reader sees the
old document or the new one and never a partial one. It is a closed document:
a corrupt or foreign file is a typed refusal at startup, not a crash, and an
unknown permission flag or non-boolean value is a typed refusal at write time.
An email is normalised (trimmed, lowercased) the same way by the store, the
identity check and the admin routes, so it cannot be spelled two ways between
them.
