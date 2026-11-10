# Web interface design — local Lightseed Studio (W milestones)

> **Current, 2026-10-24:** W3 generation is implemented. See
> [W3 run instructions, contracts and validation](web-generation-w3.md).
> The W1/W2 record below is historical: its disabled-generation statements,
> three-endpoint limit and `adapter_not_implemented` gate are superseded.

Dated 2026-10-12 (user pivot: the **web UI is the primary interface**, replacing
the planned/deferred desktop GUI; the engine and CLI are kept unchanged).
This document is the narrow design record for W1 **and the now-implemented
W2** (review-pending until root's final acceptance); PLAN.md §W carries the
milestone list. This preserves the historical W1/W2 design and review evidence.

## What FreeVideo is — and is not

The FreeVideo web studio (`FlashML-org/FreeVideo` `web/studio.js`,
`web/view_navigation.js`, `web/library.js`, inspected read-only by root+peer)
was used **only as a visual/UX reference**: header + two columns (left
controls / right stable preview), progressive disclosure, measured stat cards,
and a creations thumbnail history. **No FreeVideo code or assets are vendored;
there is no ComfyUI dependency anywhere in this project** — FreeVideo's
division of labour (ComfyUI owns execution) is *not* copied; our W2 server
(implemented) and the future W3 server talk to our own engine only. Only
T2I-relevant patterns were adapted; video/editor/nodes/LoRA/sharing/batch/
queue/automatic-download/cloud features are out of scope by directive.

## Files and boundary

* `src/qwen_engine/web_static/index.html|app.css|app.js` — original,
  self-contained (no framework, no CDN, no external fonts/analytics). Every
  request the page makes is **same-origin and relative** and goes to one of
  the literal `/api/v1/…` constants at the top of `app.js` — status on load,
  then plan/export/import, jobs, references, rewrite, the rewriter settings,
  the model catalogue/download, and (local listener only) the admin routes.
  Nothing else: no host, no absolute URL, no CDN, no third party. Opening the
  files over `file://` still works: the single status fetch fails and the
  chip honestly says **"Not connected"** — static hosting remains
  disconnected from any engine.
* `src/qwen_engine/web/{api.py,server.py,__main__.py}` — W2 server
  (implemented, review-pending): `api.py` is the **contract module** with no
  socket code and no ML — but NOT filesystem-free: `status` serves
  startup-cached manifest metadata, while `plan`/`export` run the shared
  `GenerationConfig` validator, which **stats the configured runtime-path
  directories** at request time. No response path ever reads model weights,
  streams bytes of an artifact or computes a hash. `server.py` is the stdlib
  transport (loopback bind,
  header policy, exact route allowlist, static serving, typed errors).
  Runnable as `qwen-web` or `python -m qwen_engine.web`; `--help` prints
  without binding, reading or importing ML. Generation is not implemented
  anywhere (W3).

  **Stale gate (2026-10-24)**: the CLI now has a working adapter and produces
  real images, but this web status surface still reports
  `adapter_not_implemented` and the static page still disables Generate with
  "not implemented" copy. That is deliberate in one narrow sense — the web
  process does not import or load the adapter, so it still cannot generate —
  but the *reason* text is now wrong. Fixing it is a coordinated change across
  `web/api.py` (`_gates`/`_blockers`), the closed gate/blocker validator in
  `web_static/app.js` (which rejects any payload whose four gate codes are not
  exactly the current set, so a server-only change breaks the page), the
  handler guard and copy at `app.js:1099+`, and both web test files. Not done
  as part of the V100 generation work.
* `tests/test_web_static.py` — guarded text/structure checks (no browser, no
  network, ML-free; asserts originality-relevant invariants: no external
  origins, no `innerHTML`, BigInt seed handling with a ≤ 20-digit bounded
  grammar, the strict same-origin endpoint allowlist, always-disabled
  Generate, `aria-invalid` wiring, no storage/service APIs, exact
  `config.py` enum values compared against the real Python enums, and the
  W2 envelope/blob/export invariants).
* `tests/test_web_api.py`, `tests/test_web_server_guard.py` — guarded
  contract/policy tests: pure dispatch on bytes (status/plan/export happy
  paths, seed grammar, hostile JSON, leak prevention), the pure transport
  policy functions, `--help`, entrypoint/packaging declarations. No sockets,
  no ML, no writes.
* `scripts/chrome_browser_check.mjs` — **real-Chrome behavioural check** over
  `file://` (dependency-free Node, global WebSocket CDP, `$CHROME_PATH`,
  throwaway profile, own-child process-group cleanup). **Not** part of the
  guarded suite; requires `CHROME_PATH`.
* `scripts/web_integration_check.mjs` — **real-socket integration check**,
  also separate from the guarded suite: spawns its OWN app-server child on an
  ephemeral 127.0.0.1 port with a tiny temp manifest, runs raw-HTTP checks
  (static, traversal/target refusal, Host/Origin/Sec-Fetch/CORS/UI-header/
  length/Expect limits, full 64-bit seed round-trip, byte-exact export,
  hostile JSON) and cleans up only its own process group. The real-Chrome
  served-UI leg is **opt-in** (`QWEN_RUN_CHROME=1` + `CHROME_PATH`).
* **Not in W2 (by design):** job manager, generation endpoints, progress,
  History wiring, anything that touches model weights. Those are W3.

## Visual hierarchy

Full-browser dark charcoal shell (`#16181b`), fine 1px borders, one brand
periwinkle accent (`#7C8CFF`; the cyan/violet gradients live inside the logo).
Header: the Lightseed wordmark image plus the live word **Studio** — the product
name **Lightseed Studio** — with Create/History navigation,
Settings, and a connection chip. Desktop layout is a 380 px controls column
plus a flexible preview column; under ~940 px it stacks with no horizontal
overflow. Visible focus rings, labels for every control, `aria-live` error
text paired with `aria-invalid="true|false"` on every validated control
(prompt, steps, seed, CFG, and **both** dimension inputs for the shared
dimension error), skip link, `aria-current` on the active route,
`aria-pressed` on presets. View routing is hash-driven with `tabindex="-1"`
headings focused **only** on user-driven `hashchange` (never on initial
load); back/forward restore views — an **empty hash routes to Create** (so
Back from History after a hash-less start restores Create), while an unknown
non-empty hash such as the skip link's `#main` keeps the current view
instead of rerouting History/Settings back to Create.

## Honesty model (the load-bearing part)

* W1 had **no server to talk to and made no network requests at all** (the
  speculative `/api/v1/capability` probe was removed in the 2026-10-12 review
  because it enabled Generate on a speculative `ready: true`). W2 replaces
  that with a **strict same-origin contract**: the page connects to
  `GET /api/v1/status` and trusts it only after validating the schema tag,
  the literal booleans (`generation_available` must be exactly `false`, or
  the whole document is rejected), bounded gate/blocker shapes — a response
  it does not fully validate is reported as a typed **"Status rejected"**
  state, distinct from **"Not connected"** (no server / timeout), and never
  as ready. Every fetch carries an `AbortController` timeout; there is no
  polling. Plan and Export start **disabled** and are only offered after a
  recognized status (Check configuration doubles as the reconnect action);
  any disconnect or schema rejection takes them away again. The status can
  only ever say **planning available, generation not available**: every
  generation gate is explicitly unsatisfied (adapter not implemented,
  hardware not probed, assets not verified, calibration **not verified for
  this server** — *unknown is never rendered as absent or verified*), and
  the server's own `generation_available` boolean is displayed, never
  inferred. Generate remains **always disabled** — the W2 plan/export
  endpoints do not generate anything; the submit handlers still enforce that
  directly (`preventDefault` + "Nothing was submitted"). Plan results never
  masquerade as current: the panel is hidden while a new attempt runs,
  out-of-order answers are ignored, a failed attempt says "no planning
  result received; your draft is unchanged; nothing was generated", and a
  draft edited during a wait is labelled as a submitted snapshot. Whether
  opened over `file://` or a static server the page is still disconnected;
  locality is enforced by the `qwen-web` server binding loopback only —
  static hosting enforces nothing.
  Nothing is pretend: no fake ready state, no synthetic results, no fake
  jobs, no invented timings. Stat cards show **em dash / Unknown**, never 0.
  Plan values are rendered **only from an authoritative server answer**, via
  `textContent`, after successful validation; a failed plan/export attempt
  preserves the draft and says so.
* The browser checks (dimensions 64–4096 multiples of 32; steps 1–1000; finite
  CFG ≥ 1; seed grammar) are **lightweight form validation**, labelled as such
  in the UI — explicitly *not* a RAM/model fit check. Authoritative
  `GenerationConfig` validation and runtime paths are owned by the W2
  server (`api.py` + `server.py`), which is implemented (review-pending);
  the frontend parsers additionally refuse any server document they cannot
  fully validate — including a blank/Python-whitespace-only echoed
  `config.prompt`, exactly as shared `GenerationConfig` does. The prompt
  check mirrors `config.py` truthfully
  (`prompt` must be a non-empty trimmed string) and is surfaced only after
  the user edits the prompt or attempts submit — never a first-load alert,
  and nothing is ever auto-filled. Generate itself stays disabled regardless.
* No draft autosave, no storage APIs in W1 (Settings explains). Any future
  draft persistence must be opt-in and store only validated form values.

## Form model

* Prompt first: blank textarea, helpful placeholder, **nothing silently
  generated or auto-filled**; used exactly as typed (`textContent` only — no
  `innerHTML` injection paths anywhere). Like `GenerationConfig` in
  `config.py`, a prompt must not be empty; the lightweight form check says
  so accessibly after editing or submit — not on first load.
* Presets: Square 1024×1024 / Landscape 1344×768 / Portrait 768×1344
  (`aria-pressed`), each driving width/height and the preview's true
  aspect-ratio frame (kept honest by the ratio-aware `max-width` cap).
  Manual dimension edits **recompute the exact preset match**: typing
  1344×768 re-presses Landscape, typing any non-preset size clears every
  `aria-pressed` — no stale preset state.
* Steps default **40**; seed gets its **own full row** (never squeezed beside
  Steps so all 20 digits stay inspectable at the 380 px desktop side panel and
  at 375/320 mobile) with text default **"0"** and an explicit **Random**
  action. The field deliberately has **no `maxlength`**: maxlength would
  silently truncate a pasted 21-digit value into a *different* valid seed;
  `parseSeed` rejects the original text instead and says so inline. Helper
  text is "Leave at 0 or choose Random."; the precise range explanation lives
  in the input's `title` (and here) — not crammed into the visible helper.
* Advanced (collapsible): negative prompt; **CFG default 1.0**; width/height;
  quantization **fp16 + q8/q4 candidates explicitly marked unvalidated**;
  offload **`resident`/`streamed`/`offload`** and attention
  **`auto`/`sdpa`/`efficient`/`eager`** — the exact `OffloadMode` /
  `AttentionBackend` values declared in `qwen_engine/config.py` (no invented
  options; nothing is remapped browser-side); VAE tiling checkbox.
* `QwenStudio.formSnapshot()` keys mirror `GenerationConfig`
  (`negative_prompt`, `true_cfg_scale`, `attention_backend`,
  `vae_tiled_decode`, …); the one deliberate difference is the UI-only
  decimal-**STRING** seed — the snapshot is a UI *display* format. The actual
  wire format is `buildEnvelope()` → `{schema, config}`: same editable keys,
  seed still a decimal string, `kind` deliberately **excluded** (it is UI
  metadata, not config). No aliases, no remapping, no runtime-path fields.
* **Seed semantics:** the full uint64 range `0–18446744073709551615` is kept
  as a **decimal string** and handled with `BigInt` internally — never coerced
  through JS `Number`. Grammar, enforced in `parseSeed` **before** any BigInt:
  trimmed, ASCII decimal digits only, **at most 20 characters — length is
  checked FIRST** — so longer digit strings are rejected on length alone,
  including ones whose leading zeros would make the value small (documented
  policy; the input keeps the user's exact text and explains the rejection —
  nothing is truncated, which is why there is no `maxlength`) — then the
  `[0, 2^64−1]` range.
  Random seeds come from `crypto.getRandomValues` (8
  bytes → exact 64 bits). Null-vs-explicit-random semantics belong to the W2
  API contract, not to invented frontend behaviour.
* `QwenStudio.formSnapshot()` produces the **UI wire format** (seed is a
  STRING) and is *not* the engine's `GenerationConfig` JSON; the authoritative
  serializer is the implemented W2 `POST /api/v1/config/export` (raw canonical
  JSON attachment, exact uint64 seed, runtime paths excluded), which the page
  downloads without reserialization.
* Ctrl/Meta+Enter runs the same unconditional refusal as the submit event:
  validation is shown, then the handler states plainly that nothing was
  submitted — it can never pretend a job exists.

## W2 — server, contract and security model (implemented, review-pending)

### Running it

```
qwen-web                          # or: python -m qwen_engine.web
qwen-web --port 8901 --manifest /path/manifest.json --models-dir /path/models
```

Binds **127.0.0.1:8765** by default (never an arbitrary interface, never port
3081 — that is the review harness, not this product). `--host` accepts only
the literal `127.0.0.1`; `::1` is refused with an explicit message because an
IPv6 loopback bind needs its own actually-tested milestone. The page must be
opened at `http://127.0.0.1:<port>/` — and that URL is only reachable
*inside* the serving container: from any other machine nothing is exposed
(report the fact; the server makes no claim about why a given host is or is
not reachable). `--help` performs no bind, no read and no ML import.

Runtime paths are **server startup flags only** — `--manifest`,
`--models-dir`, `--output-dir`, `--cache-dir`, `--log-dir`,
`--benchmarks-dir`. Omitted flags mean *unconfigured*, never a silent default
write location; each given path is validated through `GenerationConfig` and
canonicalized **once at startup** (bad path → exit 2, before any bind). A
browser JSON body can never set or override any of them: any runtime-path
field appearing in a request body is a typed `unknown_config_field` refusal,
not a strip.

### GET /api/v1/status (`qwen_engine.web_status.v1`)

`planning_available: true`, `generation_available: false`, the four explicit
gates above, their union as `generation_blockers` (plus a `manifest_*` code
when the manifest is missing/invalid), and a **metadata-only** manifest block
(configured / valid / schema_version / artifact_count /
`verification: "not_performed"` / typed reason when not valid). The status
path never calls `resolve_manifest` or `manifest_status` — verifying weights
means stream-hashing gigabytes, which no UI poll may ever do — and never
opens a weight file, loads a processor or imports torch.

### POST /api/v1/plan (`qwen_engine.web_plan_request.v1` → `qwen_engine.web_plan.v1`)

Exact envelope `{schema, config}` — extra or missing envelope keys are
refused. `config` carries **exactly** the editable subset of
`GenerationConfig`: `prompt, negative_prompt, width, height, steps,
true_cfg_scale, seed, quant, offload, attention_backend, vae_tiled_decode`;
anything else (including any runtime path) → typed refusal. The body merges
with the server's startup paths and shares `GenerationConfig.from_dict` —
one validator, not two. **Seed wire format:** a decimal **string** (`null` =
server-chosen later) of 1–20 ASCII digits, parsed to an int and range-checked
to uint64 — never a JSON float, never `Number`, never JS-rounded; the plan
and status responses keep the seed as that decimal string. Internally the
server calls the existing `core.plan` on the merged config, but returns a
**web projection** — the raw core document (which contains runtime paths and
a numeric seed) is never emitted. The projection adds `cfg_active`,
`latent_tokens`, the honest manifest state (`fit: "not_evaluated"`,
`ready: false`, `verification: "not_performed"`) and the unknown hardware /
calibration records.

### POST /api/v1/config/export

Same validated envelope; responds with an attachment
(`qwen-editable-config.json`) whose bytes are the canonical JSON of exactly
the editable fields — runtime paths **omitted** (an exported machine's
folders are not part of a portable config; import will be designed with them
explicitly), and the seed an **exact uint64 JSON number** (Python serializes
it losslessly; browsers would not). The browser downloads the raw attachment
as a `Blob` and never `JSON.parse`/reserializes it — and it builds a download
ONLY from complete, actual `Uint8Array` bytes: a success status whose body
cannot be read bounded, exceeds the cap, or is otherwise unusable is refused
as a typed schema rejection (availability invalidated, distinct rejected
chip) with nothing downloaded and the draft untouched. Import is deliberately
**not** implemented yet.

### Hostile-input policy

Duplicate JSON object keys (at any depth) → `duplicate_json_key`; `NaN` /
`Infinity` literals and overflow floats (`1e999`) → typed refusal; nesting
beyond the shared depth limit → typed; int literals that cannot be parsed →
typed; lone surrogates in any string → `invalid_text`; invalid UTF-8 →
`invalid_utf8`. Every error is `qwen_engine.web_error.v1` with status
400/403/404/405/411/413/414/415/417/422/500/503/505, a short message and
field list — never a
traceback, never a filesystem path, never an internal module name.

### Transport hardening

Literal-loopback bind; **one** `Host` header that must name the bound literal
(or the `localhost` alias) **and the actual bound port** — userinfo,
whitespace, leading-zero ports, over-long digit ports, extra values and any
other name are refused (digit strings are length-bounded **before** any
`int()` conversion, so an arbitrary-length digit header is a typed 400, never
a `ValueError` → 500; no DNS is consulted, which is also why `same-site` is
denied: another port on loopback would be same-site but not same-origin).
Every route — including method refusals (`OPTIONS`/`HEAD`/`PUT`/`DELETE`/
`PATCH`) — passes the Host and Origin/site policy **first**; every method
that then lands on a known route with the wrong method gets a 405 with a
correct `Allow` header and **no CORS headers are ever emitted** (OPTIONS
grants nothing). A foreign `Origin` or a cross-site/cross-origin/same-site
`Sec-Fetch-Site` is denied with 403 on every route. Every POST requires the
exact `X-Qwen-UI: 1` header (a cross-origin form post cannot set it without a
preflight, and no preflight is granted) and an exact `application/json`
content type (optionally `; charset=utf-8`; anything else — form encodings
included — is refused 415 **before the body is read**).
`Content-Length` must appear once as plain ASCII digits with no sign, space or
leading zeros, and is checked against the 65536-byte cap **before any body
byte is read**; `Transfer-Encoding` (incl. chunked) → 400; **every `Expect`
extension, including `100-continue`, is refused 417** — the server never
emits an early 100 before Host/Origin/length validation and never reads an
unsupported Expect body. Finite socket/read timeouts apply before header
parsing; every response closes the connection. **EVERY response** — including
the early request-line/unsupported-version errors the stdlib parser emits
BEFORE any valid version is recognized — carries a real `HTTP/1.x` status
line and the full defensive header set below: the inherited stdlib framing
suppresses ALL header output while `request_version` sits at its `HTTP/0.9`
default, so that legacy/error state is framed explicitly through the SAME
shared header constant (never a second list). Legacy requests are REFUSED,
never served: a request line without an HTTP version, or an explicit
`HTTP/0.x` token, is a typed `unsupported_request_version` 400 raised
BEFORE any dispatch decision (the version is never normalized to make a
legacy request servable); `HTTP/2.x` is a framed 505; malformed HEAD legs
keep the framing contract — headers with an honest `Content-Length`, NEVER
a payload. Standard HTTP/1.0/1.1 handling is unchanged. Routing uses the **raw wire
request target** (Python's own parser folds a leading `//` in the normalized
path — the raw target is checked too, so a real `//x` request is a typed 400,
never accidentally served) and is an exact origin-form allowlist — absolute
targets, query strings, fragments, any percent-encoding, dot segments and
unknown routes are refused (400/404) and **no path is ever translated to a
filesystem path**; there is no directory listing. Requests are served by a
**bounded pool of persistent worker threads** (four workers, small bounded
queue, lazy start, typed 503 when saturated, serial fallback if the box
cannot start a thread at all) — no thread is ever spawned per connection.
Responses carry `Cache-Control: no-store` (API), `nosniff`,
`Referrer-Policy: no-referrer`, CORP `same-origin`, `X-Frame-Options: DENY`
and a CSP of `default-src 'none'; script-src 'self'; style-src 'self';
connect-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none';
form-action 'self'`; JSON serialization uses `allow_nan=False`. Static assets
are three literal routes with fixed MIME types. A configured runtime path
that stops validating **after** startup (e.g. a dir later replaced by a
file) produces the generic `server_runtime_configuration` error — never a
message embedding the private path. Manifest unreadability reasons are a
closed code set (`manifest_unreadable` etc.); arbitrary caller strings never
reach the wire.

## Railway multiuser: remote page and local Admin (contracts C3/C4)

Two listeners, one static page. The **local** listener (loopback, LAN,
Tailscale) is the owner's machine: no `user` block, no sign-out, every
control, and the Admin tab. A **remote** listener (the Railway container,
behind an OAuth proxy) adds a `user` block to `/api/v1/status`; the page then
follows that block and nothing else. Absence of the block is what keeps the
owner's page byte-for-byte what it was.

### The status additions (all optional, all closed)

| key | when | the page's rule |
|---|---|---|
| `admin` | always, from the listener | `true` **only** where the admin routes are actually served; a widened/partial block, or a non-boolean, rejects the whole status |
| `server_version` | always (C5) | a short startup sha, or `unknown`; shown on the offline/error pages |
| `user` | remote listeners with an identity hook | exactly `{email, display_name, allow, history}` |

`user.allow` carries the same four flags the store holds (`q8`, `2k`, `edit`,
`official_enhancer`); `user.history` carries `{count, cap}` and optionally
`will_replace_oldest`. A block that is widened, partial, or mistyped is
**refused whole** — half an identity would hide the wrong controls — and the
page falls back to "not connected" rather than guessing. `parseStatus`
accepts each addition only when its own closed validator passes, so an older
server that never heard of multiuser keeps working unchanged.

### What the remote page hides — courtesy, not enforcement

Hiding is a **courtesy**: the server refuses a disallowed feature with
`feature_not_allowed`, and the page prints that sentence. Nothing is hidden
for correctness.

* 2K presets, the Edit-with-references box and the Q8 option follow their
  flags; hiding a *selected* Q8 moves the draft to `fp16` (a hidden option
  would otherwise still show its text in the closed select), keeping
  references but not using them.
* The official-enhancer choices disappear without `official_enhancer`; a
  remote user's rewriter is otherwise the endpoint.
* `Settings` writes and the model download controls belong to the owner's
  machine and are hidden entirely (their routes would 404).
* The Admin tab is hidden unless `admin: true`.

### The remote page shows

* **Who is signed in** — display name or address — plus one **Sign out** link
  to `/oauth2/sign_out` (the proxy's own route; the page never invents one).
* **The history cap**, before Generate, from the server's own numbers:
  "You have N saved images. Generating another will permanently delete your
  oldest one." It stays visible (no modal) and quotes `cap` rather than
  re-deriving 5.
* **The queue position**: `queue_position` 0 is "Your job is generating now.",
  N > 0 is "Your job is #N in the queue."; an absent value says nothing.
* **One job at a time**: with a job queued or running the Generate button is
  disabled and says "One job at a time", with `queue_limit`'s sentence as the
  reason.
* **Typed refusals as plain sentences**: `not_invited` replaces the whole
  studio ("This account isn't invited — ask the owner", with the proxy-claimed
  address and the same sign-out link); `feature_not_allowed`, `queue_limit`
  and `rewrite_unavailable` each have one fixed sentence. Codes outside that
  closed list keep the bounded generic message — server text is never echoed.

### Admin routes (local listener only)

`GET /api/v1/admin/users`, and `POST` to `/users/add`, `/users/remove`,
`/users/permissions`, `/users/display-name`. Every write is one closed
envelope with the shared CSRF header and content-type rules, answers with the
**whole refreshed document**, and typed errors carry no paths.

| route | envelope | note |
|---|---|---|
| `GET /users` | — | `qwen_engine.web_admin.v1`: `{schema, cap, users[]}` |
| `POST /users/add` | `..._add.v1`: `{schema, email, display_name?}` | no `allow` key: a new account always starts with q8 only |
| `POST /users/remove` | `..._remove.v1`: `{schema, email, delete_data}` | `delete_data` must be a boolean |
| `POST /users/permissions` | `..._permissions.v1`: `{schema, email, allow}` | `allow` is closed over the four flags; a partial block is refused |
| `POST /users/display-name` | `..._display_name.v1`: `{schema, email, display_name}` | trimmed, bounded |

One row per invited account: `email`, `display_name`, `invited`,
`first_sign_in`, `signed_in`, `history` (finished jobs), `active` (queued or
running), `allow`. Usage is counted from the job store on each request —
never cached, so the table cannot drift from History. `first_sign_in` is
`null` when the store holds no stamp (it records an invite time, not a
sign-in time): the page shows the store's own date when there is one and
otherwise "Yes"/"Not yet" from `signed_in`, so it never invents a date.

**Removal order is deliberate.** Data goes first — a store entry removed
before the jobs would leave files with no way to name their owner. The
controller is asked to forget that owner's **finished** jobs (with their
files); a running job is never killed from the admin page. The reference
store is asked through a capability hook (`forget_owner`) because owner-aware
upload cleanup arrives with the server-core work; without the hook its files
stay. Only a ticked "Also delete their images and references" reaches the
delete path, and unticked the account's files stay on disk but leave its
history.

**Why the refusal is a 404.** The remote listener answers the admin routes
with the ordinary `404 not_found`, not a 403: "there is no admin page here"
is the truth a remote user may learn, and it is exactly what a route that was
never published says — nothing about the owner's machine is revealed. Two
independent gates must both pass: the listener carries an admin service at all
(`server.admin`), and *this request* may use it (`server.admin_enabled`, a
bool or a per-request callable; a hook that raises fails **closed**).

### Page behaviour on the admin tab

Every write is disabled while one is in flight; the table is rebuilt only
from a document that passes the same closed validator the server applies, so
a refusal leaves the previous table intact and names the code
(`user_exists: …`). Local typos (an address without an `@`) never reach the
wire. Remove is confirmed **in the page** — never `alert()`/`confirm()` —
with the delete-files tick box unticked by default, focus moved to the
question, and the tick reset after each removal.

### Verification (this worktree, 2026-10-27)

Guarded CPU suite, runner-owned temp root, no sockets and no ML:
`3937 passed, 4 skipped`. New: `tests/test_web_admin.py` (37 tests —
envelope builders, the closed document validator, `AdminService` write paths,
the status additions and the admin flag gates, the admin routes through an
in-memory `_Handler`, a no-paths assertion, plus a Node harness executed
under `node --check` semantics against the real `app.js` functions: table
cells and ARIA, 16 malformed documents refused plus `null`/`[]`, the
first-sign-in column,
status/refusal sentences, and each write's wire body and refreshed table) and
`tests/test_web_multiuser_ui.py` (12 tests — the static contract plus an
executed `applyUserState` matrix over local/remote/denied-flag/cap/queue
states and the closed `parseUserBlock`). Mutation-checked: reverting the
`parseAdminDocument` `INVALID` identity test (the sentinel is an object, so a
truthiness test renders half a table) fails 12 of those tests.

**Real Chrome, real server** (`python -m qwen_engine.web --host 127.0.0.1
--port 8774 --cache-dir <temp>`, i.e. the local owner's listener; throwaway
CDP scripts, not product assets): the Admin tab is offered only after the
status document arrives; clicking it makes `#admin` current, shows the view,
focuses the heading and draws the empty table with the cap line ("0 of 5
accounts invited"). Inviting `  Carol@Example.com ` through the form wrote the
lowered address, an empty name field, "Not yet", zero counts and Q8-only
ticks; ticking 2K rebuilt the table from the server's answer and said "2K
sizes enabled for carol@example.com."; Remove opened the in-page panel with
its question focused and the delete-files tick unticked, and with the tick
set the row went, the panel closed and the tick reset. A typo'd address was
refused in the page with no request. Zero console errors. Reload while the
address ends in `#admin` also lands on the tab. Two behaviours are deliberate
and documented in code: on a **cold** load `#admin` falls back to Create (the
admin flag has not arrived yet, so the tab cannot win the initial route), and
on a remote listener `#admin` is refused exactly like any unknown hash.

## Verification evidence

**2026-10-12 (W1 original, commit `89041f9`)** — real headless Chrome 154:
screenshots at 1440×1000 and 375×812 (`/tmp/w1-*.png`, not product assets);
scripted DOM checks (seed BigInt edges, unique crypto random seeds, snapshot
seed-is-string, validation rejects, preset/aria state, preview ratio, hash
routing, Generate gated only by the then-present capability probe) — all
pass; no console errors; mobile `scrollWidth` ≤ viewport. Guarded CPU suite
780 passed + 1 skipped.

**2026-10-12 (mid-day review fixes — HISTORICAL counts)** — guarded CPU suite
on this container: 786 passed + 1 skipped (default interpreter), 787 passed
0 skipped under the pinned env. Superseded by the final counts below. The
environment note written at that time — "no Python 3.11 binary exists here"
— was **later shown wrong** (root's 16:26/16:27 corrections; see the current
section): the existing ML 3.11 interpreter and a separate review venv are
both Python 3.11.15 and a 3.11 leg HAS since been executed. Kept only as
the historical record of the counts. The three properties stated then are
still accurate: ML-free, socket-free, writes only under a fresh temp root —
*these describe the guarded pytest suite only*: the integration script below
opens real loopback sockets and writes tiny fixtures inside its own temp
root, and the served-Chrome legs run a real browser.

Real-Chrome behaviour (the paragraph below is **historical** — the
process-exhausted container state BEFORE the 2026-10-12 15:35 restart):
`scripts/chrome_browser_check.mjs` is shipped and
runs its own checks (seed grammar incl. length/non-ASCII, snapshot keys,
preset↔manual sync, ratio, nav/focus/`#main`, aria-invalid, always-disabled
with zero HTTP probes, overflow at 1440/375/320), **but it did not pass
green in that container at that fix time**: the sandbox's cgroup pids controller
(512) was saturated by ~390 unreaped zombie processes leaked by earlier
sessions (their parent PID 1 is `sleep` and never reaps), so Chrome failed
`pthread_create` before loading the page. The script detects exactly
this and exits 3 with an honest diagnosis instead of hanging — an
environment problem, not a product failure. The static suite encodes the
same invariants as machine checks.

### Current final verification (2026-10-12, post-restart, final review fixes + cap-branch unification)

* Guarded CPU suites on this container (`bash run_cpu_tests.sh`, runner-owned
  fresh basetemp, network denied, ML imports denied, writes only under a
  fresh temp root — no installs, nothing added to any environment):
  * default interpreter (Python 3.13.5): **879 passed + 1 skipped**
  * pinned review venv (Python 3.13.5, `PYTHON=/tmp/qwen-review.bIFHcI/venv/
    bin/python PYTHONPATH=/work/src QWEN_DIFFUSERS_SRC=/tmp/v100_m0_ml/src/
    diffusers QWEN_TRANSFORMERS_SRC=/tmp/v100_m0_ml/src/transformers`):
    **880 passed**
  * pinned review venv311 (Python **3.11.15**, same pinned env with
    `PYTHON=/tmp/qwen-review.bIFHcI/venv311/bin/python`): **880 passed**
  These are the writer's runs on the current HEAD, which additionally
  unifies the export `too_large` branch (one guarded static regression
  added: 878+1/879 → 879+1/880/880). The earlier "no Python 3.11 binary
  exists here" claim was **wrong and is removed**: root directly verified
  both the existing ML interpreter
  (`/home/dev/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11`,
  Python 3.11.15) and the separate review venv
  (`/tmp/qwen-review.bIFHcI/venv311/bin/python`, Python 3.11.15), and ran
  the full pinned guarded suite with it on `08329c8`: **877 passed in
  8.85 s, no installation** (root's independently verified baseline on that
  commit was 876+1 skip / 877 pinned / 877 with venv311). These
  no-socket/no-repo-write claims cover the guarded suites ONLY — the
  integration check opens real loopback sockets and writes tiny fixtures
  under its own temp root, and the served-UI leg runs a real browser.
* Root's independent probes on `08329c8` (root-run, root-verified):
  `/tmp/v100-web-api-review.py` **7/7 PASS**, `/tmp/v100-web-wire-review.py`
  **5/5 PASS**. CURRENT addendum from root's 16:42 server review: the wire
  script has since been extended to **9 cases** — the original **5 still
  PASS**, but **4 NEW legacy-request-version cases FAIL** (missing version,
  explicit HTTP/0.9, broken legacy request line, HTTP/0.9 parse errors
  return BARE bodies without status line/defensive headers). That is a
  server-side all-responses contract blocker reported by root —
  **unaddressed by this writer** (this fix round is scoped to the frontend
  export cap branch), recorded here so the "5/5" line is not read as
  current for the extended script. Root's independent real-Chrome reviewer session (16:31,
  exited 0): **39/39 assertions PASS** — the extended script (27 DOM/behaviour
  checks at 1440/375/320 px with no exceptions and no external requests,
  valid + blank-plan schema legs ×2, **4 REAL-HTTP status-race checks**,
  **2 explicitly controlled-backend unreadable-export checks**, and **2
  REAL-HTTP 15 s partial-body timeout checks with no exceptions**). The
  original pre-extension script had **35** items — the earlier "33" claim
  was wrong. Root also viewed desktop/mobile screenshots (no horizontal
  overflow, full 20-digit seed visible, honest empty preview / disabled
  Generate). **PENDING, not passed:** root's new REAL oversized-export
  HTTP case (making the set **40**) is deliberately held until the export
  cap-branch unification below lands, and will be run by root
  independently; it has NOT been run or passed yet.
* Real Chrome, `file://` behaviour suite (`scripts/chrome_browser_check.mjs`):
  **61/61 passed** after the restart (pre-final-fixes snapshot; not rerun
  for the cap fix — the served pass below and root's independent runs cover
  the changed export/parse branches).
* Served real-Chrome integration (`scripts/web_integration_check.mjs`,
  sandbox ENABLED, `QWEN_PYTHON` = pinned venv, `QWEN_RUN_CHROME=1`):
  **96/96 executed checks passed** (0 NOT RUN) on the pre-cap-fix HEAD —
  including the two final-review legs: a controlled-backend export response
  without a readable body is refused as a typed schema rejection with no
  download (explicitly a fixture, not real stalled-network proof), and
  `parsePlan` refuses blank/Python-whitespace-only echoed prompts against
  the real served plan echo. Per the review protocol the writer did not
  relaunch Chrome for the cap fix; root's independent 40-check real-Chrome
  run covers the final HEAD.
* Export cap-branch unification (this writer, current HEAD): the early
  `bodyError === 'too_large'` return in `exportConfig` that previously
  refused the over-cap body WITHOUT invalidating availability is removed;
  one unified `if (r.bodyError)` refusal now precedes both the HTTP-error
  path and the raw-bytes download path — every bodyError (`too_large`,
  `unsupported`, any unexpected decoding error) downloads nothing, keeps
  the draft, invalidates the trusted availability and shows the distinct
  'Status rejected' chip (never 'Not connected', never success) with the
  accurate reason. Success still requires actual `Uint8Array` bytes; the
  export bytes are never parsed or reserialized. Guarded static regression
  added (`tests/test_web_static.py::
  test_export_too_large_cannot_bypass_the_unified_refusal`).
* Process hygiene: the two live orphan repro Chrome trees from the previous
  session (PIDs 3284/3410, profiles `/tmp/repro-WoE5LY`/`/tmp/repro-fdL5vs`)
  were closed through their OWN DevTools endpoints (`Browser.close`) after
  pid/profile/process-group validation; their child trees reaped, the roots
  became unreapable zombies at PPID 1 (`sleep` never reaps — platform issue,
  recorded honestly; no broad kill, no PID 1 or service changes). Residual
  chrome-headless zombies remain unkillable by this agent; browser launches
  stayed bounded and usage stayed around 150/512 cgroup slots.

* Legacy/early-error HTTP framing fix (this writer, 2026-10-12, the four
  root-report FAIL cases): the two legacy states that were SERVED bare
  (versionless request line, explicit HTTP/0.9) are now refused with a
  framed typed `unsupported_request_version` 400 BEFORE dispatch, and the
  early errors the stdlib emits before any version is recognized (broken
  request line, HTTP/2.x → framed 505, legacy POST type refusal) are framed
  with the real HTTP/1.1 status line and the full shared defensive header
  set through the one shared frame builder; malformed HEAD legs carry
  honest headers with zero payload. Supported HTTP/1.0/1.1 behavior is
  unchanged. Writer's verification on this HEAD (no Chrome and no model
  launched by this writer): root's 9-case memory-socket probe
  `/tmp/v100-web-wire-review.py` **9/9 PASS on BOTH actual interpreters**
  (Python 3.13.5 review venv and Python 3.11.15 review venv311); guarded
  CPU suites (`run_cpu_tests.sh`, runner-owned fresh basetemp, no
  override): default interpreter **889 passed + 1 skipped**, pinned venv
  (3.13.5) **890 passed**, pinned venv311 (**3.11.15**) **890 passed**
  (10 new guarded memory-socket framing regressions, none weakened);
  integration check HTTP-ONLY (`QWEN_PYTHON` = pinned venv, NO
  `QWEN_RUN_CHROME`) **55/55 executed checks passed** including the six
  new REAL-socket framing legs (four legacy legs + two malformed-HEAD
  no-payload legs), 1 NOT RUN Chrome leg by design. Root's independent
  acceptance on 2026-10-12 reran the full integration command with real
  Chrome 154 and the current Python 3.13.5 venv: **102/102 executed checks
  passed**, including the real-socket legacy framing cases, desktop/mobile
  UI paths, export byte checks, browser race/timeout/rejection cases, no
  third-party requests, and no uncaught page exceptions. The guarded suite
  independently passed **889 + 1 skipped** on that venv. W2 is accepted;
  the earlier "4 NEW legacy cases FAIL" addendum and the pending-review
  wording above are historical snapshots, corrected by this bullet. The
  previous Python 3.11.15 verification remains writer-reported; that
  interpreter was not installed in the root review environment.
