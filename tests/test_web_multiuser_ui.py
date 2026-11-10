"""Remote users on ONE listener (contracts C3/C4): the page's identity rules.

Two halves, in the order the page applies them:

* the STATIC contract — the markup elements exist, the routing guard refuses
  ``#admin`` where the listener does not serve it, and the remote page never
  reaches a browser dialog;
* the EXECUTED contract — ``applyUserState``, ``updateHistoryCapWarning``,
  ``updateGenerate`` and the three refusal renderers are extracted from app.js
  verbatim and driven over the REAL status-document shapes, with the smallest
  DOM they touch. Nothing here reimplements a product rule; every assertion is
  about what a person would see.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "qwen_engine" / "web_static"


def _read(name: str) -> str:
    path = WEB_DIR / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    return path.read_text(encoding="utf-8")


def _js_function(js: str, signature: str) -> str:
    m = re.search(rf"function {re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def _js_var(js: str, name: str) -> str:
    m = re.search(rf"^  var {re.escape(name)} =[\s\S]*?;$", js, re.M)
    assert m, f"{name} not found in app.js"
    return m.group(0)


# ----------------------------------------------------------- static contract


def test_the_identity_and_admin_nav_ship_hidden_and_the_panel_exists():
    html = _read("index.html")
    assert re.search(r'<a href="#admin" id="nav-admin"[^>]*hidden', html), \
        "the Admin tab must start hidden: the status document decides"
    identity = re.search(r'<p class="identity" id="identity"[^>]*>', html).group(0)
    assert "hidden" in identity, identity
    assert 'id="identity-name"' in html
    assert 'id="sign-out"' in html and 'href="/oauth2/sign_out"' in html
    panel = re.search(r'<section class="not-invited" id="not-invited"[^>]*>', html).group(0)
    assert "hidden" in panel, panel
    assert 'id="not-invited-email"' in panel or 'id="not-invited-email"' in html
    heading = re.search(r'<h2 id="not-invited-heading"[^>]*>', html).group(0)
    assert 'tabindex="-1"' in heading, "the panel takes focus when it replaces the page"
    warning = re.search(r'<p class="history-cap-warning" id="history-cap-warning"[^>]*>',
                        html).group(0)
    assert "hidden" in warning and 'role="status"' in warning
    assert 'id="job-queue"' in html and 'id="job-queue"' in html
    queue = re.search(r'<p class="job-queue" id="job-queue"[^>]*>', html).group(0)
    assert "hidden" in queue and "role=\"status\"" in queue


def test_no_browser_dialog_and_the_admin_hash_is_guarded_before_routing():
    js = _read("app.js")
    for call in ("alert(", "confirm("):
        for line in js.splitlines():
            if call in line:
                stripped = line.strip()
                assert (stripped.startswith("//") or stripped.startswith("*")
                        or "never" in stripped), f"live browser dialog call: {line}"
    route = _js_function(js, "routeOnHashchange")
    assert "name === 'admin' && !adminEnabled" in route
    assert "name === 'settings' && currentUser" in route
    # A remote page typing #admin lands on Create, exactly like any unknown
    # hash — never on a table whose every write would 404.
    guard = route.split("name === 'admin' && !adminEnabled")[1]
    assert guard.lstrip().startswith(") {") and "renderView('create')" in guard[:200]
    initial = _js_function(js, "initialRoute")
    assert "name === 'admin'" in initial, "a cold load must never route to Admin"
    view = _js_function(js, "renderView")
    assert "if (!link) return;" in view, "a page without the Admin tab must render"


def test_the_status_parser_accepts_the_three_additions_or_refuses_them_whole():
    js = _read("app.js")
    status = _js_function(js, "parseStatus")
    for key in ("s.admin", "s.server_version", "s.user"):
        assert key in status, key
    assert "typeof s.admin !== 'boolean'" in status
    assert "s.server_version.length > 64" in status
    user = _js_function(js, "parseUserBlock")
    assert "email !== email.toLowerCase()" in user, \
        "the proxy lowercases; another case is a block this page cannot trust"
    assert "USER_HISTORY_OPTIONAL_KEYS" in user
    assert js.count("function parseUserBlock(") == 1


# ------------------------------------------------------- executed: the page

HARNESS_VARS = ("ERROR_SCHEMA", "USER_BLOCK_KEYS", "USER_HISTORY_KEYS",
                "USER_HISTORY_OPTIONAL_KEYS", "PERMISSION_KEYS",
                "TYPED_ERROR_SENTENCES", "SIGN_OUT_HREF", "INVALID",
                "MODELS_GENERATE_REASON", "GENERATION_UNAVAILABLE_REASON",
                "adminEnabled", "currentUser", "serverVersion", "activeView",
                "jobCache", "apiConnected", "generationConnected",
                "submittingJob", "selectedJob")

HARNESS_FNS = ("setText", "parseUserBlock", "typedErrorSentence",
               "showNotInvited", "applyUserState", "hideIf",
               "updateHistoryCapWarning", "jobTerminal", "ownActiveJob",
               "updateGenerate", "renderApiError", "renderRewriteError",
               "renderSelectedJob")

_HARNESS = """// Built by tests/test_web_multiuser_ui.py: the real identity functions, run
// over the real status-document shapes with the smallest DOM they touch.
var fs = require("node:fs");
__PARTS__

var NODES = new Map();
function el(id) {
  return {id: id, textContent: "", value: "", checked: false, hidden: false,
          disabled: false, dataset: {}, attrs: {}, focused: false,
          src: "",
          setAttribute: function (name, value) { this.attrs[name] = value; },
          getAttribute: function (name) {
            return Object.prototype.hasOwnProperty.call(this.attrs, name)
              ? this.attrs[name] : null;
          },
          removeAttribute: function (name) { delete this.attrs[name]; },
          focus: function () { this.focused = true; },
          addEventListener: function () {}};
}
function $(id) {
  if (!NODES.has(id)) NODES.set(id, el(id));
  return NODES.get(id);
}
var WRITE_BLOCKS = [el("rewrite-save-actions"), el("rewrite-test-actions")];
var document = {
  getElementById: function (id) { return NODES.has(id) ? NODES.get(id) : null; },
  querySelectorAll: function (selector) {
    return selector === ".settings-writes" ? WRITE_BLOCKS : [];
  }
};
function applyEditMode() {}
function renderValidation() { validationRan++; }
function validateForm() { return {errors: []}; }
function syncPlanStaleness() { stalenessRan++; }
function renderView(name) { activeView = name; }
// The rewriter's own availability is not under test here: none is configured.
function rewritingSomewhere() { return false; }
// renderSelectedJob's two neighbours on the Create view: each has its own
// section, and neither decides the queue sentence under test.
function updateUseAsReferenceControl() {}
function syncPreviewShape() {}
var validationRan = 0, stalenessRan = 0;
// A state flag whose declaration is followed by a comment block in app.js: set
// here directly so the extractor never swallows the next function.
var modelsMissingBlocked = false;

function reset() {
  NODES.clear();
  WRITE_BLOCKS = [el("rewrite-save-actions"), el("rewrite-test-actions")];
  adminEnabled = false; currentUser = null; serverVersion = null;
  activeView = "create"; jobCache = {}; apiConnected = true;
  generationConnected = true; submittingJob = false;
  modelsMissingBlocked = false; selectedJob = null;
  validationRan = 0; stalenessRan = 0;
  $("quant").value = "fp16";
  $("edit-mode").checked = false;
  $("main").hidden = false;
}
function visible(id) { return $(id).hidden !== true; }
function snapshot() {
  return {
    adminNav: visible("nav-admin"),
    settingsNav: visible("nav-settings"),
    rewriteHint: visible("rewrite-hint"),
    identity: visible("identity"),
    identityName: $("identity-name").textContent,
    signOutHref: $("sign-out").attrs.href || "",
    main: visible("main"),
    notInvited: visible("not-invited"),
    notInvitedEmail: $("not-invited-email").textContent,
    notInvitedFocused: $("not-invited-heading").focused,
    twoK: visible("presets-2k-group"),
    editField: visible("edit-mode-field"),
    editMode: $("edit-mode").checked,
    quant: $("quant").value,
    quantHidden: $("quant-q8").hidden,
    quantDisabled: $("quant-q8").disabled,
    modelControls: visible("models-controls"),
    writeBlocks: WRITE_BLOCKS.map(function (n) { return n.hidden !== true; }),
    t2iPe: visible("rewrite-t2i-pe-option"),
    editPe: visible("rewrite-edit-pe-option"),
    gpuNote: visible("rewrite-gpu-note"),
    capWarning: visible("history-cap-warning"),
    capText: $("history-cap-warning").textContent,
    generateDisabled: $("generate-btn").disabled,
    generateText: $("generate-btn").textContent,
    generateReason: $("generate-reason").textContent,
    activeView: activeView,
    validationRan: validationRan,
    stalenessRan: stalenessRan
  };
}

var out = {};

/* ---- 1. the owner's own page is the page that always existed ---- */
out.local = (function () {
  reset();
  generationConnected = true;
  applyUserState({recognized: true, admin: true, serverVersion: "abc1234",
                  user: null});
  return snapshot();
})();

/* ---- 2. a remote account: identity shown, denied features gone ---- */
out.remote = (function () {
  reset();
  $("edit-mode").checked = true;          // a draft the account may NOT keep
  $("quant").value = "q8";
  applyUserState({recognized: true, admin: false, serverVersion: "abc1234",
                  user: {email: "alice@example.com", displayName: "Alice",
                         allow: {q8: false, "2k": false, edit: false,
                                 official_enhancer: false},
                         history: {count: 2, cap: 5, willReplaceOldest: null}}});
  return snapshot();
})();

/* ---- 3. a remote account that may do everything still has no Admin ---- */
out.remoteAllowed = (function () {
  reset();
  applyUserState({recognized: true, admin: false, serverVersion: "v1",
                  user: {email: "bob@example.com", displayName: "",
                         allow: {q8: true, "2k": true, edit: true,
                                 official_enhancer: true},
                         history: {count: 0, cap: 5, willReplaceOldest: false}}});
  return snapshot();
})();

/* ---- 4. the history cap warning quotes the server's own numbers ---- */
out.capAtCap = (function () {
  reset();
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 5, cap: 5, willReplaceOldest: null}}});
  return snapshot();
})();

out.capBelow = (function () {
  reset();
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 4, cap: 5, willReplaceOldest: null}}});
  return snapshot();
})();

out.capServerSaysNo = (function () {
  // The server's explicit `will_replace_oldest: false` wins over the count
  // comparison, because the server is the one that replaces.
  reset();
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 5, cap: 5, willReplaceOldest: false}}});
  return snapshot();
})();

out.capOne = (function () {
  reset();
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 1, cap: 1, willReplaceOldest: true}}});
  return snapshot();
})();

/* ---- 5. the owner's page never shows the warning or the identity ---- */
out.localClearsRemoteState = (function () {
  reset();
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: false, "2k": false, edit: false, official_enhancer: false},
    history: {count: 5, cap: 5, willReplaceOldest: true}}});
  applyUserState({recognized: true, admin: true, user: null});
  return snapshot();
})();

/* ---- 6. a remote page sitting on Admin is moved off it ---- */
out.remoteLeavesAdmin = (function () {
  reset();
  activeView = "admin";
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 0, cap: 5, willReplaceOldest: false}}});
  return {activeView: activeView, adminNav: visible("nav-admin")};
})();

/* ---- 6b. a remote page sitting on Settings is moved off it ---- */
out.remoteLeavesSettings = (function () {
  reset();
  activeView = "settings";
  applyUserState({recognized: true, admin: false, user: {
    email: "alice@example.com", displayName: "Alice",
    allow: {q8: true, "2k": false, edit: false, official_enhancer: false},
    history: {count: 0, cap: 5, willReplaceOldest: false}}});
  return {activeView: activeView, settingsNav: visible("nav-settings")};
})();

/* ---- 7. not invited: the studio is replaced whole ---- */
out.notInvited = (function () {
  reset();
  applyUserState({recognized: true, admin: false, user: null});
  currentUser = {email: "stranger@example.com", displayName: "",
                 allow: {q8: false, "2k": false, edit: false,
                         official_enhancer: false},
                 history: {count: 0, cap: 5, willReplaceOldest: false}};
  showNotInvited();
  return snapshot();
})();

/* ---- 8. the queue sentence: one job at a time for a remote account ---- */
function job(id, state) {
  return {job_id: id, state: state, phase: null, cancel_requested: false,
          error_code: null, elapsed_ms: null, progress: null, phase_ms: {},
          image_url: null, image_filename: null,
          config: {steps: 20, prompt: "a test prompt", width: 1024,
                   height: 1024, seed: "1", recipe: "base", references: []}};
}

out.queueRemoteBusy = (function () {
  reset();
  var UID = {email: "alice@example.com", displayName: "Alice",
             allow: {q8: true, "2k": false, edit: false,
                     official_enhancer: false},
             history: {count: 0, cap: 5, willReplaceOldest: false}};
  applyUserState({recognized: true, admin: false, user: UID});
  jobCache["a".repeat(32)] = job("a".repeat(32), "running");
  updateGenerate();
  return snapshot();
})();

out.queueLocalBusy = (function () {
  reset();
  applyUserState({recognized: true, admin: true, user: null});
  jobCache["a".repeat(32)] = job("a".repeat(32), "running");
  updateGenerate();
  return snapshot();
})();

out.queueRemoteFinished = (function () {
  reset();
  var UID = {email: "alice@example.com", displayName: "Alice",
             allow: {q8: true, "2k": false, edit: false,
                     official_enhancer: false},
             history: {count: 1, cap: 5, willReplaceOldest: false}};
  applyUserState({recognized: true, admin: false, user: UID});
  jobCache["a".repeat(32)] = job("a".repeat(32), "succeeded");
  updateGenerate();
  return snapshot();
})();

out.queuePosition = (function () {
  reset();
  var id = "b".repeat(32);
  var record = job(id, "queued");
  record.queue_position = 2;
  renderSelectedJob(record);
  var second = job(id, "queued");
  second.queue_position = 0;
  var first = {queue: $("job-queue").textContent, hidden: $("job-queue").hidden};
  renderSelectedJob(second);
  var now = {queue: $("job-queue").textContent, hidden: $("job-queue").hidden};
  renderSelectedJob(job(id, "succeeded"));
  return {waiting: first, now: now,
          finished: {queue: $("job-queue").textContent,
                     hidden: $("job-queue").hidden}};
})();

/* ---- 9. typed refusals: ONE sentence per code, the generic text otherwise ---- */
out.typed = (function () {
  reset();
  var result = {sentences: {}};
  ["not_invited", "feature_not_allowed", "queue_limit",
   "rewrite_unavailable"].forEach(function (code) {
    result.sentences[code] = typedErrorSentence(code);
  });
  result.unknown = typedErrorSentence("out_of_gpu_memory");
  result.notString = typedErrorSentence(null);
  result.inherited = typedErrorSentence("constructor");

  renderApiError({status: 403}, {schema: ERROR_SCHEMA, code: "queue_limit",
                                 message: "server wording"});
  result.api = $("api-message").textContent;
  renderApiError({status: 422},
                 {schema: ERROR_SCHEMA, code: "invalid_envelope",
                  message: "seed is out of range",
                  fields: [{field: "seed", message: "must be a decimal string"}]});
  result.apiUnknown = $("api-message").textContent;
  renderApiError({status: 500}, {schema: "not-ours", code: "queue_limit"});
  result.apiForeign = $("api-message").textContent;

  renderRewriteError(403, {schema: ERROR_SCHEMA, code: "rewrite_unavailable",
                           message: "no rewriter"});
  result.rewrite = $("rewrite-error").textContent;
  renderRewriteError(403, {schema: ERROR_SCHEMA, code: "feature_not_allowed",
                           message: "not allowed"});
  result.rewriteTyped = $("rewrite-error").textContent;
  return result;
})();

/* ---- 10. the user block parser is closed ---- */
out.userBlock = (function () {
  var good = {email: "alice@example.com", display_name: "Alice",
              allow: {q8: true, "2k": false, edit: false,
                      official_enhancer: false},
              history: {count: 3, cap: 5, will_replace_oldest: true}};
  var result = {good: parseUserBlock(good)};
  var cases = {};
  function rejected(name, mutate) {
    var copy = JSON.parse(JSON.stringify(good));
    mutate(copy);
    cases[name] = parseUserBlock(copy) === INVALID;
  }
  rejected("upperCaseEmail", function (v) { v.email = "Alice@example.com"; });
  rejected("noAtSign", function (v) { v.email = "alice"; });
  rejected("spaceInEmail", function (v) { v.email = "alice @example.com"; });
  rejected("extraKey", function (v) { v.sub = "123"; });
  rejected("missingKey", function (v) { delete v.display_name; });
  rejected("stringFlag", function (v) { v.allow["2k"] = "false"; });
  rejected("partialAllow", function (v) { delete v.allow.edit; });
  rejected("countAboveCap", function (v) { v.history.count = 6; });
  rejected("capAboveHundred", function (v) { v.history.cap = 101; });
  rejected("nonIntegerCount", function (v) { v.history.count = 1.5; });
  rejected("badReplaces", function (v) { v.history.will_replace_oldest = "yes"; });
  rejected("historyExtraKey", function (v) { v.history.floor = 0; });
  rejected("longName", function (v) { v.display_name = "n".repeat(81); });
  result.rejected = cases;
  var noOptional = JSON.parse(JSON.stringify(good));
  delete noOptional.history.will_replace_oldest;
  result.optionalAbsent = parseUserBlock(noOptional);
  result.nullBlock = parseUserBlock(null) === INVALID;
  result.arrayBlock = parseUserBlock([]) === INVALID;
  return result;
})();

process.stdout.write(JSON.stringify(out));
"""


def _run_multiuser_harness(tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in HARNESS_VARS]
                      + [_js_function(js, fn) for fn in HARNESS_FNS])
    harness = tmp_path / "multiuser_harness.js"
    harness.write_text(_HARNESS.replace("__PARTS__", parts), encoding="utf-8")
    run = subprocess.run([node, str(harness)], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_owners_page_is_unchanged_by_the_remote_additions(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    local = out["local"]
    assert local["adminNav"] is True             # the owner's own listener
    assert local["settingsNav"] is True and local["rewriteHint"] is True
    assert local["identity"] is False            # no account to name
    assert local["twoK"] is True
    assert local["editField"] is True
    assert local["quantHidden"] is False and local["quantDisabled"] is False
    assert local["modelControls"] is True
    assert local["writeBlocks"] == [True, True]
    assert local["t2iPe"] is True and local["gpuNote"] is True
    assert local["capWarning"] is False          # the owner has no cap
    assert local["generateDisabled"] is False
    assert local["generateText"] == "Generate"


def test_a_remote_account_sees_only_what_its_four_flags_allow(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    remote = out["remote"]
    assert remote["adminNav"] is False           # never a tab whose writes 404
    # The server chooses a remote account's rewriter: no Settings tab, and no
    # hint sending the person there.
    assert remote["settingsNav"] is False and remote["rewriteHint"] is False
    assert remote["identity"] is True
    assert remote["identityName"] == "Alice"
    assert remote["signOutHref"] == "/oauth2/sign_out"
    assert remote["twoK"] is False               # 2k not granted
    assert remote["editField"] is False and remote["editMode"] is False
    assert remote["quantHidden"] is True and remote["quantDisabled"] is True
    assert remote["quant"] == "fp16", "a hidden q8 must not stay selected"
    assert remote["validationRan"] == 1 and remote["stalenessRan"] == 1
    assert remote["modelControls"] is False
    assert remote["writeBlocks"] == [False, False]
    assert remote["t2iPe"] is False and remote["editPe"] is False
    assert remote["gpuNote"] is False

    allowed = out["remoteAllowed"]
    assert allowed["twoK"] is True and allowed["editField"] is True
    assert allowed["quantHidden"] is False
    assert allowed["modelControls"] is False     # a model download is never remote
    assert allowed["writeBlocks"] == [False, False]
    assert allowed["adminNav"] is False


def test_the_page_converges_when_the_identity_changes(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    local = out["localClearsRemoteState"]
    assert local["identity"] is False
    assert local["capWarning"] is False
    assert local["quantHidden"] is False and local["quant"] == "fp16"
    assert local["twoK"] is True and local["editField"] is True
    assert local["writeBlocks"] == [True, True]
    assert local["adminNav"] is True
    moved = out["remoteLeavesAdmin"]
    assert moved["activeView"] == "create"
    assert moved["adminNav"] is False


def test_a_remote_page_sitting_on_settings_is_moved_to_create(tmp_path):
    moved = _run_multiuser_harness(tmp_path)["remoteLeavesSettings"]
    assert moved == {"activeView": "create", "settingsNav": False}


def test_the_hidden_attribute_wins_over_component_display_rules():
    # Without this rule the remote page's hidden Settings blocks (display:
    # flex) stayed on screen although the script had hidden them.
    css = _read("app.css")
    assert "[hidden] { display: none !important; }" in css


def test_the_history_cap_warning_quotes_the_servers_numbers(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    at_cap = out["capAtCap"]
    assert at_cap["capWarning"] is True
    assert at_cap["capText"] == ("You have 5 saved images. Generating another "
                                "will permanently delete your oldest one.")
    assert out["capBelow"]["capWarning"] is False
    assert out["capBelow"]["capText"] == ""
    assert out["capServerSaysNo"]["capWarning"] is False
    one = out["capOne"]
    assert one["capWarning"] is True
    assert "1 saved image." in one["capText"]          # singular, not "images"


def test_an_uninvited_account_gets_one_panel_and_no_studio(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    denied = out["notInvited"]
    assert denied["main"] is False
    assert denied["notInvited"] is True
    assert denied["notInvitedEmail"] == "stranger@example.com"
    assert denied["notInvitedFocused"] is True, "focus moves with the panel"


def test_one_job_at_a_time_is_said_before_the_button_is_pressed(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    busy = out["queueRemoteBusy"]
    assert busy["generateDisabled"] is True
    assert busy["generateText"] == "One job at a time"
    assert "You already have an image generating" in busy["generateReason"]
    local = out["queueLocalBusy"]
    assert local["generateText"] == "Queue another image", \
        "the owner's page keeps its queue"
    assert local["generateDisabled"] is False
    finished = out["queueRemoteFinished"]
    assert finished["generateText"] == "Generate"
    assert finished["generateDisabled"] is False


def test_the_queue_position_is_the_servers_ordinal(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    position = out["queuePosition"]
    assert position["waiting"] == {"queue": "Your job is #2 in the queue.",
                                  "hidden": False}
    assert position["now"] == {"queue": "Your job is generating now.",
                              "hidden": False}
    assert position["finished"] == {"queue": "", "hidden": True}


def test_each_actionable_refusal_is_printed_as_its_own_sentence(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    typed = out["typed"]
    sentences = typed["sentences"]
    assert sentences["not_invited"].startswith("This account isn't invited")
    assert "does not have that feature" in sentences["feature_not_allowed"]
    assert sentences["queue_limit"].startswith("You already have an image")
    assert "prompt rewriter is not available" in sentences["rewrite_unavailable"]
    assert len(set(sentences.values())) == 4, "one sentence per code, no aliases"
    assert typed["unknown"] is None
    assert typed["notString"] is None
    assert typed["inherited"] is None, "prototype keys are not sentences"
    assert typed["api"] == sentences["queue_limit"]
    assert typed["apiUnknown"].startswith("invalid_envelope: ")
    assert "seed: must be a decimal string" in typed["apiUnknown"]
    assert typed["apiForeign"].startswith("The server refused the request")
    assert typed["rewrite"] == sentences["rewrite_unavailable"] + \
        " Your draft is unchanged."
    assert typed["rewriteTyped"].startswith(sentences["feature_not_allowed"])


def test_the_identity_block_is_refused_whole_when_anything_is_off(tmp_path):
    out = _run_multiuser_harness(tmp_path)
    block = out["userBlock"]
    assert block["good"]["email"] == "alice@example.com"
    assert block["good"]["displayName"] == "Alice"
    assert block["good"]["history"]["willReplaceOldest"] is True
    assert all(block["rejected"].values()), \
        [name for name, ok in block["rejected"].items() if not ok]
    assert block["optionalAbsent"]["history"]["willReplaceOldest"] is None
    assert block["nullBlock"] is True
    assert block["arrayBlock"] is True
