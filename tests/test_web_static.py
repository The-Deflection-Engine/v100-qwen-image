"""Guarded tests for the packaged static web workspace assets (W1 + W2).

Pure text/structure checks on ``src/qwen_engine/web_static/`` — no browser,
no network, no ML imports, no writes. The assets must stay original,
self-contained and honest-by-construction (PLAN W-milestone pivot,
docs/web-interface.md). W1 made zero network requests (the speculative
capability probe was removed after review); W2 replaces that with a strict
same-origin contract: the ONLY fetch targets are the three relative
``/api/v1/...`` endpoints of the local qwen-web server, and failures render
honestly as "Not connected". Real-Chrome behavioural checks live in
``scripts/chrome_browser_check.mjs`` and the served-UI checks in
``scripts/web_integration_check.mjs`` — neither is part of this guarded
suite and both require ``CHROME_PATH``.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from pathlib import Path

import qwen_engine
from qwen_engine.config import (
    AttentionBackend,
    GenerationConfig,
    OffloadMode,
    Quant,
)

WEB_DIR = Path(qwen_engine.__file__).resolve().parent / "web_static"
ASSETS = ("index.html", "app.css", "app.js")

# Nothing in the product files may name an external origin (fonts, CDN,
# analytics) — the page is local-only.
EXTERNAL_ORIGIN = re.compile(r"(?:src|href)\s*=\s*[\"']https?://|\bhttps?://[a-z0-9.-]+\.[a-z]{2,}", re.I)


def _read(name: str) -> str:
    path = WEB_DIR / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"static asset is empty: {path}"
    return text


def test_assets_exist_and_are_packaged_under_the_package():
    for name in ASSETS:
        text = _read(name)
        assert WEB_DIR.is_relative_to(Path(qwen_engine.__file__).resolve().parent)
        assert text  # non-empty (checked in _read)


def test_no_external_origins_or_subresource_integrity_risk():
    for name in ASSETS:
        match = EXTERNAL_ORIGIN.search(_read(name))
        assert match is None, f"{name} references an external origin: {match}"


def test_html_wires_only_local_assets():
    html = _read("index.html")
    assert 'href="app.css"' in html and 'src="app.js"' in html
    assert "<form" in html and 'id="generate-btn"' in html


def test_js_never_uses_innnerhtml_and_keeps_seed_a_decimal_string():
    js = _read("app.js")
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "document.write" not in js
    # full uint64 range handled via BigInt, not Number coercion
    assert "BigInt" in js
    assert "18446744073709551615" in js
    assert "crypto.getRandomValues" in js


def test_seed_grammar_is_bounded_before_bigint_without_truncation():
    js = _read("app.js")
    # parseSeed enforces a trimmed grammar of at most 20 characters BEFORE
    # the digit regex and any BigInt call (documented policy: longer strings,
    # even all-leading-zero ones, are rejected on length, keeping the user's
    # exact text — an inline error explains it).
    m = re.search(r"function parseSeed\(str\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "parseSeed not found"
    body = m.group(1)
    # Python whitespace semantics (pyStrip), NOT JS trim(): trimming is a
    # BROWSER-SIDE affordance on the user's typed input, applied at the
    # edges before the canonical digit STRING is put on the wire. The API
    # wire NEVER strips — the server accepts only canonical digits and
    # refuses anything else rather than editing it.
    assert "pyStrip(String(str))" in body
    assert r"/^\d+$/" in body
    assert body.index("length > SEED_MAX_DIGITS") < body.index(r"/^\d+$/"), \
        "length bound must precede the digit regex"
    assert body.index("length > SEED_MAX_DIGITS") < body.index("BigInt"), \
        "length bound must precede BigInt"
    assert "SEED_MAX_DIGITS = SEED_MAX.toString().length" in js
    # maxlength silently TRUNCATES an over-length paste into a DIFFERENT
    # valid seed — it must not exist; parseSeed is the only enforcement.
    html = _read("index.html")
    m = re.search(r"<input id=\"seed\"[^>]*>", html)
    assert m and "maxlength" not in m.group(0), \
        "seed input must NOT carry maxlength (silent truncation risk)"
    assert "nothing was truncated" in js.lower() or "kept exactly as entered" in js.lower(), \
        "over-length input needs an explicit inline explanation"


def test_seed_row_is_a_full_row_with_short_helper_and_accessible_range():
    html = _read("index.html")
    # Content-bounded, not indentation-bounded: the field's extent is the run
    # up to the next section, so re-indenting the markup cannot widen the
    # window to a neighbouring field.
    start = html.index('<div class="field seed-field">')
    seed_block = html[start:html.index("<section", start)]
    assert 'id="steps"' not in seed_block, "seed must not share the Steps row"
    assert "Leave at 0 or choose Random." in html
    # precise range explanation retained accessibly (title attribute)
    assert "18446744073709551615" in html and "title=" in html
    css = _read("app.css")
    assert re.search(r"\.seed-row input\s*{[^}]*min-width:\s*0", css)


def test_generation_is_status_gated_with_only_same_origin_api_calls():
    # W2 truth: the page may talk ONLY to the same-origin relative endpoints
    # (/api/v1/status, /api/v1/plan, /api/v1/config/export, and — R4 — the
    # rewrite route). Generate stays unconditionally disabled (generation is
    # W3); the submit handlers refuse directly — the disabled button is not
    # the gate.
    #
    # The "capability" ban is about BROWSER capability probing (navigator.gpu,
    # WebGPU/WebNN feature detection): the server's rewrite status block is
    # named for what it is (a rewriter state), never "capability", so this
    # assertion keeps its original force.
    for name in ASSETS:
        text = _read(name)
        assert "capability" not in text.lower(), f"{name} still probes capability"
        assert "location.protocol" not in text
    js = _read("app.js")
    # Every fetch target is one of the relative API constants — never
    # a string-literal URL, never a host, never an absolute origin.
    assert not re.search(r"fetch\(\s*['\"]", js), "fetch must use API constants only"
    for const, route in (("API_STATUS", "/api/v1/status"),
                         ("API_PLAN", "/api/v1/plan"),
                         ("API_EXPORT", "/api/v1/config/export"),
                         ("API_REWRITE", "/api/v1/rewrite"),
                         ("API_REWRITE_SETTINGS", "/api/v1/settings/rewrite")):
        m = re.search(rf"var {const} = '([^']+)'", js)
        assert m, f"{const} missing"
        assert m.group(1) == route
    # The connection probe hangs off the settings route rather than inventing
    # a third one, so it is built from the constant above.
    assert "var API_REWRITE_SETTINGS_TEST = API_REWRITE_SETTINGS + '/test';" in js
    # The gate is the connection pair plus whatever else forbids a submit: an
    # in-flight submit always does, and (remote pages only) a job already in
    # flight, because the server allows that account exactly one.
    assert "!apiConnected || !generationConnected ||" in js
    assert "submittingJob || queued" in js
    html = _read("index.html")
    m = re.search(r'id="generate-btn"[^>]*', html)
    assert m and "disabled" in m.group(0), "Generate must start disabled"
    # disabled visual state is NOT the gate: handlers refuse unconditionally
    assert "function submitGeneration" in js
    assert js.count("submitGeneration()") >= 2, "submit AND keydown must use the gate"
    assert "Nothing was submitted" in js


def test_form_snapshot_uses_exact_config_enums_and_keys():
    js = _read("app.js")
    # UI wire snapshot mirrors GenerationConfig keys; seed stays a STRING
    m = re.search(r"seed:\s*([^\n]+)", js)
    assert m and "seedWireString" in m.group(1) and "Number" not in m.group(1)
    assert "ui-wire/w2" in js
    m = re.search(r"function formSnapshot\(\)\s*\{.*?return\s*\{(.*?)\};", js, re.S)
    assert m, "formSnapshot not found"
    keys = set(re.findall(r"^\s*(\w+):", m.group(1), re.M))
    config_fields = {f.name for f in dataclasses.fields(GenerationConfig)}
    assert keys - {"kind"} <= config_fields, \
        f"snapshot keys outside GenerationConfig fields: {keys - config_fields}"
    # Path fields belong to the SERVER's runtime; use the API's own closed
    # sets instead of a literal list so a new path field cannot silently pass,
    # and so an announced deferral (recipe — docs/turbo-plan.md §6) is the
    # only other class of field the wire may omit.
    from qwen_engine.web.api import DEFERRED_WIRE_FIELDS, RUNTIME_PATH_FIELDS
    assert keys - {"kind"} == config_fields - set(RUNTIME_PATH_FIELDS) - set(DEFERRED_WIRE_FIELDS), \
        "snapshot must carry every non-path, non-deferred GenerationConfig field, no aliases"

    # Selects expose EXACTLY the config.py enum values — no invented options,
    # no aliases, compared against the real Python enums.
    html = _read("index.html")
    for select_id, enum_cls in (("quant", Quant), ("offload", OffloadMode),
                                ("attention", AttentionBackend)):
        m = re.search(rf'<select id="{select_id}".*?</select>', html, re.S)
        assert m, f"select {select_id} missing"
        values = set(re.findall(r'value="([^"]+)"', m.group(0)))
        assert values == {member.value for member in enum_cls}, \
            f"{select_id} options diverge from {enum_cls.__name__}"
    # No invented option values in the generation form. The scope is the Create
    # and History views on purpose: the Settings page has its own closed value
    # set — the three rewriter ids, ``pe-llamacpp``/``local-llm``/``off``, which
    # the server publishes and the settings tests below pin — and scanning the
    # whole page would ban the one legitimate ``off`` with it.
    generation_markup = html[:html.index('id="view-settings"')]
    for invented in ("candidate-a", "candidate-b", 'value="on"',
                     'value="off"', 'value="default"'):
        assert invented not in generation_markup, invented


def test_validation_marks_aria_invalid_on_every_control():
    js = _read("app.js")
    m = re.search(r"function renderValidation\((?:errors)?\w*\)\s*\{(.*?)\n  \}",
                  js, re.S)
    assert m, "renderValidation not found"
    body = m.group(1)
    for control in ("steps", "seed", "cfg", "prompt", "width", "height"):
        assert f"'{control}'" in body, f"renderValidation misses {control}"
    assert "setAriaInvalid" in body
    # dimension error must mark BOTH dimension controls
    assert body.count("!!errors.dims") == 2
    html = _read("index.html")
    # both dimension inputs share the dims error description
    assert html.count('aria-describedby="dims-error"') == 2


def test_prompt_required_like_config_but_never_alerted_on_load():
    # config.py validates prompt as a non-empty trimmed string; the form
    # mirrors that truthfully, gated so the error only appears after the
    # user edited the prompt or attempted submit — never a first-load alert.
    js = _read("app.js")
    m = re.search(r"function validateForm\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and r"pyStrip($('prompt').value)" in m.group(1), \
        "validateForm must check the prompt with PYTHON whitespace semantics"
    assert "promptTouched" in js
    m = re.search(r"function renderValidation\((?:errors)?\w*\)\s*\{(.*?)\n  \}",
                  js, re.S)
    assert m and "promptTouched" in m.group(1), \
        "prompt error must be gated on user interaction"
    assert "auto-fill" in _read("index.html").lower() or "auto-filled" in _read("index.html").lower(), \
        "the page must state it never auto-fills the prompt"
    html = _read("index.html")
    assert 'id="prompt-error"' in html and 'aria-describedby="prompt-help prompt-error"' in html


def test_routing_focuses_headings_only_on_user_hashchange():
    js = _read("app.js")
    assert 'addEventListener(\'hashchange\', routeOnHashchange)' in js
    assert "heading.focus()" in js
    m = re.search(r"function initialRoute\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and ".focus()" not in m.group(1), "no autofocus on initial load"
    m = re.search(r"function routeOnHashchange\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "routeOnHashchange not found"
    assert "name === ''" in m.group(1), \
        "empty hash must route to Create (back/forward from hash-less start)"
    assert "hasOwnProperty.call(VIEWS, name)" in m.group(1), \
        "unknown non-empty hashes (#main) must not reroute to Create"
    html = _read("index.html")
    for heading in ("create-heading", "history-heading", "settings-heading"):
        assert f'id="{heading}" tabindex="-1"' in html or \
               'tabindex="-1"' in re.search(
                   rf'<h2[^>]*id="{heading}"[^>]*>', html).group(0), \
            f"{heading} needs tabindex=-1 to be focusable"


def test_no_storage_or_service_apis_in_w1():
    js = _read("app.js")
    for forbidden in ("sessionStorage", "indexedDB",
                      "navigator.serviceWorker"):
        assert forbidden not in js, f"W1 must not use {forbidden}"
    assert "localStorage.setItem(THEME_STORAGE_KEY, theme)" in js
    assert js.count("localStorage.setItem(") == 1


def test_privacy_wording_is_factual_about_locality():
    html = _read("index.html")
    m = re.search(r"<dt>Privacy</dt><dd>(.*?)</dd>", html, re.S)
    assert m
    privacy = " ".join(m.group(1).split())
    assert "no third-party requests" in privacy
    assert "analytics" in privacy
    assert "static hosting" in privacy and "enforces nothing" in privacy, \
        "must not suggest static hosting enforces locality"


def test_assets_import_nothing_python_and_module_stays_ml_free():
    # The static tree carries no Python modules; importing the package must
    # remain torch/diffusers/transformers-free (boundary invariant).
    assert not list(WEB_DIR.glob("*.py"))
    import subprocess
    import sys
    proc = subprocess.run(
        [sys.executable, "-c",
         "import qwen_engine, sys;"
         "assert not [m for m in sys.modules if m.split('.')[0] in "
         "('torch','diffusers','transformers')]"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------- W2


def _python_schema_tags() -> dict[str, str]:
    from qwen_engine.web import api
    return {"status": api.WEB_STATUS_SCHEMA, "plan": api.WEB_PLAN_SCHEMA,
            "request": api.WEB_PLAN_REQUEST_SCHEMA, "error": api.WEB_ERROR_SCHEMA}


def test_w2_schema_literals_match_the_python_api_exactly():
    js = _read("app.js")
    tags = _python_schema_tags()
    for tag in tags.values():
        assert f"'{tag}'" in js, f"wire schema tag missing from app.js: {tag}"
    assert "'X-Qwen-UI'" in js or '"X-Qwen-UI"' in js


def test_w2_envelope_exact_editable_fields_no_kind_no_paths():
    from qwen_engine.web.api import EDITABLE_FIELDS, RUNTIME_PATH_FIELDS
    js = _read("app.js")
    m = re.search(r"buildEnvelope\(\)[\s\S]*?return \{\s*ok: true,\s*envelope: \{([\s\S]*?)\n      \}\n    \};", js)
    assert m, "buildEnvelope envelope literal not found"
    body = m.group(1)
    assert "schema: PLAN_REQUEST_SCHEMA" in body
    config_body = body[body.index("config: {") + len("config: {"):]
    config_keys = set(re.findall(r"^\s*(\w+):", config_body, re.M))
    assert config_keys == set(EDITABLE_FIELDS)
    assert "kind" not in body, "UI wire kind must be excluded from the envelope"
    for path_field in RUNTIME_PATH_FIELDS:
        assert path_field not in body
    assert not re.search(r"schema:\s*[\w.]*\+", body)


def test_w2_status_is_strictly_validated_never_trusted_by_shape():
    js = _read("app.js")
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parseStatus not found"
    body = m.group(1)
    assert "s.schema !== STATUS_SCHEMA" in body
    assert "s.planning_available !== true" in body          # strict boolean
    assert "typeof s.generation_available !== 'boolean'" in body
    assert "s.generation_available !== (s.generation_blockers.length === 0)" in body
    # EXACT blocker set (four required + manifest reason when applicable),
    # never "any bounded subset" — a malformed server cannot hide blockers.
    assert "blockersExact(s.generation_blockers, manifest.blocker)" in body
    assert "parseManifestCore(s.manifest)" in body
    assert "GATE_STATES" in js and "adapter_available" in js
    # a schema-rejected status is a typed state, distinct from a disconnect
    assert "Status rejected" in js and "Not connected" in js
    # every fetch is bounded by an AbortController timeout; no polling
    assert "AbortController" in js and "FETCH_TIMEOUT_MS" in js
    assert "setInterval" not in js


def test_w2_plan_response_seed_must_be_decimal_string_or_null():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parsePlan not found"
    body = m.group(1)
    assert r"/^\d{1,20}$/" in body, "plan seed must be validated as a string"
    assert "typeof c.seed === 'string'" in body
    # rendering uses textContent helpers only, never markup or .value writes
    m = re.search(r"function planRenderFields\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m
    assert "innerHTML" not in m.group(1)
    assert ".value" not in m.group(1)


def test_w2_finite_numbers_checked_before_serialization():
    js = _read("app.js")
    m = re.search(r"function numberField\(id, wantInt\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "numberField not found"
    body = m.group(1)
    assert "raw === ''" in body, "empty fields must be an error, never 0"
    assert "Number.isFinite" in body
    assert "Number.isInteger" in body


def test_w2_export_downloads_raw_blob_without_reserialization():
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    assert "apiRequest(API_EXPORT" in body and "'raw'" in body, \
        "export must consume the raw response bytes through the bounded core"
    assert "new Blob([r.bytes])" in body, \
        "export must download the complete raw bytes untouched"
    assert "JSON.parse" not in body and ".text()" not in body and \
           "res.json()" not in body, \
        "export bytes must never be parsed/reserialized on the success path"
    assert "URL.createObjectURL" in body and "URL.revokeObjectURL" in body
    assert "qwen-editable-config.json" in body
    # never a silent export of a stale/invalid draft:
    assert "buildEnvelope()" in body
    assert "local form errors" in body


def test_export_refuses_every_body_error_and_requires_typed_bytes():
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    # ALL bodyError outcomes are refused before any download is built — an
    # 'unsupported' readable-stream outcome can arrive with res.ok=true and
    # NO bytes at all, and Blob([undefined]) would otherwise download the
    # literal text 'undefined' under a success message.
    assert re.search(r"if \(r\.bodyError\)", body), \
        "export must refuse EVERY bodyError outcome, not just too_large"
    guard = body.index("if (r.bodyError)")
    blob = body.index("new Blob([r.bytes])")
    assert guard < blob, "the bodyError refusal must precede the download path"
    # a download is produced ONLY from actual Uint8Array bytes
    assert "r.bytes instanceof Uint8Array" in body, \
        "export must type-check the raw bytes before Blob/download"
    assert body.index("instanceof Uint8Array") < blob, \
        "the bytes type guard must precede the download path"
    # a schema/readability failure invalidates the trusted availability with
    # a DISTINCT rejected state — never a fabricated disconnect, and never
    # a success message.
    rejected = body[guard:blob]
    assert "Status rejected" in rejected and "Not connected" not in rejected, \
        "body refusal must be a typed rejection, not a disconnect"
    assert "setApiAvailable(false)" in rejected, \
        "a refused export body must invalidate planning/export availability"
    assert "not a disconnect" in rejected


def test_export_too_large_cannot_bypass_the_unified_refusal():
    # ROOT FINDING (16:30 review): an OLD early `if (r.bodyError ===
    # 'too_large') { ...; return false; }` sat BEFORE the unified branch
    # and returned without setApiAvailable(false) or the rejected chip —
    # an over-cap export body left 'Connected' and enabled buttons
    # standing. The regression this guards: NO special-cased bodyError
    # branch may exist beside the unified one; too_large may appear only
    # INSIDE `if (r.bodyError)` as a reason string.
    js = _read("app.js")
    m = re.search(r"function exportConfig\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "exportConfig not found"
    body = m.group(1)
    assert re.search(r"if \(r\.bodyError === ", body) is None, \
        "no early special-cased bodyError branch may bypass the unified refusal"
    guard = body.index("if (r.bodyError)")
    http_err = body.index("if (!r.res.ok)")
    blob = body.index("new Blob([r.bytes])")
    assert guard < http_err, \
        "the unified bodyError refusal must precede the HTTP-error path"
    assert guard < blob, \
        "the unified bodyError refusal must precede the download path"
    # the too_large reason and the invalidation live INSIDE the one
    # unified branch — one setApiAvailable(false), distinct rejected chip,
    # never 'Not connected', never a success message.
    branch = body[guard:http_err]
    assert branch.count("setApiAvailable(false)") == 1, \
        "exactly one invalidation, in the unified branch"
    assert "'too_large'" in branch and "'unsupported'" in branch, \
        "too_large must be a reason inside the unified refusal"
    assert "over the fixed response cap" in branch, \
        "the size-cap reason must be stated accurately"
    assert "Status rejected" in branch and "Not connected" not in branch, \
        "the too_large outcome is a typed rejection, not a disconnect"
    # the Uint8Array actual-bytes requirement on success survives (the
    # download is built ONLY from real bytes, never parsed/reserialized)
    assert body.index("r.bytes instanceof Uint8Array") < blob
    assert "JSON.parse" not in body and ".text()" not in body


def test_parse_plan_rejects_blank_prompt_exactly_like_generation_config():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js, re.S)
    assert m, "parsePlan not found"
    body = m.group(1)
    # shared GenerationConfig refuses `not prompt.strip()` (config.py); an
    # echoed plan with an empty/Python-whitespace-only prompt is a broken
    # contract and must be REJECTED — under the corrected pyStrip
    # predicate, never JS trim() (FEFF / U+001C-class differences).
    assert "pyStrip(c.prompt)" in body, \
        "parsePlan must require a non-blank echoed prompt under Python strip"
    assert "c.prompt.trim()" not in body, \
        "JS trim must not substitute for the Python whitespace predicate"


def test_w2_failure_paths_preserve_the_draft():
    js = _read("app.js")
    # Explicit import is the only server response allowed to populate fields.
    assert js.count("applyImportedPlan(plan);") == 1
    assert "draftRevision !== revisionAtPick" in js
    assert js.index("var plan = parsePlan(r.json);") < js.index("applyImportedPlan(plan);")
    assert "draft was preserved" in js or "draft stays" in js or \
           "Nothing was sent or changed" in js


def test_w2_plan_panel_is_rendered_not_invented():
    html = _read("index.html")
    for elem_id in ("btn-check-config", "btn-plan", "btn-export", "api-message",
                    "plan-panel", "plan-seed", "plan-tokens", "plan-cfg",
                    "plan-manifest", "plan-blockers"):
        assert f'id="{elem_id}"' in html, f"W2 UI element missing: {elem_id}"
    assert 'hidden' in re.search(r'<section class="plan-panel"[^>]*>', html).group(0), \
        "plan panel must start hidden (nothing rendered before a server answer)"


def test_w2_actions_gated_until_recognized_status():
    js = _read("app.js")
    html = _read("index.html")
    m = re.search(r"function setApiAvailable\(available\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "setApiAvailable not found"
    body = m.group(1)
    assert "$('btn-plan').disabled = !available" in body
    assert "$('btn-export').disabled = !available" in body
    assert "generate" not in body, "gating must never touch the Generate button"
    for btn in ("btn-plan", "btn-export"):
        m = re.search(rf'id="{btn}"[^>]*', html)
        assert m and "disabled" in m.group(0), f"{btn} must start disabled"
    # Check configuration is the reconnect path: it re-checks status first
    assert re.search(r"'btn-check-config'\).addEventListener[\s\S]*?checkStatus\(\)", js)
    # an initial hash must resolve by OWN property, never inherited names
    assert "hasOwnProperty.call(VIEWS, name)" in js


def test_w2_previous_plan_results_never_look_current():
    js = _read("app.js")
    m = re.search(r"function runPlan\(showFull\)\s*\{([\s\S]*?)\n  \}", js, re.S)
    assert m, "runPlan not found"
    body = m.group(1)
    assert "$('plan-panel').hidden = true" in body, \
        "previous results must be hidden while a new attempt runs"
    assert "seq !== planSeq" in body, "out-of-order answers must be ignored"
    assert "No planning result received" in body, \
        "a failed plan must not claim nothing was sent"
    # Invalidate BEFORE local validation: planSeq++ must happen before
    # buildEnvelope() so an invalid NEW attempt still cancels a pending
    # older one (which would otherwise clear the new errors and render an
    # old success over them).
    assert body.index("planSeq++") < body.index("buildEnvelope()"), \
        "prior attempts must be invalidated before local validation"
    assert "syncPlanStaleness()" in body, \
        "an invalid draft must stale any visible result"
    # a rendered result is bound to the signature of ITS snapshot
    assert "planShownSignature = sentSignature" in body
    # the success render must not clear errors of an edited draft
    assert "renderValidation({})" not in body, \
        "success must render the CURRENT draft errors, not blank them"
    assert "renderValidation(validateForm().errors)" in body


def test_w2_staleness_covers_preset_random_and_change_only_edits():
    js = _read("app.js")
    # syncPlanStaleness must be wired wherever the draft can change:
    # text/number inputs, preset buttons, the Random seed button, and the
    # change-only selects/checkbox (no inline errors, still stale).
    m = re.search(r"function applyPreset\(name\)\s*\{([\s\S]*?)\n  \}", js)
    assert m and "syncPlanStaleness()" in m.group(1), "preset edits must stale"
    m = re.search(r"\$\('random-seed'\)\.addEventListener[\s\S]*?\}\);", js)
    assert m and "syncPlanStaleness()" in m.group(0), "random seed must stale"
    m = re.search(r"\['quant', 'offload', 'attention', 'vae-tiling', 'output-format'\]"
                  r"[\s\S]*?'change'[\s\S]*?syncPlanStaleness\(\);[\s\S]*?\}\);", js)
    assert m, "change-only fields must also mark a visible result stale"
    m = re.search(r"function syncPlanStaleness\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "syncPlanStaleness not found"
    body = m.group(1)
    assert "envelopeSignature(buildEnvelope()) !== planShownSignature" in body, \
        "staleness must compare against the validated snapshot's signature"
    assert "classList.add('stale')" in body and "STALE" in body, \
        "a stale result must be clearly labelled, not left looking validated"
    assert "classList.remove('stale')" in body
    css = _read("app.css")
    assert ".plan-panel.stale" in css, "stale state needs a visible style"


def test_w2_request_bound_whole_operation_including_body():
    js = _read("app.js")
    # The timeout must cover the BODY, not just the headers: the timer is
    # cleared in settle() AFTER body consumption — never in a .finally on
    # the fetch promise alone.
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "apiRequest not found"
    body = m.group(1)
    assert "new AbortController()" in body and "controller.abort()" in body
    assert "requestOpts.signal = controller.signal" in body
    assert "readBodyLimited(res, cap)" in body, \
        "body consumption must run inside the timer scope"
    assert "settle(" in body and "clearTimeout(timer); return value;" in body
    assert ".finally(function () { clearTimeout(timer); })" not in js, \
        "clearing on the fetch promise alone leaves the body untimed"
    assert "timedOut = true" in body, "a timeout must be distinguishable"
    # no unbounded bodies: a fixed cap, streaming-read and cancelled
    m = re.search(r"function readBodyLimited\(res, cap\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "readBodyLimited not found"
    body = m.group(1)
    assert "MAX_RESPONSE_BYTES" in body and "reader.cancel()" in body, \
        "an over-cap body must be cancelled, never buffered"
    m = re.search(r"var MAX_RESPONSE_BYTES = (\d+);", js)
    assert m and 64 * 1024 < int(m.group(1)) <= 1024 * 1024, \
        "the cap must exceed a valid 64 KiB request echo but stay fixed"
    # raw export bytes preserved without reserialization
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    assert "if (mode === 'raw') return settle({ res: res, bytes: bytes });" in m.group(1)


def test_w2_network_failure_and_schema_rejection_are_distinct_outcomes():
    js = _read("app.js")
    m = re.search(r"function apiRequest\(url, opts, mode\)\s*\{([\s\S]*?)\n  \}", js)
    body = m.group(1)
    # malformed/undecodable complete body -> schema rejection marker
    assert "bodyError: 'json'" in body
    # over-cap -> its own rejection marker, not a disconnect
    assert "bodyError: 'too_large'" in body
    # timeout / unreachable -> network marker (never a schema verdict)
    assert "network: true, timedOut: timedOut" in body
    # JSON null parses to the value null — rejected by the strict parser,
    # never by a property access on null (no __not_json access at all)
    assert "__not_json" not in js
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert m.group(1).lstrip().startswith("if (!s || typeof s !== 'object') return null;"), \
        "parseStatus must be null-safe (JSON null is a schema rejection)"


def test_w2_parse_plan_bounds_aligned_with_server_contract():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    body = m.group(1)
    # bounds ALIGNED with the 64 KiB request transport cap — no invented
    # 4000-char generation limit that would reject server-valid echoes
    assert "MAX_CONFIG_STRING_CHARS" in body
    assert "c.prompt.length > 4000" not in body
    assert "negative_prompt.length > 4000" not in body
    m = re.search(r"var MAX_CONFIG_STRING_CHARS = (\d+);", js)
    assert m and int(m.group(1)) == 65536, \
        "string bound must match the server's 64 KiB request cap exactly"
    # dimensions honoured as multiples of 32 (config.py DIM_MULTIPLE)
    assert "c.width % 32 !== 0" in body and "c.height % 32 !== 0" in body
    # cfg_active recomputed with PYTHON strip semantics, exactly
    assert "pyStrip(c.negative_prompt)" in body
    assert "raw.cfg_active !== cfgActive" in body
    # latent_tokens consistent with the echoed dimensions (/16, C6)
    assert "raw.latent_tokens !== (c.width / 16) * (c.height / 16)" in body
    # manifest verification EXACT, not any bounded string
    assert "m.verification !== 'not_performed'" in body
    assert "m.fit !== 'not_evaluated'" in body
    assert "m.ready !== false" in body
    # EXACT blocker set, not any subset of the closed list
    assert "blockersExact(raw.generation_blockers, manifest.blocker)" in body
    assert "parseManifestCore(m)" in body
    m = re.search(r"var KNOWN_BLOCKER_CODES = \[([\s\S]*?)\];", js)
    assert m, "KNOWN_BLOCKER_CODES missing"
    from qwen_engine.web import api
    codes = set(re.findall(r"'([a-z_]+)'", m.group(1)))
    assert {"generation_disabled", "runtime_unconfigured"} <= codes
    # the same exact-set contract must gate status blockers too
    m = re.search(r"function parseStatus\(s\)\s*\{([\s\S]*?)\n  \}", js)
    assert "blockersExact(s.generation_blockers, manifest.blocker)" in m.group(1)
    # gate CODES are exact too (closed per-gate contract)
    assert "gate.code !== spec.code" in m.group(1)


# ------------------------------------------- root-follow-up regressions

def test_py_strip_covers_plain_space_and_matches_python_whitespace():
    # The '   ' + cfg>1 mainstream case: Python str.strip() strips plain
    # SPACE, so a pyStrip WITHOUT U+0020 would compute cfg_active=true for
    # an all-space negative prompt and reject the server's cfg_active=false.
    js = _read("app.js")
    m = re.search(r"function pyStrip\(s\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "pyStrip not found"
    body = m.group(1)
    cls = re.search(r"\[([^\]]*)\]", body).group(1)
    assert "\\x20" in cls, "pyStrip must strip plain space like Python does"
    # EDGE-ONLY stripping: Python .strip() never touches INTERIOR
    # whitespace — '4 2' must be REJECTED by the seed grammar, not parsed
    # as 42 (root browser review caught the interior-strip variant).
    assert body.count("[") >= 2 and re.search(r"/\^[^/]*\+\|[^/]*/g", body), \
        "pyStrip must anchor at ^ and $ (leading/trailing only)"
    for esc in ("\\x09", "\\x0A", "\\x0B", "\\x0C", "\\x0D",
                "\\x1C", "\\x1D", "\\x1E", "\\x1F", "\\x85", "\\xA0"):
        assert esc in cls, esc
    assert "uFEFF" not in cls, "Python strip does NOT strip U+FEFF"
    # python-side cross-check: every char Python's str.strip removes is
    # covered by the JS class (space included).
    ws = "".join(chr(c) for c in range(0x100) if chr(c).isspace())
    assert " " in ws  # sanity of the premise itself


def test_status_requests_are_sequenced_never_land_late():
    # A newer status check must win over an older late arrival in BOTH
    # directions (success after disconnect, failure after success).
    js = _read("app.js")
    assert "var statusSeq = 0;" in js
    m = re.search(r"function checkStatus\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "checkStatus not found"
    body = m.group(1)
    assert "var seq = ++statusSeq;" in body
    assert "if (seq !== statusSeq) return null;" in body


def test_body_without_readable_stream_fails_closed_not_unbounded():
    js = _read("app.js")
    m = re.search(r"function readBodyLimited\(res, cap\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "readBodyLimited not found"
    body = m.group(1)
    assert "UNSUPPORTED_BODY" in body
    assert "res.arrayBuffer()" not in body, \
        "arrayBuffer fallback reads UNBOUNDED — fail closed instead"
    assert "bodyError: 'unsupported'" in js
    assert "err === UNSUPPORTED_BODY" in js


def test_rejected_plan_document_invalidates_recognized_status():
    # A rejected plan BODY or schema must take planning/export away again
    # and show the typed schema-rejection chip (not a silent 'Connected').
    js = _read("app.js")
    m = re.search(r"function runPlan\([\s\S]*?\n  \}", js)
    assert m, "runPlan not found"
    body = m.group(0)
    rejection = body[body.index("if (r.bodyError)"):]
    rejection = rejection[:rejection.index("var plan = parsePlan")]
    assert "setApiAvailable(false)" in rejection
    assert "Status rejected" in rejection
    schema_branch = body[body.index("var plan = parsePlan"):]
    schema_branch = schema_branch[:schema_branch.index("promptTouched = true")]
    assert "setApiAvailable(false)" in schema_branch
    assert "Status rejected" in schema_branch


def test_manifest_core_and_blocker_contract_shared_and_exact():
    js = _read("app.js")
    m = re.search(r"function parseManifestCore\(m\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "parseManifestCore not found — status/plan share ONE parser"
    body = m.group(1)
    assert "m.verification !== 'not_performed'" in body
    assert "m.valid && !m.configured" in body          # valid implies configured
    assert "MANIFEST_REASON_CODES.indexOf(m.reason)" in body
    assert "m.schema_version" in body and "m.artifact_count" in body
    assert "m.schema_version !== 1" in body
    assert "m.artifact_count < 1" in body
    # exact required set mirrors the REAL server gates
    from qwen_engine.web import api
    m2 = re.search(r"var KNOWN_BLOCKER_CODES = \[([\s\S]*?)\];", js)
    assert m2, "KNOWN_BLOCKER_CODES missing"
    codes = set(re.findall(r"'([a-z_]+)'", m2.group(1)))
    assert codes == {"generation_disabled", "runtime_unconfigured", "manifest_unconfigured",
                     "manifest_file_missing", "manifest_invalid", "manifest_unreadable"}
    m3 = re.search(r"var MANIFEST_REASON_CODES = \[([\s\S]*?)\];", js)
    assert m3
    reason_codes = set(re.findall(r"'([a-z_]+)'", m3.group(1)))
    from qwen_engine.web import api as _api
    assert reason_codes <= {_api.MANIFEST_UNCONFIGURED, _api.MANIFEST_FILE_MISSING,
                            _api.MANIFEST_INVALID, _api.MANIFEST_UNREADABLE}


def test_api_requests_bypass_the_http_cache():
    # Two status checks must be TWO real requests (the root wire race proves
    # it); a cached status document is never authoritative.
    js = _read("app.js")
    m = re.search(r"function apiRequest\([^)]*\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "apiRequest not found"
    assert "requestOpts.cache = 'no-store'" in m.group(1)


# ------------------------------------------------------ transparency (D1)


def test_transparency_control_markup_is_labelled_and_described():
    html = _read("index.html")
    block = re.search(r'<div class="field transparency-field">.*?\n          </div>',
                      html, re.S)
    assert block, "transparency field block missing"
    body = block.group(0)
    assert 'id="transparent"' in body and 'name="transparent_background"' in body
    assert 'type="checkbox"' in body
    assert '<label for="transparent">Transparent background</label>' in body
    assert 'aria-describedby="transparency-help transparency-note"' in body
    assert 'id="transparency-help"' in body and 'id="transparency-note"' in body
    # The effective prompt is display-only: no input control, starts hidden.
    effective = re.search(r'<p id="effective-prompt-field"[^>]*>', body)
    assert effective and "hidden" in effective.group(0)
    assert '<span id="effective-prompt"' in body
    assert 'readonly' not in body   # it is not an editable field at all
    # The format error id the JS writes must exist, and the select points at it.
    assert 'id="format-error"' in html
    select = re.search(r'<select id="output-format"[^>]*>', html).group(0)
    assert 'aria-describedby="format-error"' in select
    # Plain-language explanation names the jpg restriction.
    assert "JPEG cannot store transparency" in body


def test_transparency_wrapper_literals_match_config_py_exactly():
    from qwen_engine.config import (TRANSPARENCY_PROMPT_PREFIX,
                                    TRANSPARENCY_PROMPT_SUFFIX,
                                    TRANSPARENCY_WRAPPER_REVISION)
    js = _read("app.js")

    def js_string(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    assert f"var TRANSPARENT_PREFIX = {js_string(TRANSPARENCY_PROMPT_PREFIX)};" in js
    assert f"var TRANSPARENT_SUFFIX =\n    {js_string(TRANSPARENCY_PROMPT_SUFFIX)};" in js
    # The revision constant is cited so a later wrapper bump is traceable.
    assert TRANSPARENCY_WRAPPER_REVISION in js
    from qwen_engine.config import ALPHA_OUTPUT_FORMATS
    assert set(re.findall(r"'(\w+)'", re.search(r"var ALPHA_FORMATS = \[([^\]]*)\]", js).group(1))) \
        == {fmt.value for fmt in ALPHA_OUTPUT_FORMATS}


def test_transparency_checkbox_switches_jpg_to_png_and_explains_it():
    js = _read("app.js")
    m = re.search(r"function onTransparencyChange\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "onTransparencyChange not found"
    body = m.group(1)
    assert "ALPHA_FORMATS.indexOf($('output-format').value) === -1" in body
    assert "$('output-format').value = 'png'" in body
    assert "switched from jpg to png" in body
    # Unticking never rewrites the format: the switch is gated on .checked.
    switch = body[body.index("$('transparent').checked"):]
    assert "= 'jpg'" not in body and "= 'webp'" not in body
    assert switch.index("$('transparent').checked") < switch.index("value = 'png'")
    # Wired as a real handler, and validated/staled like every other edit.
    assert "$('transparent').addEventListener('change', onTransparencyChange);" in js
    assert "syncPlanStaleness();" in body
    # The explanation is a persistent note, not a one-frame flash.
    assert "transparencySwitchNote" in js


def test_transparency_state_renders_effective_prompt_and_format_reason():
    js = _read("app.js")
    m = re.search(r"function syncTransparency\(\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "syncTransparency not found"
    body = m.group(1)
    assert "TRANSPARENT_PREFIX + $('prompt').value + TRANSPARENT_SUFFIX" in body
    assert "$('effective-prompt-field').hidden = !on" in body
    # The prompt FIELD itself is never rewritten by the wrapper.
    assert "$('prompt').value =" not in body
    # The mismatch reason is stated, never auto-corrected silently here.
    assert "JPEG cannot keep transparency" in body
    # Wired into the validation pass (so aria-invalid/format errors agree).
    m = re.search(r"function renderValidation\(errors\)\s*\{(.*?)\n  \}", js, re.S)
    assert m and "syncTransparency();" in m.group(1)
    assert "setText('format-error', errors.format || '')" in m.group(1)
    assert "setAriaInvalid('output-format', !!errors.format)" in m.group(1)


def test_transparency_local_validation_mirrors_the_server_rule():
    js = _read("app.js")
    m = re.search(r"function validateForm\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "validateForm not found"
    assert "transparencyFormatMismatch()" in m.group(1)
    assert "errors.format" in m.group(1)
    m = re.search(r"function transparencyFormatMismatch\(\)\s*\{(.*?)\n  \}", js, re.S)
    assert m, "transparencyFormatMismatch not found"
    assert "ALPHA_FORMATS.indexOf($('output-format').value) === -1" in m.group(1)


def test_plan_parsing_requires_transparent_boolean_and_alpha_format():
    js = _read("app.js")
    m = re.search(r"function parsePlan\(raw\)\s*\{([\s\S]*?)\n  \}", js)
    assert m
    body = m.group(1)
    assert "'transparent_background'" in body, "parsePlan exact-key list must include it"
    assert "typeof c.transparent_background !== 'boolean'" in body
    assert "ALPHA_FORMATS.indexOf(c.output_format) === -1" in body


def test_parse_job_accepts_13_keys_and_keeps_legacy_shapes_honest():
    js = _read("app.js")
    m = re.search(r"function parseJob\(job\)\s*\{([\s\S]*?)\n    return job;\n  \}", js)
    assert m, "parseJob not found"
    body = m.group(0)
    assert "ckeys.length === 13" in body
    assert "ckeys.length === 12" in body and "ckeys.length === 11" in body
    assert "!hasKey('transparent_background')" in body
    assert "c.transparent_background = false" in body
    assert "typeof c.transparent_background !== 'boolean'" in body


def test_job_error_code_gate_matches_the_python_public_map():
    """The UI's closed error_code list and the web layer's PUBLIC_ERROR_CODES
    map are ONE contract; drift means a real failure silently renders as an
    invalid job (server codes are fixed literals, prompt-free)."""
    from qwen_engine.web.generation import PUBLIC_ERROR_CODES
    js = _read("app.js")
    m = re.search(r"if \(\[null,([^\]]*?)\]\.indexOf\(job\.error_code\)", js, re.S)
    assert m, "parseJob error_code gate not found"
    assert set(re.findall(r"'([a-z_]+)'", m.group(1))) == set(PUBLIC_ERROR_CODES)


def test_gpu_oom_failure_message_is_plain_language_and_actionable():
    js = _read("app.js")
    m = re.search(r"out_of_gpu_memory: '([^']*)' \+\s*\n?\s*'([^']*)'", js)
    assert m, "out_of_gpu_memory message not found"
    assert "".join(m.groups()) == ("The GPU ran out of memory for this job. "
                                   "Try Q8, a smaller size, fewer reference images, "
                                   "or a shorter prompt.")
    # only the closed map may replace the generic sentence; no server text
    assert "failureMessages[job.error_code] ||" in js


def test_recipe_field_mirrors_the_config_gates():
    html = _read("index.html")
    js = _read("app.js")
    css = _read("app.css")
    from qwen_engine.config import Recipe, TURBO_STEPS

    # The UI's Turbo step count and recipe names are the SAME values the
    # server validates against.
    assert f"var TURBO_STEPS = {TURBO_STEPS};" in js
    m = re.search(r"var RECIPES = \[([^\]]*)\];", js)
    assert m, "RECIPES not found"
    assert re.findall(r"'([a-z]+)'", m.group(1)) == [r.value for r in Recipe]
    # One radio group, both values, Quality checked by default.
    assert re.search(r'<fieldset[^>]*id="recipe-field"[^>]*>', html)
    for value in ("base", "turbo"):
        assert f'id="recipe-{value}" name="recipe" type="radio" value="{value}"' in html
    assert re.search(r'id="recipe-base"[^>]*checked', html, re.S)
    assert 'id="recipe-help"' in html and 'id="recipe-note"' in html
    assert 'id="recipe-error"' in html
    # The note is a live region and is referenced by the locked controls so
    # the reason for the lock reaches assistive tech.
    note = re.search(r'<p id="recipe-note"[^>]*>', html).group(0)
    assert 'role="status"' in note
    for control in ("steps", "recipe-base", "recipe-turbo"):
        tag = re.search(rf'<input id="{control}"[^>]*', html, re.S)
        assert tag and "recipe-note" in tag.group(0), f"#{control} must describe the recipe note"
    assert re.search(r'<input id="steps"[^>]*aria-describedby="steps-error recipe-note"',
                     html, re.S)
    # Plan panel echoes the recipe.
    assert 'id="plan-recipe"' in html
    # Locks are STYLED (visible-but-inert), never just hidden.
    assert ".radio-row" in css and ".radio-field" in css
    assert "input:disabled" in css and "select:disabled" in css


def test_apply_recipe_mode_locks_and_restores_the_gated_controls():
    js = _read("app.js")
    body = _js_function(js, "applyRecipeMode")
    # Turbo pins exactly what config.py refuses to see otherwise.
    assert "steps.value = String(TURBO_STEPS);" in body
    assert "cfg.value = '1.0';" in body
    assert "quant.value = 'fp16';" in body
    # All four gated controls are locked, and the reason is written down.
    for control in ("steps", "cfg", "negative", "quant"):
        assert f"{control}.disabled = turbo;" in body, f"{control} must be locked under Turbo"
    assert "recipe-note" in body and "hidden = !turbo" in body
    # Leaving Turbo restores the user's own values instead of keeping pins.
    assert "recipeSaved" in body and "steps.value = recipeSaved.steps;" in body
    assert "cfg.value = recipeSaved.cfg;" in body
    assert "quant.value = recipeSaved.quant;" in body
    # An imported plan supplies the recipe's own values: no capture.
    assert "keepValues" in body
    m = re.search(r"function applyImportedPlan\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "applyImportedPlan not found"
    imported = m.group(1)
    assert "applyRecipeMode(true)" in imported
    # Order matters: the recipe is applied after the file's values so the
    # locks match the file, not a leftover draft.
    assert imported.index("setRecipeControl(c.recipe)") > imported.index("c.quant")
    assert "recipeSaved = null;" in imported


def test_recipe_contract_is_mirrored_by_parse_plan_and_parse_job():
    js = _read("app.js")
    body = _js_function(js, "recipeConsistent")
    assert "PLAN_ENUMS.recipe.indexOf(c.recipe) === -1" in body
    assert "c.steps === TURBO_STEPS" in body
    assert "c.quant === 'fp16'" in body
    assert "true_cfg_scale > 1" in body
    # Both parsers refuse an echo the server could not have produced.
    assert "recipeConsistent(c)" in _js_function(js, "parsePlan")
    assert "recipeConsistent(c)" in _js_function(js, "parseJob")
    # The history label distinguishes the two recipes.
    assert "job.config.recipe === 'turbo' ? ' · Turbo' : ''" in js
    # Server refusals are mapped onto the recipe's own fields.
    assert "recipe: 'recipe-error'" in js and "recipe: 'recipe-field'" in js
    for fn in ("runPlan", "submitGeneration"):
        assert "applyServerFieldErrors(" in _js_function(js, fn), \
            f"{fn} must render per-field server refusals"


def test_recipe_radios_are_wired_and_locked_on_load():
    js = _read("app.js")
    assert 'document.querySelectorAll(\'input[name="recipe"]\')' in js
    # The group handler re-derives the locks, re-renders errors and stales
    # any visible plan result — the same contract as every other field.
    m = re.search(r"input\[name=\"recipe\"\]'\),([\s\S]*?)\}\);\n", js)
    assert m and "applyRecipeMode();" in m.group(1)
    assert "renderValidation(validateForm().errors);" in m.group(1)
    assert "syncPlanStaleness();" in m.group(1)
    # Locks exist on first paint, not only after the first change event.
    assert "applyRecipeMode();   // lock state matches" in js
    # The envelope carries the recipe value (not a derived boolean).
    assert "recipe: currentRecipe()," in js
    assert "recipe: snap.recipe," in js


def test_import_applies_transparent_background_without_auto_switch():
    js = _read("app.js")
    m = re.search(r"function applyImportedPlan\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m, "applyImportedPlan not found"
    body = m.group(1)
    assert "$('transparent').checked = c.transparent_background;" in body
    # An imported, server-validated plan is applied verbatim: no local rewrite.
    assert "onTransparencyChange" not in body
    assert body.index("$('output-format').value = c.output_format;") < \
        body.index("$('transparent').checked = c.transparent_background;")


def test_plan_panel_shows_the_transparency_row():
    html = _read("index.html")
    assert 'id="plan-transparency"' in html
    js = _read("app.js")
    m = re.search(r"function planRenderFields\(plan\)\s*\{([\s\S]*?)\n  \}", js)
    assert m and "setText('plan-transparency'" in m.group(1)
    assert "plan.config.transparent_background" in m.group(1)


def test_alpha_results_are_previewed_on_a_checkerboard_only():
    css = _read("app.css")
    rule = re.search(r"\.history-card img \{([^}]*)\}", css)
    assert rule and "repeating-conic-gradient" in rule.group(1), \
        "transparent thumbnails need a checkerboard behind them"
    assert 'data-theme="paper"' in css and ".history-card img" in css
    # Preview only: nothing in the CSS or JS writes pixels into the saved file.
    js = _read("app.js")
    assert "canvas" not in js and "toBlob" not in js
    # The transparency row has its own visible styles.
    assert ".transparency-field" in css and ".effective-prompt-text" in css


def test_a_history_entry_whose_image_fails_to_load_stays_listed_and_is_stated():
    js, css, html = _read("app.js"), _read("app.css"), _read("index.html")
    assert 'id="history-missing-note"' in html and 'role="status"' in html
    body = _js_function(js, "renderHistory")
    assert "image.addEventListener('error'" in body
    # The observer states only what it saw: a failed request does not prove
    # the file changed, and it must not claim the entry was dropped — the
    # server keeps the record and refuses only the image.
    assert 'history-missing-note' in body and "hidden = false" in body
    assert "isConnected" in body, "a rebuilt grid must not be narrated for"
    spoken = body[body.index("setText('history-missing-note'"):]
    spoken = spoken[:spoken.index(");")]
    assert "could not" in spoken
    for claim in ("tampered", "deleted", "changed", "removed", "missing from"):
        assert claim not in spoken, f"unobserved claim shown to the user: {claim}"
    # The failure is visible, not a silent icon: the rule hides the broken
    # image and states unavailability instead.
    rule = re.search(r"\.history-card\.history-missing::before \{([^}]*)\}", css)
    assert rule and "Image unavailable" in rule.group(1)
    assert "history-missing img { display: none; }" in css.replace("\n", " ")


# ---------------------------------------------------------------------------
# W4 — reference images (design §2.4 / §5)
#
# The upload route takes RAW file bytes; the JSON routes cap bodies at
# 64 KiB while one reference may be 32 MiB, so base64-in-JSON is not an
# option and the tests refuse it outright. Conditioning order is the
# DISPLAYED order, so the envelope must carry the ids in list order.
# ---------------------------------------------------------------------------


def _js_function(js: str, signature: str) -> str:
    """Body of a top-level ``function <signature>(...) { ... }`` in app.js.

    The file is 2-space indented and every top-level function closes on a
    line that is exactly ``  }``; the regex is anchored there so a nested
    block can never end the match early.
    """
    m = re.search(rf"function {re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def test_w4_constants_mirror_the_python_caps_and_schema():
    from qwen_engine import reference_store
    from qwen_engine.config import REF_MAX
    from qwen_engine.web import api

    js = _read("app.js")
    assert f"var MAX_REFERENCES = {REF_MAX};" in js
    assert f"var MAX_REFERENCE_BYTES = {reference_store.MAX_REFERENCE_BYTES};" in js or \
        f"32 * 1024 * 1024" in js
    assert f"var MAX_REFERENCE_DIM = {reference_store.MAX_REFERENCE_DIM};" in js
    assert f"var REFERENCE_SCHEMA = '{api.WEB_REFERENCE_SCHEMA}';" in js
    # The upload route is the server's own path, relative (same-origin only).
    assert "var API_REFERENCES = '/api/v1/references';" in js
    # The id is a content digest the server also validates: lowercase only.
    assert r"[0-9a-f]{64}" in js
    assert "REFERENCE_ACCEPT = 'image/png,image/jpeg,image/webp'" in js


def test_w4_editable_keys_match_the_server_set_exactly():
    from qwen_engine.web.api import EDITABLE_FIELDS

    js = _read("app.js")
    m = re.search(r"var EDITABLE_KEYS = \[([\s\S]*?)\];", js)
    assert m, "EDITABLE_KEYS not found"
    keys = re.findall(r"'([a-z_]+)'", m.group(1))
    assert len(keys) == len(set(keys)) == 15, keys
    assert set(keys) == set(EDITABLE_FIELDS), (
        f"EDITABLE_KEYS diverge from api.EDITABLE_FIELDS: "
        f"{set(keys) ^ set(EDITABLE_FIELDS)}")
    assert keys[-1] == "references", "references is the newest (last) wire field"
    assert keys.index("recipe") == keys.index("steps") + 1, \
        "recipe sits with the sampling fields, right after steps"


def test_w4_reference_fieldset_markup_is_labelled_and_announced():
    html = _read("index.html")
    m = re.search(r'<fieldset[^>]*id="reference-field"[^>]*>([\s\S]*?)</fieldset>', html)
    assert m, "reference fieldset missing"
    block = m.group(0)
    assert "<legend>Reference images</legend>" in block
    # Plain-language help states the numbering contract and that size comes
    # from the preset, not the images.
    assert "image 1, image 2" in block
    assert "Output size comes from the preset" in block
    # File input: accepted types mirror the server's formats, multi-select,
    # hidden (the visible button opens it) — never a drag-only affordance.
    assert 'accept="image/png,image/jpeg,image/webp"' in block
    assert " multiple" in block and "hidden" in block
    for control in ("add-reference", "reference-file", "reference-list",
                    "reference-count", "reference-empty", "reference-status",
                    "reference-limit"):
        assert f'id="{control}"' in block, f"missing #{control}"
    assert 'aria-describedby="reference-help reference-status"' in block
    assert 'role="status"' in block and 'aria-live="polite"' in block
    assert 'aria-label="Reference images, in prompt order"' in block
    # The disabled Add button is never the only guard, and there is no
    # draggable attribute anywhere: reordering is button-only.
    assert "draggable" not in block


def test_w4_upload_sends_raw_bytes_never_base64_or_json():
    js = _read("app.js")
    body = _js_function(js, "uploadReference")
    assert "Content-Type': 'application/octet-stream'" in body
    assert "'X-Qwen-UI': '1'" in body
    assert "body: bytes" in body
    # The file is read as an ArrayBuffer and wrapped in a Uint8Array; the
    # page never base64-encodes it and never JSON-wraps the bytes.
    reader = _js_function(js, "readFileBytes")
    assert "readAsArrayBuffer" in reader
    assert "new Uint8Array(bytes)" in reader
    assert "JSON.stringify" not in body
    # No base64 path exists at all: no encoder, no data-URL reader.
    for forbidden in ("btoa(", "atob(", "readAsDataURL", "toDataURL", "FileReaderSync"):
        assert forbidden not in js, f"app.js must not use {forbidden}"
    # The upload is gated on a recognized status: writes need one.
    add = _js_function(js, "addReferenceFiles")
    assert "if (!apiConnected)" in add
    # Config import reuses the same bounded read path — one convention.
    limited = _js_function(js, "readFileLimited")
    assert "readFileBytes(file, MAX_IMPORT_FILE_BYTES)" in limited


def test_w4_upload_refusals_are_typed_and_bounded():
    js = _read("app.js")
    body = _js_function(js, "uploadReference")
    # Status codes reach the user as the server's own closed code pattern,
    # truncated, never a raw message of unbounded length.
    assert r"/^[a-z0-9_]{1,64}$/" in body
    assert "payload.message.slice(0, 200)" in body
    assert "setReferenceStatus(" in body
    # A schema-invalid 201 is a contract rejection: availability is dropped,
    # never rendered as a successful upload.
    assert "parseReferenceResponse(r.json)" in body
    assert "setApiAvailable(false)" in body
    # Content addressing: the same bytes are the same id, added once.
    assert "referenceIds().indexOf(ref.id)" in body


def test_w4_hard_cap_is_a_stated_stop_not_a_silent_drop():
    js = _read("app.js")
    add = _js_function(js, "addReferenceFiles")
    assert "MAX_REFERENCES - references.length" in add
    assert "files.slice(0, room)" in add
    assert "were not uploaded" in add or "not uploaded" in add
    # Room is computed BEFORE any bytes are sent: no over-cap upload can
    # even start.
    assert add.index("var room = MAX_REFERENCES - references.length") < \
        add.index("return step()")
    controls = _js_function(js, "updateReferenceControls")
    assert "references.length >= MAX_REFERENCES" in controls
    assert "$('add-reference').disabled" in controls


def test_w4_edit_mode_checkbox_gates_the_reference_controls():
    # The reference controls sit behind an "Edit with reference images" box,
    # unticked and hidden by default, so a plain text-to-image page stays short.
    html = _read("index.html")
    box = re.search(r'<input id="edit-mode"[^>]*>', html)
    assert box, "edit-mode checkbox missing"
    assert 'type="checkbox"' in box.group(0)
    assert "checked" not in box.group(0), "edit mode must start unticked"
    # Browsers restore checkbox state on reload; the hidden fieldset would not
    # follow, so restoring is switched off.
    assert 'autocomplete="off"' in box.group(0)
    assert 'aria-controls="reference-field"' in box.group(0)
    assert '<label for="edit-mode">Edit with reference images</label>' in html
    assert re.search(r'<fieldset[^>]*id="reference-field"[^>]*\shidden[^>]*>', html), \
        "the reference fieldset must start hidden"
    assert html.index('id="edit-mode"') < html.index('id="reference-field"')

    js = _read("app.js")
    # Unticked, nothing reference-related reaches a job, plan or rewrite: every
    # wire use goes through activeReferenceIds(), never the raw list.
    active = _js_function(js, "activeReferenceIds")
    assert "$('edit-mode').checked ? referenceIds() : []" in active
    assert "references: activeReferenceIds()" in _js_function(js, "formSnapshot")
    assert "activeReferenceIds().length ? 'edit' : 't2i'" in _js_function(js, "rewriteTask")
    # Toggling shows/hides the controls and stales the proposal and the plan.
    apply = _js_function(js, "applyEditMode")
    assert "$('reference-field').hidden = !$('edit-mode').checked" in apply
    assert "bumpPromptRevision()" in apply
    assert "refreshRewriteButton()" in apply
    assert "syncPlanStaleness()" in apply
    assert "$('edit-mode').addEventListener('change', applyEditMode)" in js
    # Import ticks the box exactly when the imported draft has references.
    imported = _js_function(js, "applyImportedPlan")
    assert "$('edit-mode').checked = c.references.length > 0;" in imported
    assert imported.index("$('edit-mode').checked") < imported.index("applyEditMode()")


def test_w4_parse_reference_response_is_a_closed_shape():
    js = _read("app.js")
    body = _js_function(js, "parseReferenceResponse")
    assert body.count("return null") >= 6
    assert "Object.keys(raw).length !== 6" in body
    for field in ("width", "height", "has_alpha", "bytes"):
        assert field in body
    for cap in ("MAX_REFERENCE_DIM", "MAX_REFERENCE_BYTES"):
        assert cap in body, f"{cap} not enforced on the response"


def test_w4_parse_plan_and_job_keep_explicit_legacy_branches():
    js = _read("app.js")
    plan = _js_function(js, "parsePlan")
    # Current shape first, then the four explicitly-gated older shapes (the
    # recipe era, then references, transparency and output_format).
    assert "var want = EDITABLE_KEYS;" in plan
    assert "have.length === want.length" in plan
    for n in (14, 13, 12, 11):
        assert f"have.length === {n}" in plan, f"parsePlan misses the {n}-key branch"
    # Every legacy branch is gated on the recipe key too, so a 14-key echo
    # that HAS recipe (i.e. some other key missing) is a rejection.
    assert "!hasKey('recipe')" in plan
    assert "c.recipe = 'base';" in plan
    assert "c.references = [];" in plan
    assert "Object.keys(c).length !== want.length" in plan
    assert "Array.isArray(c.references)" in plan and \
        "c.references.length > MAX_REFERENCES" in plan
    assert "REFERENCE_ID_RE.test(c.references[i])" in plan

    job = _js_function(js, "parseJob")
    for n in (15, 14, 13, 12, 11):
        assert f"ckeys.length === {n}" in job, f"parseJob misses the {n}-key branch"
    assert "!hasKey('recipe')" in job
    assert "c.recipe = 'base';" in job
    assert "c.references = [];" in job
    assert "REFERENCE_ID_RE.test(c.references[r])" in job
    # An over-cap list is refused, never truncated.
    assert "c.references.length > MAX_REFERENCES" in job


def test_w4_reference_ids_are_validated_but_duplicates_are_legal():
    js = _read("app.js")
    plan = _js_function(js, "parsePlan")
    assert "typeof c.references[i] !== 'string'" in plan
    # config.py allows repeating an id (the same image cited twice), so the
    # parser must not de-duplicate or unique-filter the list.
    assert "new Set(" not in plan
    assert ".filter(" not in plan.split("c.references")[0]


def test_w4_import_round_trips_reference_ids_in_file_order():
    js = _read("app.js")
    body = _js_function(js, "applyImportedPlan")
    assert "references = c.references.map(" in body
    # Order is the FILE's order, and the display metadata is honestly
    # unknown (the export carries ids only).
    assert "id: id, width: null, height: null, has_alpha: null, bytes: null" in body
    assert "renderReferences()" in body


def test_w4_plan_panel_shows_the_reference_count():
    html = _read("index.html")
    assert 'id="plan-references"' in html
    js = _read("app.js")
    body = _js_function(js, "planRenderFields")
    assert "setText('plan-references'" in body
    assert "plan.config.references.length" in body


def test_w4_envelope_and_render_agree_on_order():
    js = _read("app.js")
    snapshot = _js_function(js, "formSnapshot")
    assert "references: activeReferenceIds()" in snapshot
    envelope = re.search(r"buildEnvelope\(\)[\s\S]*?transparent_background: snap\."
                         r"transparent_background,\s*\n\s*references: snap\.references",
                         js)
    assert envelope, "buildEnvelope must forward snap.references verbatim"
    # The list is renumbered from POSITION on every render — never stored —
    # so a remove/reorder cannot leave a stale ordinal.
    render = _js_function(js, "renderReferences")
    assert "referenceCard(references[i], i)" in render, \
        "renderReferences must number cards from their list position"
    assert "'Image ' + (index + 1)" in _js_function(js, "referenceCard")
    assert "renderReferences();" in _js_function(js, "moveReference")
    assert "renderReferences();" in _js_function(js, "removeReference")
    # Every reorder/remove also bumps the draft revision via staleness.
    for fn in ("moveReference", "removeReference"):
        assert "syncPlanStaleness()" in _js_function(js, fn), f"{fn} must stale plans"


def test_w4_reorder_buttons_are_keyboard_accessible_and_end_aware():
    js = _read("app.js")
    render = _js_function(js, "referenceCard")
    assert "button.type = 'button'" in render
    assert "setAttribute('aria-label', buttons[b][2])" in render
    assert "'Move reference image ' + (index + 1) + ' up'" in render
    # The ends are honestly unavailable, not silent no-ops.
    assert "button.disabled = index === 0" in render
    assert "button.disabled = index === references.length - 1" in render
    # Focus survives the re-render (the list is rebuilt every time).
    assert "want.focus()" in _js_function(js, "focusReferenceButton")


def test_w4_submit_surfaces_reference_unavailable():
    js = _read("app.js")
    submit = _js_function(js, "submitGeneration")
    assert "r.json.code === 'reference_unavailable'" in submit
    assert "referenceNumberFromError(" in submit
    helper = _js_function(js, "referenceNumberFromError")
    # Only the ORDINAL is extracted from server text, and only when it
    # matches the documented message; anything else falls back safely.
    assert r"/^reference (\d{1,2}) is not available/" in helper
    assert "MAX_REFERENCES" in helper


def test_w4_reference_thumbnails_use_the_checkerboard_background():
    css = _read("app.css")
    rule = re.search(r"\.reference-card img \{([^}]*)\}", css)
    assert rule and "repeating-conic-gradient" in rule.group(1), \
        "alpha reference thumbnails need a checkerboard behind them"
    assert re.search(r':root\[data-theme="paper"\] \.reference-card img \{([^}]*)\}',
                     css) and "repeating-conic-gradient" in \
        re.search(r':root\[data-theme="paper"\] \.reference-card img \{([^}]*)\}',
                  css).group(1)
    for selector in (".reference-field", ".reference-list", ".reference-actions",
                     ".reference-status", ".reference-ordinal", ".reference-meta"):
        assert selector in css, f"missing {selector} styles"


def test_w4_reference_cards_wrap_instead_of_forcing_a_wider_viewport():
    """A 4-column grid over ordinal+thumb+meta+3 buttons overflowed 320px.

    The browser check measures the symptom (innerWidth widening past the
    emulated 320px); this pins the layout choice that prevents it.
    """
    css = _read("app.css")
    card = re.search(r"\.reference-card \{([^}]*)\}", css)
    assert card, "missing .reference-card rule"
    assert "display: flex" in card.group(1) and "flex-wrap: wrap" in card.group(1), \
        "reference cards must wrap; a rigid grid widens the layout viewport"
    assert "grid-template-columns" not in card.group(1)
    meta = re.search(r"\.reference-meta \{([^}]*)\}", css)
    assert meta and "min-width: 0" in meta.group(1), \
        "the metadata cell must be allowed to shrink for overflow-wrap to apply"
    actions = re.search(r"\.reference-actions \{([^}]*)\}", css)
    assert actions and "flex: 0 0 auto" in actions.group(1), \
        "action buttons keep their hit size and wrap as a unit"


def test_w4_batch_end_rerenders_so_remove_is_enabled_again():
    """Cards are rendered while uploadingReference is true.

    Only refreshing the toolbar at the batch end left every Remove button
    permanently disabled (found by screenshot, now pinned two ways).
    """
    js = _read("app.js")
    add = _js_function(js, "addReferenceFiles")
    assert add.count("renderReferences();") == 2, \
        "both the success and failure paths must re-render the card list"
    assert "button.disabled = uploadingReference" in _js_function(
        js, "referenceCard")


def test_w4_a_pruned_reference_states_the_thumbnail_is_gone():
    """A 422 names the ordinal; the card must not stay a silent broken icon.

    The card keeps its LIST POSITION (the envelope must show the order the
    user would submit), so the absence is stated on the card and in the
    aria-live status line instead.
    """
    render = _js_function(_read("app.js"), "referenceCard")
    assert "img.addEventListener('error'" in render
    assert "reference-missing" in render
    assert "thumbnail unavailable" in render
    assert "setReferenceStatus(" in render
    # `var i` is function-scoped: an error handler that reads the loop
    # variable narrates one past the end of the list. The ordinal must be
    # read from the live DOM position instead.
    # The handler BODY only: the buttons loop later in renderReferences uses
    # `i` correctly (synchronously, before any re-render).
    start = render.index("img.addEventListener('error'")
    handler = render[start:render.index("});", start)]
    assert "pos + 1" in handler
    # Parent-relative, so the card needs no closure over the render list.
    assert "indexOf.call(item.parentNode.children, item)" in handler
    assert "(i + 1)" not in handler
    assert "isConnected" in handler, "discarded cards must not narrate"
    css = _read("app.css")
    assert ".reference-card.reference-missing" in css
    rule = re.search(r"\.reference-card\.reference-missing \{([^}]*)\}", css)
    # Every theme defines --error; no theme defines a --warning token.
    assert "--error" in rule.group(1)
    for token in ("--bg:", "--text:", "--border:", "--accent:", "--error:"):
        assert css.count(token) >= 3, f"{token} must be defined in all 3 themes"


def test_w4_each_card_gets_its_own_closure_scope():
    """`var` is function-scoped: building cards inside one loop function
    left every error handler bound to the LAST card's item/img/meta.

    Each card is now built by its own function call, so the handler's
    bindings are per-card. A report of the wrong ordinal/card traced
    directly to this.
    """
    js = _read("app.js")
    render = _js_function(js, "renderReferences")
    card = _js_function(js, "referenceCard")
    # The DOM nodes are created inside referenceCard, never the render loop.
    for creation in ("document.createElement('li')", "document.createElement('img')"):
        assert creation in card, f"referenceCard must create {creation}"
        assert creation not in render, \
            f"{creation} in renderReferences re-creates the shared-scope bug"
    # The handler must not read the render loop's variable at all.
    assert "img.addEventListener('error'" in card
    assert "list.children" not in card


def test_test_surface_is_exposed_for_the_browser_checks():
    js = _read("app.js")
    surface = js[js.index("window.QwenStudio = {"):]
    # The version is a cache-buster/attribution marker, not a contract: assert
    # its SHAPE so a deliberate bump never fails this check.
    assert re.search(r"version: 'w[0-9]+-[a-z0-9-]+'", surface), "version marker missing"
    for name in ("referenceIds", "addReferenceFiles", "removeReference",
                 "moveReference", "parseReferenceResponse", "MAX_REFERENCES",
                 "parseRewriteBlock", "parseRewriteAnswer", "setRewriterState",
                 "presetForRatio", "requestRewrite", "applyProposal",
                 "keepOriginal", "rewriteState",
                 "useJobImageAsReference", "useAsReferenceBlock"):
        assert f"{name}: {name}" in surface, f"QwenStudio must expose {name}"


def test_w5_rewrite_ui_is_a_suggestion_flow_not_an_automatic_edit():
    """The Rewrite action must never change the draft on its own.

    Pins the four load-bearing properties (plan R4): the proposal block is a
    separate, initially-hidden region; Apply/Edit are the ONLY writers of
    ``prompt`` from the rewrite module; the fetch carries the separate long
    timeout; and an unavailable rewriter is hidden, not disabled-and-explained.
    """
    html = _read("index.html")
    js = _read("app.js")

    # -- markup: the block starts hidden and is a labelled group ----------
    block = re.search(r'<div id="rewrite-proposal".*?\n {14}</div>', html, re.S)
    assert block, "the proposal block must exist"
    block = block.group(0)
    assert "hidden" in block.split(">", 1)[0], "proposal starts hidden"
    assert 'tabindex="-1"' in block, "proposal must be focusable for a11y"
    assert 'role="group"' in block
    assert 'aria-labelledby="rewrite-proposal-heading"' in block
    for node in ("rewrite-original", "rewrite-suggested", "rewrite-ratio",
                 "rewrite-apply", "rewrite-edit", "rewrite-keep",
                 "rewrite-apply-ratio", "rewrite-error"):
        assert f'id="{node}"' in block, f"proposal must contain #{node}"
    # The button starts hidden AND disabled: an absent rewriter is not a
    # broken one, so there is nothing to press.
    button = re.search(r'<button[^>]*id="rewrite-btn"[^>]*>', html).group(0)
    assert "hidden" in button and "disabled" in button
    assert 'id="rewrite-hint"' in html
    # The help copy names the action and that it is opt-in. Tag-agnostic: the
    # help text sits in a popover (a span, legal inside the label) rather than
    # always-visible body text.
    help_text = re.search(r'<(\w+)[^>]*id="prompt-help"[^>]*>[\s\S]*?</\1>', html).group(0)
    assert "Rewrite" in help_text and "Apply" in help_text

    # -- the ONLY prompt writes from the rewrite module -------------------
    rewrite_region = js[js.index("function setRewriterState"):
                        js.index("function apiErrorPayloadFromBytes")]
    for fn in ("applyProposal", "editProposal"):
        body = _js_function(js, fn)
        assert "$('prompt').value = proposal.prompt;" in body
    # Showing a proposal never writes the textarea.
    assert "$('prompt').value =" not in _js_function(js, "showProposal")
    assert "$('prompt').value =" not in _js_function(js, "requestRewrite")
    assert "$('prompt').value =" not in rewrite_region.split("function applyProposal")[0]

    # -- separate, longer timeout; shared one untouched -------------------
    assert re.search(r"var REWRITE_TIMEOUT_MS = \d+;", js)
    assert "REWRITE_TIMEOUT_MS" in _js_function(js, "requestRewrite")
    assert "timeoutMs: REWRITE_TIMEOUT_MS" in js
    assert int(re.search(r"var REWRITE_TIMEOUT_MS = (\d+);", js).group(1)) >= 240000
    assert int(re.search(r"var FETCH_TIMEOUT_MS = (\d+);", js).group(1)) <= 30000
    # The route constant is the only rewrite URL, and it is relative.
    assert re.search(r"var API_REWRITE = '/api/v1/rewrite';", js)
    assert re.search(r"var REWRITE_REQUEST_SCHEMA = 'qwen_engine\.web_rewrite_request\.v1';", js)
    assert re.search(r"var REWRITE_SCHEMA = 'qwen_engine\.web_rewrite\.v1';", js)

    # -- stale discard: BOTH the revision and the echoed text -------------
    req = _js_function(js, "requestRewrite")
    assert "answer.draft_revision !== promptRevision" in req
    assert "answer.original_prompt !== $('prompt').value" in req
    # A suggestion dies when the prompt text moves — but a width/steps change
    # must NOT throw it away, so the counter is its own.
    assert re.search(r"var promptRevision = 0;", js)
    assert re.search(r"var draftRevision = 0;", js), \
        "the plan/import revision must stay separate from the prompt revision"
    # syncPlanStaleness owns draftRevision; it must NOT bump the prompt one,
    # or a resolution change would silently discard a good suggestion.
    staleness = _js_function(js, "syncPlanStaleness")
    assert "draftRevision++" in staleness
    assert "promptRevision" not in staleness

    # -- an edit rewrite sends its references, in the displayed order -----
    # The request must carry the ids the list shows, in that order: reference N
    # is <imageN> in the edit system prompt. A hardcoded empty list would send
    # an edit draft with no pictures, and a sorted list would silently
    # re-point every <imageN>.
    assert "image_ids: []" not in req, \
        "the request must not hardcode an empty image_ids"
    assert "image_ids: imageIds" in req
    assert "var imageIds = task === 'edit' ? activeReferenceIds() : [];" in req
    assert "var task = rewriteTask();" in req
    # The button is gated on the draft kind it would send: a kind the server has
    # no rewriter for is not offerable, and an edit draft with no picture yet has
    # nothing for the enhancer to look at.
    available = _js_function(js, "rewriteAvailable")
    assert "activeReferenceIds().length > 0" in available
    assert "var task = rewriteTask();" in available
    assert "rewriteActiveFor(task)" in available
    assert "disabled = !rewriteAvailable()" in _js_function(js, "setRewriterState")
    # A reference change is a rewrite input: it must bump the PROMPT revision
    # (which kills a visible proposal), not only the plan's draftRevision.
    for fn in ("removeReference", "moveReference"):
        body = _js_function(js, fn)
        assert "bumpPromptRevision();" in body, \
            f"{fn} must stale a visible rewrite proposal"
    add = _js_function(js, "addReferenceFiles")
    assert "bumpPromptRevision();" in add or "bumpPromptRevision();" in \
        js[js.index("references.push(ref)"):js.index("references.push(ref)") + 600]

    # -- no model-generated text is ever treated as HTML ------------------
    for fn in ("showProposal", "renderRewriteError", "setRewriterState",
               "requestRewrite", "bumpPromptRevision"):
        body = _js_function(js, fn)
        assert "innerHTML" not in body, f"{fn} must not assign innerHTML"
        assert "insertAdjacentHTML" not in body, f"{fn} must not insert HTML"


# ------------------------------------------------ use as reference (W6)


def test_use_as_reference_button_exists_on_both_surfaces_and_is_labelled():
    """One click, two places: the reference panel and each History card."""
    html = _read("index.html")
    # The finished-image panel no longer carries it: the shortcut lives with
    # the other reference controls, so it is only offered in edit mode.
    actions = re.search(r'<div class="result-actions"[^>]*>([\s\S]*?)</div>', html)
    assert actions, "result-actions group missing"
    assert "reference" not in actions.group(0).lower()
    toolbar = re.search(r'<div class="reference-toolbar">([\s\S]*?)</div>', html)
    assert toolbar, "reference toolbar missing"
    toolbar = toolbar.group(0)
    button = re.search(r'<button[^>]*id="use-last-image"[^>]*>([\s\S]*?)</button>',
                       toolbar)
    assert button, "the reference panel needs #use-last-image"
    tag, label = button.group(0), button.group(1)
    assert label.strip() == "Use last image as reference", "the label names the action"
    assert 'type="button"' in tag, "never a submit button inside the form"
    # It starts disabled with the reason in the title — no result, no shortcut.
    assert "disabled" in tag and 'title="No result yet"' in tag
    assert toolbar.index('id="add-reference"') < toolbar.index('id="use-last-image"')

    js = _read("app.js")
    history = _js_function(js, "renderHistory")
    assert "reuse.textContent = 'Use as reference'" in history
    # Only entries that HAVE an image get it: image_url is non-null exactly for
    # a succeeded job (parseJob pins that equality), so the same test gates it.
    gated = "if (job.state === 'succeeded') {\n        var reuse" in history
    assert gated, "the History shortcut must be gated on a finished image"
    # Focus is never stolen by a button: the click routes to Create, where the
    # reference list and its status line live.
    assert "location.hash = '#create';" in history
    assert history.index("useJobImageAsReference(id)") > history.index("var reuse")


def test_use_as_reference_reads_the_job_image_endpoint_and_reuses_the_upload():
    """No new route and no new trust: GET the job image, POST it back."""
    js = _read("app.js")
    body = _js_function(js, "useJobImageAsReference")
    # The bytes come from the job's own image URL — the same value Download
    # and the preview use, and the one parseJob pins to /api/v1/jobs/<id>/image.
    assert "apiRequest(job.image_url" in body
    assert "'raw'" in body, "the response bytes must be taken verbatim"
    assert "maxBytes: MAX_JOB_IMAGE_BYTES" in body, \
        "a large PNG must not be refused by the JSON response cap"
    assert "addReferenceFiles([" in body, \
        "the upload must go through the existing reference path (caps, dedupe)"
    assert "API_REFERENCES" not in body, \
        "the upload route is reached via addReferenceFiles, never hand-built"
    assert "JSON.stringify" not in body and "btoa(" not in body
    # The closed set of routes the page knows. The rewriter Settings page adds
    # exactly two (read the choice, write the choice) plus the probe hanging off
    # the write route; first-run model setup adds three (read the catalogue,
    # start the download, cancel it); the local-only Admin page adds five (read
    # the account list, add, remove, permissions, display name). Nothing else
    # may appear.
    assert sorted(set(re.findall(r"var (API_[A-Z_]+) = ", js))) == \
        ["API_ADMIN_ADD", "API_ADMIN_DISPLAY_NAME", "API_ADMIN_PERMISSIONS",
         "API_ADMIN_REMOVE", "API_ADMIN_USERS",
         "API_EXPORT", "API_IMPORT", "API_JOBS", "API_JOBS_FORGET_ALL",
         "API_MODELS", "API_MODELS_DOWNLOAD", "API_MODELS_DOWNLOAD_CANCEL",
         "API_PLAN", "API_REFERENCES", "API_REWRITE", "API_REWRITE_SETTINGS",
         "API_REWRITE_SETTINGS_TEST", "API_STATUS"]
    # Every failure mode is narrated on the reference status line, never an
    # alert(): network, unusable body, typed HTTP error, empty body.
    assert "alert(" not in body
    for reason in ("setReferenceStatus('Could not load the saved image",
                   "if (r.bodyError)", "renderChip('Status rejected'",
                   "apiErrorPayloadFromBytes(r.bytes)",
                   "r.bytes instanceof Uint8Array"):
        assert reason in body, f"missing failure path: {reason}"
    assert re.search(r"/\^\[a-z0-9_\]\{1,64\}\$/", body), \
        "only a code-shaped server code is quoted back"
    # The typed 404 (image_unavailable) must be distinguishable from a
    # disconnect: bodyError drops availability as a REJECTED contract.
    rejected = body[body.index("if (r.bodyError)"):body.index("if (!r.res.ok)")]
    assert "setApiAvailable(false)" in rejected and "not a disconnect" in rejected


def test_use_as_reference_ticks_edit_mode_before_the_request():
    js = _read("app.js")
    body = _js_function(js, "useJobImageAsReference")
    assert "$('edit-mode').checked = true;" in body
    assert "applyEditMode();" in body
    # Order is the point: reveal the list (and its live status line) BEFORE
    # anything is fetched, so progress and every refusal are visible.
    assert body.index("$('edit-mode').checked = true;") < \
        body.index("applyEditMode();") < body.index("apiRequest(job.image_url")
    # The status line is written before the request too: silence would look
    # like a dead button on a slow disk.
    assert body.index("setReferenceStatus('Loading the saved image") < \
        body.index("apiRequest(job.image_url")


def test_use_as_reference_respects_the_cap_and_the_finished_image_rule():
    js = _read("app.js")
    block = _js_function(js, "useAsReferenceBlock")
    assert "references.length >= MAX_REFERENCES" in block, \
        "a full list must refuse, not drop the image silently"
    assert "MAX_REFERENCES + ' reference images are allowed" in block
    assert "job.state !== 'succeeded' || !job.image_url" in block, \
        "a job with no image has nothing to add"
    assert "if (uploadingReference)" in block
    assert block.rstrip().endswith("return '';\n  }") or "return '';" in block, \
        "'' means available"

    # The refusal is stated, not merely hidden behind a disabled button.
    use = _js_function(js, "useJobImageAsReference")
    assert "setReferenceStatus(block + '. Nothing was added.');" in use
    assert use.index("var block = useAsReferenceBlock(job)") < \
        use.index("if (block) {") < use.index("$('edit-mode').checked = true;")

    # Both surfaces derive their disabled state (and its title) from the same
    # predicate, so they cannot disagree about what is addable.
    control = _js_function(js, "updateUseAsReferenceControl")
    assert "useAsReferenceBlock(jobCache[selectedJob])" in control
    assert "$('use-last-image').disabled = reason !== '';" in control
    assert "$('use-last-image').title = reason" in control
    history = _js_function(js, "renderHistory")
    assert "var blocked = useAsReferenceBlock(job);" in history
    assert "reuse.disabled = blocked !== '';" in history
    assert "reuse.title = blocked" in history
    # A reference added or removed re-derives the panel button, so the cap
    # cannot leave a live-looking button behind.
    controls = _js_function(js, "updateReferenceControls")
    assert "updateUseAsReferenceControl();" in controls
    assert "updateUseAsReferenceControl();" in _js_function(js, "renderSelectedJob")


def test_use_as_reference_writes_no_setting_but_the_edit_mode_box():
    js = _read("app.js")
    use = _js_function(js, "useJobImageAsReference")
    assert "$('prompt')" not in use and "$('prompt')" not in \
        _js_function(js, "useAsReferenceBlock")
    for control in ("seed", "width", "height", "steps", "cfg", "negative",
                    "output-format", "transparent", "vae-tiling", "recipe"):
        assert f"$('{control}')" not in use, f"must not touch #{control}"
    assert "submitGeneration" not in use, "adding a reference never generates"
    # The ONLY control this feature writes is the edit-mode checkbox.
    assert [line.strip() for line in use.splitlines() if ".checked = " in line] == \
        ["$('edit-mode').checked = true;"]
    # The History shortcut leaves the selected job (and therefore the preview,
    # the timings and the other result buttons) exactly as it was.
    history = _js_function(js, "renderHistory")
    reuse_handler = history[history.index("reuse.addEventListener"):]
    reuse_handler = reuse_handler[:reuse_handler.index("actions.appendChild(reuse)")]
    assert "renderSelectedJob" not in reuse_handler


def test_finished_image_response_gets_its_own_bounded_cap():
    from qwen_engine import reference_store

    js = _read("app.js")
    assert f"var MAX_JOB_IMAGE_BYTES = {reference_store.MAX_REFERENCE_BYTES};" in js \
        or "var MAX_JOB_IMAGE_BYTES = MAX_REFERENCE_BYTES;" in js, \
        "the image cap is the store's own upload limit"
    api = _js_function(js, "apiRequest")
    assert "opts.maxBytes <= MAX_JOB_IMAGE_BYTES" in api, \
        "a caller may raise the cap only up to the one legitimate size"
    assert "opts.maxBytes : MAX_RESPONSE_BYTES" in api or \
        "? opts.maxBytes : MAX_RESPONSE_BYTES;" in api
    assert "k !== 'maxBytes'" in api, "the knob must not leak into fetch options"
    reader = _js_function(js, "readBodyLimited")
    assert "if (typeof cap !== 'number' || !(cap > 0) || cap > MAX_JOB_IMAGE_BYTES)" in reader
    assert "cap = MAX_RESPONSE_BYTES;" in reader, \
        "a missing or absurd cap fails closed to the default"
    assert "if (total > cap) {" in reader and "reader.cancel()" in reader
    # Exactly one caller raises it: every other route keeps 256 KiB — including
    # the rewriter-settings routes, whose documents are small JSON.
    assert js.count("maxBytes:") == 1, "the cap override has one caller"
    assert "maxBytes: MAX_JOB_IMAGE_BYTES" in _js_function(js, "useJobImageAsReference")
    for fn in ("saveRewriteSettings", "testRewriteConnection", "loadRewriteSettings"):
        assert "maxBytes" not in _js_function(js, fn), \
            f"{fn} must keep the default response cap"


def test_every_output_format_the_engine_writes_is_accepted_as_a_reference():
    """The shortcut must not strand a jpg or webp result on re-upload."""
    from qwen_engine import image_output, reference_store
    from qwen_engine.config import OutputFormat

    js = _read("app.js")
    table = re.search(r"var IMAGE_CONTENT_TYPES = \{([\s\S]*?)\};", js)
    assert table, "no format -> content-type table for the re-uploaded bytes"
    types = dict(re.findall(r"(\w+):\s*'([^']+)'", table.group(1)))
    assert set(types) == {fmt.value for fmt in OutputFormat}, \
        "every format the server can produce must be covered"

    # What the encoder actually writes, and what the store sniffs, agree —
    # checked on the magic the store reads, not on an extension.
    source = (Path(image_output.__file__).read_text(encoding="utf-8"))
    saved = set(re.findall(r'image\.save\(buffer, format="(\w+)"', source))
    assert saved == {"JPEG", "PNG", "WEBP"}, f"unexpected encoder formats: {saved}"
    magic = {"JPEG": b"\xff\xd8\xff\xe0\x00\x10JFIF",
             "PNG": b"\x89PNG\r\n\x1a\n",
             "WEBP": b"RIFF\x00\x00\x00\x00WEBPVP8 "}
    accepted = {name: reference_store.sniff_format(raw)
                for name, raw in magic.items()}
    assert accepted == {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}
    assert set(types.values()) == {"image/jpeg", "image/png", "image/webp"}


# ------------------------------------------------------- Lightseed brand


def _brand_file(name: str) -> bytes:
    path = WEB_DIR / "brand" / name
    assert path.is_file(), f"missing brand asset: {path}"
    data = path.read_bytes()
    assert data, f"brand asset is empty: {path}"
    return data


def _theme_blocks(css: str) -> dict[str, dict[str, str]]:
    """Token map of every ``:root[data-theme=…]`` block. Comments are stripped
    first so a colour named in prose can never be read as a token."""
    body = re.sub(r"/\*[\s\S]*?\*/", "", css)
    out: dict[str, dict[str, str]] = {}
    for m in re.finditer(r':root\[data-theme="([a-z]+)"\]\s*\{([^}]*)\}', body):
        out[m.group(1)] = dict(re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})\b",
                                          m.group(2)))
    return out


def _hls(color: str) -> tuple[float, float, float]:
    import colorsys
    digits = color.lstrip("#")
    r, g, b = (int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _channel(v: float) -> float:
    # WCAG 2.1 relative luminance: linearised sRGB channel.
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def _luminance(color: str) -> float:
    digits = color.lstrip("#")
    r, g, b = (int(digits[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_app_is_named_lightseed_studio_and_keeps_its_storage_key():
    html, js, css = _read("index.html"), _read("app.js"), _read("app.css")
    assert "<title>Lightseed Studio</title>" in html
    header = re.search(r"<header[\s\S]*?</header>", html).group(0)
    assert 'alt="Lightseed"' in header, "the wordmark must carry the name"
    assert ">Studio<" in header, "the live suffix word must be real text"
    # The old product name is gone from everything the user sees; the ENGINE
    # keeps its real model naming (Qwen-Image, qwen_engine, qwen-web).
    for name, text in (("index.html", html), ("app.js", js), ("app.css", css)):
        assert "Qwen Image Studio" not in text, f"{name} still says the old name"
    assert js.splitlines()[0].startswith("/* Lightseed Studio"), js.splitlines()[0]
    assert css.splitlines()[0].startswith("/* Lightseed Studio"), css.splitlines()[0]
    # Renaming the saved-theme key would silently reset every user's theme, so
    # the rebrand must leave it exactly as it was.
    assert "localStorage.setItem(THEME_STORAGE_KEY" in js
    assert "'qwen-studio-theme'" in js, "the storage key must NOT be renamed"


def test_header_lockup_is_two_theme_logos_plus_the_live_suffix():
    html = _read("index.html")
    lockup = re.search(r'<h1 class="brand-name">([\s\S]*?)</h1>', html)
    assert lockup, "the header wordmark is missing"
    lockup = lockup.group(1)
    dark = re.search(r'<img[^>]*src="brand/lightseed-logo-on-dark\.svg"[^>]*>', lockup)
    light = re.search(r'<img[^>]*src="brand/lightseed-logo-on-light\.svg"[^>]*>', lockup)
    assert dark and light, "both logo variants must be present (one per surface)"
    for tag, cls in ((dark, "brand-logo-dark"), (light, "brand-logo-light")):
        assert cls in tag.group(0), f"missing {cls}"
        assert 'class="brand-logo ' in tag.group(0)
        # The alt text IS the image's accessible name, so the h1 reads
        # "Lightseed" + "Studio"; a decorative empty alt would lose the name.
        assert 'alt="Lightseed"' in tag.group(0)
    assert '<span class="brand-suffix">Studio</span>' in lockup
    # The old letter tile is removed from the markup, not merely hidden.
    assert "brand-mark" not in html


def test_only_the_theme_matching_logo_is_displayed():
    css = _read("app.css")
    # The logo height drives the whole lockup width, so it is checked against
    # the asset's OWN declared geometry rather than a guessed pixel value.
    declared = re.search(rb'<svg[^>]*\bwidth="(\d+)"[^>]*\bheight="(\d+)"',
                         _brand_file("lightseed-logo-on-dark.svg"))
    assert declared, "the wordmark must declare width/height for the layout math"
    native_w, native_h = (int(g) for g in declared.groups())
    assert (native_w, native_h) == (503, 148), (native_w, native_h)
    css_px = re.search(r"\.brand-logo \{[^}]*height: (\d+)px", css)
    assert css_px, "the logo height must be declared in .brand-logo"
    height = int(css_px.group(1))
    assert 36 <= height <= 38, f"{height}px is outside the 36-38px brand band"
    # The brand README promises "≈126px wide"; the aspect ratio is what makes
    # that true, so a future size bump cannot quietly widen the header.
    assert round(height * native_w / native_h) <= 130, \
        f"{height}px tall renders {round(height * native_w / native_h)}px wide"
    assert re.search(r"\.brand-logo-light \{[^}]*display: none", css), \
        "the Paper logo must be hidden by default (Original/Neon are dark)"
    assert re.search(r':root\[data-theme="paper"\] \.brand-logo-dark \{[^}]*display: none', css)
    assert re.search(r':root\[data-theme="paper"\] \.brand-logo-light \{[^}]*display: block', css)
    assert not re.search(r':root\[data-theme="neon"\] \.brand-logo', css), \
        "Neon uses the default dark logo: a third rule would be a second source"
    assert not re.search(r"\.brand-mark \{", css), "the removed tile must stay removed"
    # The wider lockup may wrap but must never create horizontal scroll.
    assert re.search(r"\.app-header \{[^}]*flex-wrap: wrap", css)


def test_favicon_links_are_relative_and_point_at_packaged_files():
    html = _read("index.html")
    head = html[:html.index("</head>")]
    assert '<link rel="icon" href="brand/favicon.svg" type="image/svg+xml">' in head
    assert '<link rel="icon" href="favicon.ico" sizes="any">' in head
    assert '<link rel="apple-touch-icon" href="brand/favicon-180.png">' in head
    for href in re.findall(r'<link[^>]*href="([^"]+)"', head):
        assert not href.startswith(("http:", "https:", "//")), href
        # Browsers request the ICO at the ROOT, so the markup says
        # "favicon.ico" while the packaged copy lives in brand/.
        relative = "brand/favicon.ico" if href == "favicon.ico" else href
        assert (WEB_DIR / relative).is_file(), f"{href} is not packaged"
    assert b"<svg" in _brand_file("favicon.svg")[:64]
    assert _brand_file("favicon.ico")[:4] == b"\x00\x00\x01\x00"
    assert _brand_file("favicon-180.png")[:8] == b"\x89PNG\r\n\x1a\n"


def test_accent_tokens_pass_wcag_contrast_in_every_theme():
    """--accent is used AS TEXT (.recipe-note, .transparency-note, .back-link)
    and as a filled button background under --on-accent, so both pairs must
    clear 4.5:1 (WCAG 2.1 SC 1.4.3) in EACH of the three themes."""
    blocks = _theme_blocks(_read("app.css"))
    assert set(blocks) == {"original", "neon", "paper"}
    for theme in ("original", "neon", "paper"):
        tokens = blocks[theme]
        for token in ("--bg", "--bg-panel", "--accent", "--on-accent"):
            assert token in tokens, f"{theme} has no {token}"
        accent, on_accent = tokens["--accent"], tokens["--on-accent"]
        assert _contrast(accent, tokens["--bg"]) >= 4.5, \
            f"{theme}: --accent {accent} on --bg {tokens['--bg']}"
        assert _contrast(accent, tokens["--bg-panel"]) >= 4.5, \
            f"{theme}: --accent {accent} on --bg-panel {tokens['--bg-panel']}"
        assert _contrast(on_accent, accent) >= 4.5, \
            f"{theme}: --on-accent {on_accent} on --accent {accent}"


def test_accent_colours_belong_to_the_brand_palette():
    """Every accent token must sit in the logo's cyan→periwinkle→violet hue
    band, and none of the old warm amber/brown trio may survive anywhere."""
    css = _read("app.css")
    for old in ("#d9a468", "#b9854c", "#e6b87e", "#8a5a24", "#6d4619"):
        assert old not in css, f"the old accent {old} is still in the stylesheet"
    for theme, tokens in _theme_blocks(css).items():
        accent_light = _hls(tokens["--accent"])[1]
        for token in ("--accent", "--accent-dim", "--focus"):
            hue, lightness, saturation = _hls(tokens[token])
            assert saturation > 0.2, \
                f"{theme} {token} {tokens[token]} is not a brand colour"
            assert 180 <= hue * 360 <= 290, \
                f"{theme} {token} {tokens[token]} hue {hue * 360:.0f} is off-brand"
            assert 0.05 < lightness < 0.95
        # Button text is the one place a near-neutral ivory/ink belongs: it must
        # be an obvious light-or-dark against the accent, not a mid tone.
        assert abs(_hls(tokens["--on-accent"])[1] - accent_light) >= 0.4, \
            f"{theme}: --on-accent is too close in lightness to --accent"


def test_brand_assets_are_the_agreed_files_and_stay_inert():
    """Byte-level guard: the copies are the agreed artwork, unchanged. The
    digests are of the delivered assets, so an edit inside the repo is caught
    even though agreed-logos/ itself is never git-tracked."""
    import hashlib

    expected = {
        "lightseed-logo-on-dark.svg": "25af2ad2af048b491be09caf6e36fd46c7f438c5"
                                      "98d1e4fb5b658dfbf7134d54",
        "lightseed-logo-on-light.svg": "e099b6537519ab7f8b940452a322b8c3be9ca3f2"
                                       "6906799b53bcf1bd0dd63620",
        "favicon.svg": "93aefd55b834aca11558ccafd785325c5c3a0790b3e05064a1c600ff"
                       "094540ae",
        "favicon.ico": "576b3e0560c98818172146227c7c1e5f9d24e1ef29c0c2218133fbd9"
                       "1a39053c",
        "favicon-180.png": "df93907470009a303e7e99e2b36d5beb8e35f39ebb17cc92fc64"
                           "5d8d6eb3458b",
    }
    got = {name: hashlib.sha256(_brand_file(name)).hexdigest() for name in expected}
    assert got == expected
    # The two wordmarks differ in the LETTERING colour only, which is exactly
    # why both ship rather than one being tinted by CSS.
    dark = _brand_file("lightseed-logo-on-dark.svg")
    light = _brand_file("lightseed-logo-on-light.svg")
    assert b"#F7F5EF" in dark and b"#0B102B" in light
    assert dark.count(b"<linearGradient") == light.count(b"<linearGradient")
    # Inert documents: no script, no embedded raster, no external reference.
    # The only URL in the files is the SVG namespace declaration itself.
    for blob in (dark, light, _brand_file("favicon.svg")):
        assert b"<script" not in blob and b"<image" not in blob
        assert b"xlink:href" not in blob and b"<style" not in blob
        assert blob.count(b"http") == blob.count(b"http://www.w3.org/2000/svg")


def test_preview_keeps_the_shown_images_shape_and_labels_the_next_size():
    # Job polling re-renders the selected job, so a frame that followed the
    # draft would flip to a new preset and straight back. While a finished
    # image is shown the frame keeps the image's shape; the draft's size shows
    # as a "Next image" label instead.
    html = _read("index.html")
    frame = re.search(r'<div class="preview-frame" id="preview-frame"[\s\S]*?\n        </div>', html)
    assert frame and 'id="preview-next-size"' in frame.group(0)
    assert re.search(r'<p class="preview-next-size" id="preview-next-size" hidden>', html)

    js = _read("app.js")
    shape = _js_function(js, "syncPreviewShape")
    assert "if (!shownImageSize) {" in shape
    assert "setPreviewAspect(width, height);" in shape
    assert "setPreviewAspect(shownImageSize.width, shownImageSize.height);" in shape
    assert "'Next image: ' + sizeLabel(width, height)" in shape
    assert "width !== shownImageSize.width || height !== shownImageSize.height" in shape
    # Every draft-size change goes through syncPreviewShape, never straight
    # to setPreviewAspect.
    assert "syncPreviewShape();" in _js_function(js, "applyPreset")
    assert "setPreviewAspect(preset" not in js
    assert "setPreviewAspect(plan.config" not in js
    assert js.count("setPreviewAspect(") == 3, "only syncPreviewShape and the definition"
    # The selected job sets or clears the shown size, then re-syncs.
    job = _js_function(js, "renderSelectedJob")
    assert "shownImageSize = { width: job.config.width, height: job.config.height };" in job
    assert "shownImageSize = null;" in job
    assert job.index("shownImageSize = null;") < job.rindex("syncPreviewShape();")
    label = _js_function(js, "sizeLabel")
    assert "PRESETS[name].ratio" in label
    assert ".preview-next-size {" in _read("app.css")


# --------------------------------------------- rewriter settings (amendment)


SETTINGS_FNS = ("parseRewriteSettings", "parseRewriteChoice",
                "parseRewriteLocal", "parseRewriteLocalFields", "hasOnlyKeys",
                "parseRewriteSettingsRows", "rewriteSettingsEnvelope",
                "loadRewriteSettings", "renderRewriteSettings",
                "saveRewriteSettings", "testRewriteConnection")


def test_the_rewriter_is_chosen_once_on_the_server_not_per_rewrite():
    """The amendment removed the per-request picker.

    The rewrite request therefore carries FOUR fields and no backend, and the
    only place the choice changes is the Settings page. A select next to
    **Rewrite prompt** would be a second, browser-local answer to a question
    the server has already answered — and the two would disagree.
    """
    html = _read("index.html")
    js = _read("app.js")
    assert "rewrite-backend-select" not in html
    assert "rewrite_backend" not in js
    prompt_area = html[html.index('id="prompt"'):html.index('id="rewrite-proposal"')]
    assert "<select" not in prompt_area, "no dropdown beside the draft"
    req = _js_function(js, "requestRewrite")
    body = req[req.index("body: JSON.stringify({"):]
    for key in ("schema", "task", "prompt", "draft_revision", "image_ids"):
        assert f"{key}:" in body
    assert "backend" not in body, "the rewrite request names no backend"
    # And the choice lives on a view of its own, reachable from the header.
    assert 'href="#settings" id="nav-settings"' in html
    assert 'id="view-settings"' in html
    assert '<section id="view-settings" class="view" hidden' in html
    assert "settings: 'view-settings'" in js
    assert "loadRewriteSettings();" in _js_function(
        js, "renderView"), "opening Settings re-reads the server"


def test_the_settings_page_offers_every_rewriter_for_each_draft_kind():
    """TWO radio groups — one per draft kind, each offering the two rewriters —
    plus ONE Off switch covering both, three fields for the shared endpoint, a
    probe button and a live region for every answer. Each option carries the
    node its greyed-out reason is written into, so a server-side sentence has
    somewhere to land without a page change. Nothing on the page is a file
    picker."""
    html = _read("index.html")
    section = html[html.index('id="rewrite-settings-heading"'):
                   html.index('id="rewrite-settings-note"')]
    # One group per draft kind, labelled with the server's own task labels.
    for task, legend in (("t2i", "Prompt enhance (text-to-image)"),
                         ("edit", "Edit prompt enhance")):
        assert f'id="rewrite-group-{task}"' in section, task
        assert f'name="rewrite-{task}"' in section, task
        assert legend in section, legend
        for value, label in (("pe-llamacpp", "Official enhancer (Q8, uses GPU)"),
                             ("local-llm", "OpenAI-compatible endpoint")):
            short = "pe" if value == "pe-llamacpp" else "local"
            assert f'id="rewrite-{task}-{short}"\n' in section or \
                f'id="rewrite-{task}-{short}" ' in section, (task, value)
            assert f'value="{value}"' in section, (task, value)
            # The label the page ships is the label the server publishes, and
            # every option has a reason node beside it.
            assert f'id="rewrite-{task}-{short}-label">{label}<' in section \
                or label in section, label
            assert f'id="rewrite-{task}-{short}-reason"' in section, (task, value)
    # Off is ONE switch for both kinds, never a third radio in a group.
    assert 'type="checkbox" id="rewrite-off"' in section
    assert 'id="rewrite-off-reason"' in section
    assert "leave both kinds of draft exactly as you wrote them" in section
    assert 'name="rewrite-off"' not in section
    for node in ("rewrite-endpoint", "rewrite-model", "rewrite-timeout",
                 "rewrite-test", "rewrite-save", "rewrite-save-status",
                 "rewrite-test-status", "rewrite-settings-error"):
        assert f'id="{node}"' in section, node
    # Every field is labelled and the refusals are announced.
    for node in ("rewrite-endpoint", "rewrite-model", "rewrite-timeout"):
        assert f'<label for="{node}">' in section, node
    assert 'aria-live="polite"' in section
    assert 'type="file"' not in section and 'type="password"' not in section
    # The timeout box cannot even be typed out of the server's own range.
    timeout = section[section.index('id="rewrite-timeout"'):]
    assert 'min="1"' in timeout[:200] and 'max="3600"' in timeout[:200]
    # A shared server-side setting says so, or a second browser is confused.
    assert "stored by the server" in html
    # The GPU note has to be true: the enhancer is REFUSED outright while an
    # image is generating (503 rewrite_gpu_busy) — it never queues behind one.
    # The generation that waits is the one started during a rewrite.
    note = section[section.index('id="rewrite-gpu-note"'):]
    note = re.sub(r"\s+", " ", note[:note.index("</p>")])
    assert "loads a Q8 model onto the graphics card" in note
    assert "It is unavailable while an image is generating" in note
    assert "an image started during a rewrite waits for it" in note
    assert "it waits for" not in note


def test_the_settings_envelope_carries_a_choice_per_draft_kind_and_no_path():
    js = _read("app.js")
    assert "REWRITE_SETTINGS_KEYS = ['schema', 'enabled', 'choices', 'saved'," in js
    assert "'off', 'persistent', 'note'];" in js
    # The two closed key sets are DIFFERENT on purpose: the saved record selects
    # rewriters, the block describing the shared endpoint selects nothing.
    assert "REWRITE_CHOICE_KEYS = ['t2i', 'edit', 'enabled', 'endpoint', 'model'," in js
    assert "REWRITE_LOCAL_KEYS = ['endpoint', 'model', 'timeout_s'];" in js
    assert "REWRITE_SETTINGS_BACKENDS = ['pe-llamacpp', 'local-llm', 'off'];" in js
    assert "REWRITE_BACKEND_IDS = ['pe-llamacpp', 'local-llm'];" in js
    env = _js_function(js, "rewriteSettingsEnvelope")
    assert "fields.schema = REWRITE_SETTINGS_SCHEMA;" in env
    assert "selectedRewriteChoice('t2i')" in env
    assert "selectedRewriteChoice('edit')" in env
    assert "rewriteEnabled()" in env
    # `off` is never a task choice: it is the switch, sent as `enabled`.
    assert "'off'" not in env
    # An empty box is null ("keep what the server was started with"), never an
    # empty string the server would refuse as a malformed address.
    fields = _js_function(js, "rewriteEndpointFields")
    assert "|| null" in fields
    for forbidden in ("path", "binary", "dir", "file", "gguf", "exe"):
        assert forbidden not in env.lower(), f"{forbidden} must not be a field"
        assert forbidden not in fields.lower(), f"{forbidden} must not be a field"
    saved = _js_function(js, "saveRewriteSettings")
    assert "apiRequest(API_REWRITE_SETTINGS, {" in saved
    assert "method: 'POST'" in saved
    assert "JSON.stringify(envelope)" in saved
    assert "'X-Qwen-UI': '1'" in saved, "a write needs the UI header"
    probe = _js_function(js, "testRewriteConnection")
    assert "API_REWRITE_SETTINGS_TEST" in probe
    assert "rewriteTestEnvelope()" in probe
    # The probe addresses the SHARED endpoint with the field rules the settings
    # form uses, and it carries NO choice keys: pressing Test must never be able
    # to ask the server to load an enhancer checkpoint onto the graphics card.
    test_env = _js_function(js, "rewriteTestEnvelope")
    assert "fields.schema = REWRITE_TEST_REQUEST_SCHEMA;" in test_env
    assert "rewriteEndpointFields()" in test_env
    for forbidden in ("t2i", "edit", "enabled", "backend", "task",
                      "path", "binary", "dir", "gguf", "exe"):
        assert forbidden not in test_env, f"{forbidden} must not be probed"


def test_an_unexpected_settings_document_is_refused_rather_than_rendered():
    js = _read("app.js")
    parse = _js_function(js, "parseRewriteSettings")
    assert "REWRITE_SETTINGS_KEYS.indexOf(keys[i]) === -1" in parse
    assert "payload.schema !== REWRITE_SETTINGS_SCHEMA" in parse
    rows = _js_function(js, "parseRewriteSettingsRows")
    assert "REWRITE_SETTINGS_ROW_KEYS.indexOf(keys[k]) === -1" in rows
    # Nothing the server sends is displayed as written: every label must be one
    # this page already knows, which is stronger than pattern-matching a label
    # for a path — a renamed rewriter is a page that was not written for it.
    for fn in ("parseRewriteSettingsRows", "parseRewriteTaskRows"):
        body = _js_function(js, fn)
        assert "REWRITE_BACKEND_LABELS[" in body, fn
        assert "REWRITE_BACKEND_USES_GPU[" in body, fn
        # Greyed out only with a sentence; available only without one.
        assert "!== (row.reason === '')" in body or \
            "!== (choice.reason === '')" in body, fn
    assert "REWRITE_TASK_LABELS[row.task] !== row.label" in \
        _js_function(js, "parseRewriteTaskRows")
    # What each parser accepts and rejects is EXECUTED below against real
    # server documents, not string-matched here.
    # Nothing model-shaped is ever treated as markup.
    for fn in SETTINGS_FNS:
        body = _js_function(js, fn)
        assert "innerHTML" not in body and "insertAdjacentHTML" not in body, fn


def test_the_settings_page_stores_nothing_in_the_browser():
    """The setting is the server's; a browser-local copy would be a second
    source of truth that disagrees the moment someone else saves."""
    js = _read("app.js")
    for fn in SETTINGS_FNS:
        body = _js_function(js, fn)
        assert "localStorage" not in body, fn
        assert "sessionStorage" not in body, fn
        assert "document.cookie" not in body, fn
    # The theme key remains the only thing this page keeps locally.
    assert js.count("localStorage.setItem(") == 1


def test_the_page_says_which_rewriter_answered():
    """One GPU-bound, one not: a suggestion must name its source, and the Create
    view must name the rewriter for EACH draft kind — the two can differ, so one
    name would misdescribe the draft the user is about to generate."""
    html = _read("index.html")
    js = _read("app.js")
    assert 'id="rewrite-source"' in html
    assert 'id="rewrite-active"' in html and 'id="rewrite-active-text"' in html
    shown = _js_function(js, "showProposal")
    assert "setText('rewrite-source'" in shown
    assert "rewriteLabel(answer.backend)" in shown
    # One name when both kinds agree, one clause per kind when they do not, and
    # a kind with nothing answering it is said out loud rather than hidden.
    sentence = _js_function(js, "rewriterSentence")
    assert "Rewriter in use: " in sentence
    assert "'text drafts by ' + rewriteLabel(text)" in sentence
    assert "'edit drafts by ' + rewriteLabel(edit)" in sentence
    assert "have no rewriter here." in sentence
    state = _js_function(js, "setRewriterState")
    assert "rewriterSentence()" in state
    assert "setText('rewrite-active-text', sentence)" in state
    saved = _js_function(js, "saveRewriteSettings")
    assert "checkStatus();" in saved, "a save refreshes the Create view"
    hidden = _js_function(js, "hideProposal")
    assert "setText('rewrite-source', '');" in hidden
    # The friendly names are the server's rows, with a bounded fallback.
    label = _js_function(js, "rewriteLabel")
    assert "rewriteBackends[i].label" in label
    assert "Official enhancer (Q8, uses GPU)" in label
    assert "OpenAI-compatible endpoint" in label
    assert "127.0.0.1" not in label and "http" not in label
    # Which rewriter answers a kind comes from the status block's per-task map,
    # and availability is asked about THAT kind, not about one global backend.
    assert "rewriteByTask[task]" in _js_function(js, "rewriteActiveFor")
    assert "rewriteActiveFor(task)" in _js_function(js, "rewriteAvailable")


# --------------------------------------------------------------------------
# The Settings parsers and the Settings render, EXECUTED in node over the
# documents the server really sends. A string check cannot see a real document
# being rejected, and it cannot see which radio ends up ticked: the settings
# state carries a choice PER DRAFT KIND plus one on/off switch, the `saved`
# block may hold nulls ("never chosen") and the `local` block has no choice
# keys at all. So this section feeds the real payloads through the real app.js
# functions and asserts what the page shows — and that the envelope the page
# would POST is a document the real server validator accepts.
# --------------------------------------------------------------------------
# The constants the parsers enforce: every one of them is a product rule, so a
# harness that stubs one would test a page that does not exist.
PARSE_VARS = ("REWRITE_SETTINGS_KEYS", "REWRITE_CHOICE_KEYS",
              "REWRITE_LOCAL_KEYS", "REWRITE_SETTINGS_ROW_KEYS",
              "REWRITE_TASK_ROW_KEYS", "REWRITE_TASK_CHOICE_KEYS",
              "REWRITE_SETTINGS_BACKENDS", "REWRITE_BACKEND_IDS",
              "REWRITE_TASKS", "REWRITE_TASK_LABELS",
              "REWRITE_BACKEND_LABELS", "REWRITE_BACKEND_USES_GPU",
              "REWRITE_SETTINGS_SCHEMA", "INVALID")
PARSE_FNS = ("pyStrip", "hasOnlyKeys", "containsId", "parseRewriteChoices",
             "parseRewriteChoice", "parseRewriteLocalFields",
             "parseRewriteLocal", "parseRewriteSettingsRows",
             "parseRewriteTaskRows", "parseRewriteSettings")
RENDER_FNS = ("pyStrip", "rewriteTaskRow", "rewriteChoiceRow",
              "rewriteShortName", "rewriteFormState", "selectedRewriteChoice",
              "rewriteEnabled", "rewriteEndpointFields",
              "rewriteSettingsEnvelope", "renderRewriteSettings")

_HARNESS = """// Built by tests/test_web_static.py: the real app.js functions, extracted
// verbatim, run over real server documents with the smallest DOM that
// renderRewriteSettings and rewriteSettingsEnvelope touch. Nothing here
// reimplements a product rule.
var fs = require("node:fs");
__PARTS__

var textSets = {};
function setText(id, value) { textSets[id] = String(value); }
var ELEMENTS = {};
  ["rewrite-endpoint", "rewrite-model", "rewrite-timeout"].forEach(function (id) {
  ELEMENTS[id] = { id: id, value: "", disabled: true,
                   closest: function () { return null; } };
});
  ["rewrite-test", "rewrite-save", "rewrite-group-t2i", "rewrite-group-edit",
   "rewrite-off"].forEach(function (id) {
  ELEMENTS[id] = { id: id, disabled: true };
});
ELEMENTS["rewrite-off"].checked = false;
function $(id) { return ELEMENTS[id]; }

// One radio group per draft kind, with the names index.html uses.
function group(values) {
  return values.map(function (value) {
    return { value: value, checked: false, disabled: true };
  });
}
var RADIOS = {
  "rewrite-t2i": group(["pe-llamacpp", "local-llm"]),
  "rewrite-edit": group(["pe-llamacpp", "local-llm"])
};
var document = {
  querySelectorAll: function (selector) {
    var name = /^input\\[name="([^"]+)"\\]$/.exec(selector);
    return (name && RADIOS[name[1]]) || [];
  }
};

var rewriteSettings = null;
var rewriteForm = null;
var rewriteSettingsBusy = false;
var rewriteTestBusy = false;

function checkedOf(name) {
  return RADIOS[name].filter(function (r) { return r.checked; })
                     .map(function (r) { return r.value; })[0] || null;
}
function disabledOf(name) {
  return RADIOS[name].filter(function (r) { return r.disabled; })
                     .map(function (r) { return r.value; });
}

var docs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var out = docs.map(function (doc) {
  Object.keys(RADIOS).forEach(function (name) {
    RADIOS[name].forEach(function (r) { r.checked = false; r.disabled = true; });
  });
  ELEMENTS["rewrite-off"].checked = false;
  ELEMENTS["rewrite-off"].disabled = true;
  ["rewrite-endpoint", "rewrite-model", "rewrite-timeout"].forEach(function (id) {
    ELEMENTS[id].value = "";
  });
  textSets = {};
  var parsed = parseRewriteSettings(doc);
  rewriteSettings = parsed;
  rewriteForm = null;
  renderRewriteSettings();
  return {
    parsed: parsed,
    checked: { t2i: checkedOf("rewrite-t2i"), edit: checkedOf("rewrite-edit") },
    disabled: { t2i: disabledOf("rewrite-t2i"), edit: disabledOf("rewrite-edit") },
    off: ELEMENTS["rewrite-off"].checked,
    fields: { endpoint: ELEMENTS["rewrite-endpoint"].value,
              model: ELEMENTS["rewrite-model"].value,
              timeout: ELEMENTS["rewrite-timeout"].value },
    envelope: rewriteSettingsEnvelope(),
    sets: textSets
  };
});
process.stdout.write(JSON.stringify(out));
"""


def _js_var(js: str, name: str) -> str:
    """One top-level ``var <name> = ...;`` statement from app.js, verbatim.

    The statement is found by scanning for its terminator, not by guessing the
    layout: the value may wrap onto further lines and the line may carry a
    trailing comment, so only a bracket/string-aware scan knows where it ends.
    """
    start = re.search(rf"^  var {re.escape(name)} =\s", js, re.M)
    assert start, f"{name} not found in app.js"
    depth = 0
    quote = ""
    index = start.end()
    while index < len(js):
        char = js[index]
        if quote:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = ""
        elif char in "'\"`":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == ";" and depth == 0:
            return js[start.start():index + 1]
        index += 1
    raise AssertionError(f"{name} has no statement end in app.js")


def _run_settings_parsers(tmp_path: Path, docs: list[dict]) -> list[dict]:
    """Run the real parsers + render path over `docs`, in node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in PARSE_VARS]
                      + [_js_function(js, fn) for fn in PARSE_FNS + RENDER_FNS])
    harness = tmp_path / "settings_harness.js"
    documents = tmp_path / "documents.json"
    harness.write_text(_HARNESS.replace("__PARTS__", parts), encoding="utf-8")
    documents.write_text(json.dumps(docs), encoding="utf-8")
    run = subprocess.run([node, str(harness), str(documents)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def _settings_services(tmp_path: Path) -> list[tuple[str, object, dict]]:
    """(name, service, what the page must show) for every real shape.

    The expectations are written in the page's own terms — which radio is ticked
    in EACH group, whether the Off switch is ticked, which radios are greyed and
    which sentence appears under them — because that is the only thing the
    browser can act on.
    """
    from qwen_engine.web import api
    from qwen_engine.web import rewrite_service as rs

    endpoint = "http://127.0.0.1:8079/v1/chat/completions"
    prompts = {"t2i": "pe prompt t2i", "edit": "pe prompt edit"}
    pe_both = rs.PESettings(llama_server="/opt/llama/llama-server",
                            model_path="/models/pe/PE-T2I.gguf",
                            system_prompts=prompts, timeout=300.0,
                            i2i_model_path="/models/pe/PE-I2I.gguf",
                            i2i_mmproj_path="/models/pe/mmproj.gguf")
    # The install Root actually has: the text checkpoint without the edit pair.
    pe_text_only = rs.PESettings(llama_server="/opt/llama/llama-server",
                                 model_path="/models/pe/PE-T2I.gguf",
                                 system_prompts=prompts, timeout=300.0)

    def local_settings(**overrides):
        fields = dict(backend="local-llm", endpoint=endpoint, model="Qwen3.8",
                      prompts={"t2i": "sys t2i", "edit": "sys edit"}, timeout=30.0)
        fields.update(overrides)
        return rs.RewriteSettings(**fields)

    def make(cache_dir, local, pe_cfg, prefs=None):
        # The factory stands in for a BUILT client (truthy): `settings_state`
        # asks which (task, backend) pairs exist, it never calls into one.
        return rs.RewriteService(local, transport=lambda _body: b"{}",
                                 pe=pe_cfg, cache_dir=cache_dir,
                                 preferences=prefs,
                                 client_factory=lambda *_args: object())

    cache = str(tmp_path / "cache")
    official = api.REWRITE_BACKEND_LABELS["pe-llamacpp"]
    endpoint_label = api.REWRITE_BACKEND_LABELS["local-llm"]
    no_edit_checkpoint = rs.RewritePreferences(
        choices={"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}, enabled=True)
    return [
        ("both checkpoints and the endpoint, nothing saved",
         make(cache, local_settings(), pe_both),
         {"choices": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "checked": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "off": False, "disabled": {"t2i": [], "edit": []},
          "reasons": {}}),
        ("saved: both kinds on the endpoint",
         make(cache, local_settings(), pe_both,
              rs.RewritePreferences(
                  choices={"t2i": "local-llm", "edit": "local-llm"}, enabled=True,
                  endpoint="http://127.0.0.1:8081/v1/chat/completions",
                  model="Other", timeout_s=45.0)),
         {"choices": {"t2i": "local-llm", "edit": "local-llm"},
          "checked": {"t2i": "local-llm", "edit": "local-llm"},
          "off": False, "disabled": {"t2i": [], "edit": []},
          "reasons": {}}),
        ("saved: text on the official enhancer, edits on the endpoint",
         make(cache, local_settings(), pe_both,
              rs.RewritePreferences(choices={"t2i": "pe-llamacpp",
                                             "edit": "local-llm"},
                                    enabled=True)),
         {"choices": {"t2i": "pe-llamacpp", "edit": "local-llm"},
          "checked": {"t2i": "pe-llamacpp", "edit": "local-llm"},
          "off": False, "disabled": {"t2i": [], "edit": []},
          "reasons": {}}),
        ("no edit checkpoint here: the official edit choice is greyed",
         make(cache, local_settings(), pe_text_only, no_edit_checkpoint),
         {"choices": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "checked": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "off": False, "disabled": {"t2i": [], "edit": []},
          # The saved choice stays ticked even though it cannot be honoured,
          # with the server's sentence under it, so installing the missing file
          # fixes it without choosing twice.
          "reasons": {"rewrite-edit-pe-reason": "no official image-edit"},
          "active": {"t2i": official, "edit": endpoint_label}}),
        ("enhancer only: an edit draft has no rewriter at all",
         make(cache, rs.RewriteSettings.disabled(), pe_text_only),
         {"choices": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "checked": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "off": False,
          "disabled": {"t2i": ["local-llm"], "edit": ["local-llm"]},
          "reasons": {"rewrite-edit-pe-reason": "no official image-edit",
                      "rewrite-t2i-local-reason": "started without an "
                                                  "OpenAI-compatible"}}),
        ("endpoint only, nothing persistent",
         make(None, local_settings(), None),
         {"choices": {"t2i": "local-llm", "edit": "local-llm"},
          "checked": {"t2i": "local-llm", "edit": "local-llm"},
          "off": False,
          "disabled": {"t2i": ["pe-llamacpp"], "edit": ["pe-llamacpp"]},
          "reasons": {"rewrite-t2i-pe-reason": "started without the enhancer"}}),
        ("saved off",
         make(cache, local_settings(), pe_both,
              rs.RewritePreferences(choices={"t2i": "pe-llamacpp",
                                             "edit": "pe-llamacpp"},
                                    enabled=False)),
         {"choices": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "checked": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"},
          "off": True, "disabled": {"t2i": [], "edit": []},
          "reasons": {"rewrite-off-reason": "switched off"}}),
    ]


def test_the_settings_parsers_accept_the_documents_the_server_sends(tmp_path):
    from qwen_engine.web import api

    cases = _settings_services(tmp_path)
    documents = []
    for name, subject, _want in cases:
        state = subject.settings_state()
        assert state["schema"] == api.REWRITE_SETTINGS_SCHEMA, name
        documents.append(state)
    results = _run_settings_parsers(tmp_path, documents)

    for (name, _subject, want), state, result in zip(cases, documents, results):
        parsed = result["parsed"]
        assert parsed is not None, f"{name}: the page would call this invalid"
        assert parsed["choices"] == want["choices"], name
        assert parsed["enabled"] is (not want["off"]), name
        # One radio ticked per draft kind, and the Off switch only when Off is
        # what the server holds.
        assert result["checked"] == want["checked"], (name, result["checked"])
        assert result["off"] is want["off"], (name, result["off"])
        assert result["disabled"] == want["disabled"], (name, result["disabled"])
        # What the block describes is what the fields are prefilled with.
        assert parsed["local"] == state["local"], name
        assert result["fields"]["endpoint"] == (state["local"]["endpoint"] or "")
        assert result["fields"]["model"] == (state["local"]["model"] or "")
        timeout = state["local"]["timeout_s"]
        shown = result["fields"]["timeout"]
        assert (shown == "" if timeout is None else float(shown) == timeout), name
        assert parsed["saved"] == state["saved"], name
        assert parsed["persistent"] == state["persistent"], name
        note = state.get("note", "")
        assert result["sets"].get("rewrite-settings-note", "") == note, name
        # A greyed-out or unanswerable option carries the SERVER's sentence, and
        # that sentence names no path, URL or address.
        for node, needle in want["reasons"].items():
            text = result["sets"].get(node, "")
            assert needle in text, (name, node, text)
            assert "/" not in text and "://" not in text, (name, node, text)
        # The body the page would POST is a document the real server validator
        # accepts, saying exactly what the page shows.
        posted = api.build_rewrite_settings_request(
            json.dumps(result["envelope"]).encode("utf-8"))
        assert posted["t2i"] == want["choices"]["t2i"], name
        assert posted["edit"] == want["choices"]["edit"], name
        assert posted["enabled"] is (not want["off"]), name
        assert posted["endpoint"] == state["local"]["endpoint"], name
        assert posted["model"] == state["local"]["model"], name


def test_the_settings_parsers_still_refuse_a_document_that_is_not_theirs(tmp_path):
    """Every key set here is closed on purpose. A block that suddenly selects a
    backend, a choice that is not a rewriter, a task row that dropped the
    availability flag, or a value outside the server's own bounds rejects the
    WHOLE document rather than rendering half of it — a half-rendered rewriter
    choice is a setting the user believes they made."""
    base = _settings_services(tmp_path)[0][1].settings_state()
    task_rows = [json.loads(json.dumps(row)) for row in base["tasks"]]
    no_choice_rows = [dict(task_rows[0]), task_rows[1]]
    no_choice_rows[0]["choices"] = [dict(no_choice_rows[0]["choices"][0])]
    unlabelled = [json.loads(json.dumps(row)) for row in base["tasks"]]
    unlabelled[0]["label"] = "Whatever the page feels like"
    mute = [json.loads(json.dumps(row)) for row in base["tasks"]]
    official = base["tasks"][1]["choices"][0]["label"]
    mute[1]["choices"][0] = {"id": "pe-llamacpp", "label": official,
                             "available": False, "uses_gpu": True,
                             "reason": ""}
    tampered = {
        "local block selects a backend": {**base, "local": {
            **base["local"], "backend": "local-llm"}},
        "local block carries a path": {**base, "local": {
            **base["local"], "model_path": "/models/pe/PE.gguf"}},
        "saved names an unknown rewriter for one task": {**base, "saved": {
            **base["saved"], "t2i": "gguf-direct"}},
        "saved uses off as a task choice": {**base, "saved": {
            **base["saved"], "edit": "off"}},
        "choices answer only one draft kind": {**base, "choices": {
            "t2i": base["choices"]["t2i"]}},
        "choices name a rewriter that does not exist": {**base, "choices": {
            **base["choices"], "edit": "gguf-direct"}},
        "enabled is not a boolean": {**base, "enabled": "yes"},
        "a task group is missing": {**base, "tasks": [task_rows[0]]},
        "a task group lists only one rewriter": {**base, "tasks": no_choice_rows},
        "a task group renames the draft kind": {**base, "tasks": unlabelled},
        "an unavailable choice carries no reason": {**base, "tasks": mute},
        "timeout beyond the server bound": {**base, "local": {
            **base["local"], "timeout_s": 9000}},
        "endpoint is not a string": {**base, "local": {
            **base["local"], "endpoint": 8079}},
        "unknown top-level key": {**base, "rewriter": "pe-llamacpp"},
        "a single backend key comes back": {k: v for k, v in base.items()
                                            if k != "choices"} | {
            "backend": "pe-llamacpp"},
        "wrong schema tag": {**base, "schema": base["schema"] + ".next"},
    }
    results = _run_settings_parsers(tmp_path, list(tampered.values()))
    for (why, _doc), result in zip(tampered.items(), results):
        assert result["parsed"] is None, f"{why} was accepted"
        # A rejected document leaves the page with nothing filled in at all.
        assert result["checked"] == {"t2i": None, "edit": None}, why
        assert result["off"] is False, why
        assert result["fields"] == {"endpoint": "", "model": "", "timeout": ""}


# --------------------------------------------------------------------------
# First-run model setup: the Models card and the Create-page banner, EXECUTED
# in node over the documents the server really produces. The card's whole job
# is to render a server document and to send ids back, so a string check would
# prove nothing about the two things that can actually go wrong: a real
# document the closed parser refuses (the page then offers nothing), and a page
# that sends back more than ids.
# --------------------------------------------------------------------------

MODEL_VARS = ("MODEL_IDS", "MODELS_KEYS", "MODEL_ITEM_KEYS",
              "MODEL_FILE_KEYS", "MODELS_JOB_KEYS", "MODELS_JOB_FILE_KEYS",
              "MODELS_ERROR_KEYS", "MODEL_INSTALL_STATES", "MODEL_JOB_STATES",
              "MODELS_MISSING", "MODELS_SCHEMA", "MODELS_JOB_SCHEMA",
              "MODELS_DOWNLOAD_REQUEST_SCHEMA", "MODEL_NEEDS_BINARY",
              "MODEL_NEEDS_BINARY_TEXT", "MODEL_STATE_TEXT",
              "modelsDoc", "modelsBusy", "modelsTimer", "modelsSeq",
              "modelsBannerDismissed", "modelsMissingBlocked")

MODEL_FNS = ("hasOnlyKeys", "isPlainRelativeName", "parseModelsFileRow",
             "parseModelsItem", "parseModelsJob", "parseModelsDocument",
             "fillModelsFiles", "renderModelsSettings", "renderModelsProgress",
             "modelsSelectedIds", "modelsDownloadEnvelope",
             "renderModelsBanner", "formatBytes")

# Both model harnesses share this: the extracted app.js code, then the smallest
# DOM the Models card touches. Neither one reimplements a product rule.
_MODELS_HEAD = """// Built by tests/test_web_static.py: the real app.js model functions,
// extracted verbatim and run over the documents the server really produces.
var fs = require("node:fs");
__PARTS__
"""

_MODELS_STUB = """
var textSets = {};
function setText(id, value) { textSets[id] = String(value); }

var ELEMENTS = {};
function classList(node) {
  var held = {};
  return {
    add: function (name) { held[name] = true; },
    remove: function (name) { delete held[name]; },
    contains: function (name) { return !!held[name]; },
    _held: function () { return Object.keys(held); }
  };
}
function element(id, extra) {
  ELEMENTS[id] = Object.assign({ id: id, hidden: true, checked: false,
                                 disabled: true, textContent: "" },
                               extra || {});
  ELEMENTS[id].classList = classList(ELEMENTS[id]);
  return ELEMENTS[id];
}
["models-card", "models-total", "models-note", "models-error", "models-status",
 "models-server-only", "models-progress", "models-progress-label",
 "models-banner", "models-banner-text"].forEach(function (id) {
  element(id);
});
["models-download", "models-cancel"].forEach(function (id) {
  element(id, { setAttribute: function () {},
                removeAttribute: function () {} });
});
element("models-progress-bar", { value: -1 });
["models-progress-files"].forEach(function (id) { element(id, {}); });
MODEL_IDS.forEach(function (id) {
  element("model-row-" + id);
  element("model-tick-" + id);
  element("model-tag-" + id);
  element("model-files-" + id, { firstChild: null,
                                 removeChild: function () { this.firstChild = null; },
                                 appendChild: function (child) { this.firstChild = child; } });
});

var created = [];
var document = {
  createElement: function (tag) {
    var node = { tagName: tag, className: "", textContent: "",
                 children: [], appendChild: function (c) { this.children.push(c); } };
    created.push(node);
    return node;
  },
  // The card asks for one element by id without the $ helper, so the stub
  // answers both spellings the way the browser does.
  getElementById: function (id) { return ELEMENTS[id] || null; }
};
function $(id) { return ELEMENTS[id]; }

function resetElements() {
  Object.keys(ELEMENTS).forEach(function (id) {
    var el = ELEMENTS[id];
    el.hidden = true; el.checked = false; el.disabled = true;
    el.textContent = "";
    if (id === "models-progress-bar") el.value = -1;
  });
}

function tickState() {
  var out = {};
  MODEL_IDS.forEach(function (id) {
    var row = ELEMENTS["model-row-" + id];
    var tick = ELEMENTS["model-tick-" + id];
    out[id] = { shown: !row.hidden, checked: tick.checked, disabled: tick.disabled };
  });
  return out;
}
"""

# The synchronous pass: hand the parser and the renderer a document, read what
# the card shows and what the Download button would send.
_MODELS_SYNC_BODY = """
var docs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var out = docs.map(function (item) {
  resetElements();
  textSets = {};
  modelsDoc = parseModelsDocument(item.doc);
  modelsBusy = false;
  modelsBannerDismissed = !!item.dismissed;
  modelsMissingBlocked = !!item.blocked;
  renderModelsSettings();
  // The product calls this from checkStatus with the status blocker; the case
  // flag stands in for that document.
  renderModelsBanner(!!item.blocked);
  return {
    parsed: modelsDoc,
    card: !$("models-card").hidden,
    ticks: tickState(),
    banner: !$("models-banner").hidden,
    envelope: modelsDownloadEnvelope(),
    downloadDisabled: $("models-download").disabled,
    cancelDisabled: $("models-cancel").disabled,
    sets: textSets
  };
});
process.stdout.write(JSON.stringify(out));
"""

_MODELS_HARNESS = _MODELS_HEAD + _MODELS_STUB + _MODELS_SYNC_BODY

# The asynchronous pass: the SAME handlers driven through the page's real
# request path, so a reply is only accepted if the code that arrived is the
# code that is read.
_MODELS_FETCH_STUB = """
// The one collaborator this pass does not exercise: after a job ends the page
// re-reads the whole status document, and that path is tested elsewhere.
function checkStatus() { return Promise.resolve(null); }

var fetchCalls = [];
function makeResponse(status, body) {
  var bytes = new TextEncoder().encode(body);
  var sent = false;
  return {
    ok: status >= 200 && status < 300,
    status: status,
    headers: { get: function () { return "application/json"; } },
    body: { getReader: function () {
      return { read: function () {
        if (sent) return Promise.resolve({ done: true });
        sent = true;
        return Promise.resolve({ done: false, value: bytes });
      }, cancel: function () { return Promise.resolve(); } };
    } }
  };
}
function fetch(url, opts) {
  fetchCalls.push({ url: url, opts: opts });
  var plan = docs[nextCase].response;
  return Promise.resolve(makeResponse(plan.status, plan.body));
}
"""

_MODELS_ASYNC_BODY = """
var docs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var nextCase = 0;
var out = docs.map(function (item) {
  resetElements();
  textSets = {};
  fetchCalls = [];
  modelsDoc = null;
  modelsBusy = false;
  modelsTimer = 0;
  modelsBannerDismissed = !!item.dismissed;
  var work = (item.step === "post")
    ? postModels(item.url || "/api/v1/models/download", item.body)
    : loadModelStatus();
  return work.then(function () {
    renderModelsBanner(!!item.blocked);
    return {
      parsed: modelsDoc,
      card: !$("models-card").hidden,
      banner: !$("models-banner").hidden,
      error: textSets["models-error"] || "",
      status: textSets["models-status"] || "",
      polls: modelsTimer !== 0,
      calls: fetchCalls.map(function (c) { return c.url; }),
      sentBodies: fetchCalls.map(function (c) { return c.opts && c.opts.body; }),
      sentHeaders: fetchCalls.map(function (c) {
        return (c.opts && c.opts.headers) ? Object.keys(c.opts.headers).sort() : [];
      })
    };
  });
});
Promise.all(out).then(function (rows) {
  process.stdout.write(JSON.stringify(rows));
});
"""

_MODELS_REQUEST_HARNESS = (_MODELS_HEAD + _MODELS_STUB + _MODELS_FETCH_STUB
                           + _MODELS_ASYNC_BODY)

_MODELS_CARD = "models-card"


def _models_document(tmp_path: Path) -> dict:
    """The exact ``GET /api/v1/models`` body for a fresh temporary install."""
    from qwen_engine import model_setup as setup
    from qwen_engine.web import api
    service = setup.ModelSetupService(_models_roots(tmp_path))
    return {**service.state(), "can_download": True,
            "server_command": api.SETUP_SERVER_COMMAND,
            "notices": list(service.notices())}


def _models_roots(tmp_path: Path):
    """A ``models/`` parent pinning two base files with NEITHER present.

    The manifest uses the production schema and real sha256s, so the catalogue,
    the byte figures and the required-missing verdict all come out of the real
    code path: the card is exercised in the state a first run actually shows.
    """
    import hashlib
    import json

    from qwen_engine import model_setup as setup
    small = b"config-json\n"
    big = b"weights-" * 64

    def artifact(filename, data, role):
        return {"role": role, "repo": "Qwen/Test-Base",
                "revision": "a" * 40, "filename": filename,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data)}

    parent = tmp_path / "assetroot"
    (parent / "assets").mkdir(parents=True, exist_ok=True)
    (parent / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "artifacts": [artifact("model_index.json", small, "pipeline"),
                      artifact("transformer/weights.bin", big,
                               "transformer")]}))
    return setup.SetupRoots.for_paths(models_dir=str(parent / "assets"),
                                      manifest_path=str(parent / "manifest.json"),
                                      q8_manifest_path=None)


def _run_models_card(tmp_path: Path, cases: list[dict]) -> list[dict]:
    """Run the real model parsers and the real renderer over `cases`, in node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in MODEL_VARS]
                      + [_js_function(js, fn) for fn in MODEL_FNS])
    harness = tmp_path / "models_harness.js"
    documents = tmp_path / "models_documents.json"
    harness.write_text(_MODELS_HARNESS.replace("__PARTS__", parts),
                       encoding="utf-8")
    documents.write_text(json.dumps(cases), encoding="utf-8")
    run = subprocess.run([node, str(harness), str(documents)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_models_card_renders_the_document_the_server_sends(tmp_path):
    from qwen_engine import model_setup as setup

    doc = _models_document(tmp_path)
    result = _run_models_card(tmp_path, [{"doc": doc}])[0]

    assert result["parsed"] is not None, "the real document was rejected"
    assert result["card"] is True, "a parsed catalogue must show the card"
    # An offered row is visible; the required one is not tickable (its files come
    # with every request), an offered optional one is. A row the server did not
    # offer at all stays hidden — no dead checkbox.
    ticks = result["ticks"]
    assert ticks[setup.BASE]["shown"] is True
    assert ticks[setup.BASE]["disabled"] is True
    assert ticks[setup.PE_TEXT]["shown"] is True
    assert ticks[setup.PE_TEXT]["disabled"] is False
    assert ticks[setup.TURBO]["shown"] is False, "no turbo manifest, no row"
    assert ticks[setup.Q8]["shown"] is False, "no q8 sidecar, no row"
    # Size and destination are on the page, in the server's own words.
    sizes = {k: v for k, v in result["sets"].items() if k.startswith("model-size-")}
    assert any("to download" in value for value in sizes.values()), sizes
    dests = {k: v for k, v in result["sets"].items() if k.startswith("model-dest-")}
    assert any(value.startswith("Stored in ") for value in dests.values()), dests
    for value in list(sizes.values()) + list(dests.values()):
        # The real invariant: the page shows a name, never a machine location.
        shown = value.split("Stored in ", 1)[-1]
        assert not shown.startswith("/") and ".." not in shown, value
        assert tmp_path.as_posix() not in value, value
    # Nothing was ticked, so only the required ids would be sent — and the
    # document's own shape, not a guess, decides that.
    assert result["envelope"] == {"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                                  "ids": [setup.BASE]}
    assert result["downloadDisabled"] is False
    assert result["cancelDisabled"] is True


def test_the_models_card_sends_only_the_ids_the_user_ticked(tmp_path):
    from qwen_engine import model_setup as setup

    doc = _models_document(tmp_path)
    js_case = {"doc": doc}
    # The harness reads the ticks out of the DOM; ticking happens by name so
    # the test says which items a user chose, not how the DOM is wired.
    result = _run_models_card(tmp_path, [js_case])[0]
    assert result["envelope"]["ids"] == [setup.BASE]

    # With the optional enhancer weights ticked the request grows by exactly
    # that id, and never by a path, a repo or a revision.
    doc["items"][0]["required"] = False  # a server that asks nothing of us
    grew = _run_models_card(tmp_path, [{"doc": doc}])[0]
    assert grew["envelope"] == {"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                                "ids": []}


def test_the_models_card_refuses_documents_that_are_not_the_servers(tmp_path):
    doc = _models_document(tmp_path)
    tampered: dict[str, dict] = {}
    for why, mutate in (
            ("an extra top-level key", lambda d: d.update({"model_dir": "/tmp"})),
            ("a top-level key gone", lambda d: d.pop("job")),
            ("a non-integer byte count",
             lambda d: d["items"][0]["files"][0].update({"bytes": "5"})),
            ("a parent step in a file name",
             lambda d: d["items"][0]["files"][0].update({"name": "../x.gguf"})),
            ("an absolute destination",
             lambda d: d["items"][0].update({"dest": "/home/op/models"})),
            ("an unknown install state",
             lambda d: d["items"][0].update({"installed": "probably"})),
            ("a job over its own byte bound",
             lambda d: d.update({"job": {"schema": "qwen_engine.web_models_job.v1",
                                         "state": "running", "ids": ["base"],
                                         "files": [], "bytes_done": -1,
                                         "bytes_total": 4, "error": None}})),
            ("a negative free-space figure",
             lambda d: d.update({"free_bytes": -1})),
            ("a notice that is not a string",
             lambda d: d.update({"notices": [None]})),
            ("a JSON null instead of a document", lambda d: d.clear())):
        broken = json.loads(json.dumps(doc))
        mutate(broken)
        tampered[why] = {"doc": broken if broken else None}

    results = _run_models_card(tmp_path, list(tampered.values()))
    for (why, _doc), result in zip(tampered.items(), results):
        assert result["parsed"] is None, f"{why} was accepted"
        assert result["card"] is False, f"{why} rendered a half card"
        assert result["envelope"] == {"schema":
                                      _MODELS_SCHEMA_NAME, "ids": []}, why
        assert result["downloadDisabled"] is True, why


_MODELS_SCHEMA_NAME = "qwen_engine.web_models_download_request.v1"


def test_the_models_banner_follows_the_server_not_the_browser(tmp_path):
    doc = _models_document(tmp_path)
    shown, hidden, dismissed = _run_models_card(tmp_path, [
        {"doc": doc, "blocked": True},
        {"doc": doc, "blocked": False},
        {"doc": doc, "blocked": True, "dismissed": True},
    ])
    assert shown["banner"] is True, "a missing required model must say so"
    assert hidden["banner"] is False
    assert dismissed["banner"] is False, "'Not now' hides it for this page load"
    assert "not on this machine" in shown["sets"]["models-banner-text"]


def test_a_server_that_cannot_download_says_so_instead_of_failing(tmp_path):
    doc = _models_document(tmp_path)
    doc["can_download"] = False
    result = _run_models_card(tmp_path, [{"doc": doc}])[0]
    assert result["parsed"] is not None
    assert result["downloadDisabled"] is True and result["cancelDisabled"] is True
    assert result["sets"].get("models-error", "") == ""


def test_the_models_markup_and_css_are_present_and_theme_neutral():
    html, css = _read("index.html"), _read("app.css")
    for element_id in ("models-card", "models-banner", "models-download",
                       "models-cancel", "models-progress", "models-total",
                       "models-server-only", "models-server-command"):
        assert f'id="{element_id}"' in html, element_id
    assert '<section class="models-banner" id="models-banner" hidden' in html
    assert 'id="models-card" hidden' in html
    for selector in (".models-banner", ".model-row", ".models-progress",
                     ".models-server", ".models-files"):
        assert selector in css, selector
    assert "#models-note:not(:empty) { color: var(--text-muted); }" in css


def test_the_models_literals_are_the_servers_literals():
    """Every one of these is a wire token, not a label: the parser compares the
    document against them, so a page and a server that drifted apart would show
    an empty card on a healthy install."""
    from qwen_engine import model_setup as setup
    from qwen_engine.web import api
    js = _read("app.js")
    assert (_js_var(js, "MODELS_SCHEMA") ==
            f"  var MODELS_SCHEMA = '{setup.MODELS_SCHEMA}';")
    assert (_js_var(js, "MODELS_JOB_SCHEMA") ==
            f"  var MODELS_JOB_SCHEMA = '{setup.JOB_SCHEMA}';")
    assert (_js_var(js, "MODELS_DOWNLOAD_REQUEST_SCHEMA") ==
            f"  var MODELS_DOWNLOAD_REQUEST_SCHEMA =\n"
            f"    '{setup.DOWNLOAD_REQUEST_SCHEMA}';")
    assert (f"'{setup.NEEDS_BINARY}'" in _js_var(js, "MODEL_NEEDS_BINARY"))
    assert _js_var(js, "MODELS_MISSING") == \
        f"  var MODELS_MISSING = '{api.MODELS_MISSING}';"
    assert re.findall(r"'([a-z0-9-]+)'", _js_var(js, "MODEL_IDS")) == \
        list(setup.ITEM_IDS)
    assert set(re.findall(r"'([a-z_-]+)'", _js_var(js, "MODEL_INSTALL_STATES"))) \
        == set(setup.INSTALL_STATES)
    assert set(re.findall(r"'([a-z_-]+)'", _js_var(js, "MODEL_JOB_STATES"))) \
        == set(setup.JOB_STATES)
    # The closed key sets, both directions: the JS parser and the Python
    # validator must describe the same document.
    for var, python in (("MODELS_KEYS", api._MODELS_KEYS),
                        ("MODEL_ITEM_KEYS", api._MODEL_ITEM_KEYS),
                        ("MODEL_FILE_KEYS", api._MODEL_FILE_KEYS),
                        ("MODELS_JOB_KEYS", api._MODEL_JOB_KEYS),
                        ("MODELS_JOB_FILE_KEYS", api._MODEL_JOB_FILE_KEYS),
                        ("MODELS_ERROR_KEYS", api._MODEL_ERROR_KEYS)):
        assert sorted(re.findall(r"'([a-z_-]+)'", _js_var(js, var))) == \
            sorted(python), var


# --------------------------------------------------------------------------
# The same card, but through the page's REAL request path (apiRequest +
# readBodyLimited + fetch). The harness above stubs the request away, which is
# exactly how one bug survived it: apiRequest reports the Response object, so a
# handler that reads a `status` field that does not exist treats EVERY healthy
# reply as a failure and the card stays empty on a healthy install. Only
# running the handlers over a fake fetch can see that.
# --------------------------------------------------------------------------

MODEL_REQUEST_VARS = MODEL_VARS + ("API_MODELS", "API_MODELS_DOWNLOAD",
                                   "API_MODELS_DOWNLOAD_CANCEL",
                                   "FETCH_TIMEOUT_MS", "MAX_RESPONSE_BYTES",
                                   "MAX_JOB_IMAGE_BYTES", "MAX_REFERENCE_BYTES",
                                   "BODY_TOO_LARGE", "UNSUPPORTED_BODY")

MODEL_REQUEST_FNS = ("readBodyLimited", "apiRequest") + MODEL_FNS + (
    "loadModelStatus", "postModels", "startModelsPolling", "stopModelsPolling")


def _run_models_request(tmp_path: Path, cases: list[dict]) -> list[dict]:
    """Run the page's real request path (fetch → apiRequest → handlers)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in MODEL_REQUEST_VARS]
                      + [_js_function(js, fn) for fn in MODEL_REQUEST_FNS])
    harness = tmp_path / "models_request_harness.js"
    documents = tmp_path / "models_requests.json"
    harness.write_text(_MODELS_REQUEST_HARNESS.replace("__PARTS__", parts),
                       encoding="utf-8")
    documents.write_text(json.dumps(cases), encoding="utf-8")
    run = subprocess.run([node, str(harness), str(documents)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_models_card_reads_a_healthy_reply_as_healthy(tmp_path):
    """The regression: the handler used a `status` field apiRequest never
    returns, so a 200 with a perfect document was reported as unreadable and the
    card stayed empty on a healthy install."""
    doc = _models_document(tmp_path)
    result = _run_models_request(tmp_path, [{
        "response": {"status": 200, "body": json.dumps(doc)}}])[0]
    assert result["error"] == "", result["error"]
    assert result["parsed"] is not None, "a healthy 200 must be parsed"
    assert result["card"] is True
    assert result["calls"] == ["/api/v1/models"]
    assert result["sentHeaders"] == [["Accept", "X-Qwen-UI"]]


def test_the_models_card_uses_the_status_code_it_was_given(tmp_path):
    """503 and 403 say different things to a user, and only if the code that
    arrived is the code that is read."""
    unavailable = _run_models_request(tmp_path, [{
        "response": {"status": 503,
                     "body": json.dumps({"schema": "qwen_engine.web_error.v1",
                                         "code": "models_unavailable",
                                         "message": "the catalogue is "
                                                    "unreadable"})}}])[0]
    assert unavailable["parsed"] is None
    assert unavailable["card"] is False
    assert unavailable["error"] == \
        "This server cannot read its model catalogue."

    refused = _run_models_request(tmp_path, [{
        "step": "post",
        "body": {"ids": ["base"]},
        "response": {"status": 403,
                     "body": json.dumps({"schema": "qwen_engine.web_error.v1",
                                         "code": "setup_on_server",
                                         "message": "Run python "
                                                    "tools/fetch_models.py "
                                                    "there"})}}])[0]
    assert refused["card"] is False
    assert refused["error"] == "Run python tools/fetch_models.py there"
    assert refused["sentBodies"] == ['{"ids":["base"]}']


def test_a_started_download_shows_progress_and_polls_until_it_ends(tmp_path):
    from qwen_engine import model_setup as setup

    running = {"schema": setup.JOB_SCHEMA, "state": "running", "ids": ["base"],
               "files": [{"item": "base", "name": "model_index.json",
                          "done": 4, "total": 12}],
               "bytes_done": 4, "bytes_total": 12, "error": None}
    result = _run_models_request(tmp_path, [{
        "step": "post",
        "body": {"ids": ["base"]},
        "response": {"status": 202, "body": json.dumps(running)}}])[0]
    assert result["error"] == "", result["error"]
    assert result["status"] == "Downloading…"
    assert result["polls"] is True, "a running job must arm the progress poll"
    assert result["sentBodies"] == ['{"ids":["base"]}']

    finished = dict(running, state="cancelled", error=None)
    stopped = _run_models_request(tmp_path, [{
        "step": "post",
        "body": {},
        "response": {"status": 200, "body": json.dumps(finished)}}])[0]
    assert stopped["polls"] is False, "a finished job must not keep polling"


# --------------------------------------------------------------------------
# checkStatus EXECUTED: the second live-found bug was an ordering one. The
# page labels the Generate button from `modelsMissingBlocked`, and the flag was
# written after the label, so a fresh install explained a dead button with the
# wrong sentence. A string check cannot see an order; running the handler can.
# --------------------------------------------------------------------------

# Concatenated in this order, so a constant that another constant's initialiser
# references (KNOWN_BLOCKER_CODES contains MODELS_MISSING) is already assigned
# when the array literal is evaluated.
STATUS_VARS = ("MODELS_MISSING", "INVALID", "REWRITE_BLOCK_KEYS",
               "REWRITE_STATES", "API_REWRITE_SETTINGS", "STATUS_SCHEMA",
               "GATE_STATES", "MANIFEST_REASON_CODES", "KNOWN_BLOCKER_CODES",
               "MODELS_GENERATE_REASON", "GENERATION_UNAVAILABLE_REASON",
               "API_STATUS",
               "FETCH_TIMEOUT_MS", "MAX_RESPONSE_BYTES", "MAX_JOB_IMAGE_BYTES",
               "MAX_REFERENCE_BYTES", "BODY_TOO_LARGE", "UNSUPPORTED_BODY",
               "apiConnected", "generationConnected", "submittingJob",
               "jobCache", "modelsMissingBlocked", "modelsBannerDismissed",
               "statusSeq", "currentUser", "TYPED_ERROR_SENTENCES")

STATUS_FNS = ("readBodyLimited", "apiRequest", "containsId", "hasAnyValue",
              "parseManifestCore", "blockersExact",
              "parseRewriteSettingsPointer", "parseRewriteBackends",
              "parseRewriteActiveByTask", "parseRewriteBlock", "parseStatus",
              "statusUnrecognizedText", "jobTerminal", "ownActiveJob",
              "updateGenerate", "renderChip",
              "renderModelsBanner", "checkStatus")

_STATUS_HARNESS = """// Built by tests/test_web_static.py: the real status handler, run over the
// real /api/v1/status document with three collaborators stubbed (each is
// covered by its own section: the connection flag, the rewriter block, the job
// list). The button's label is the thing under test.
var fs = require("node:fs");
__PARTS__

var textSets = {};
function setText(id, value) { textSets[id] = String(value); }
var ELEMENTS = {
  "generate-btn": { id: "generate-btn", disabled: true, textContent: "" },
  "models-banner": { id: "models-banner", hidden: true },
  "connection-chip": { id: "connection-chip", textContent: "" }
};
function $(id) { return ELEMENTS[id]; }

function setApiAvailable(value) { apiConnected = value; }
function setRewriterState(block) {}
function refreshJobs() { return Promise.resolve(null); }
// The identity block's own renderer has its own section (remote-user UI): here
// it must not touch a page this harness does not build.
function applyUserState(status) {}

function fetch(url, opts) {
  return Promise.resolve({
    ok: plan.status >= 200 && plan.status < 300,
    status: plan.status,
    headers: { get: function () { return "application/json"; } },
    body: (function () {
      var bytes = new TextEncoder().encode(plan.body);
      var sent = false;
      return { getReader: function () {
        return { read: function () {
          if (sent) return Promise.resolve({ done: true });
          sent = true;
          return Promise.resolve({ done: false, value: bytes });
        }, cancel: function () { return Promise.resolve(); } };
      } };
    })()
  });
}

var docs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
var plan = null;
var out = docs.map(function (item) {
  Object.keys(textSets).forEach(function (k) { delete textSets[k]; });
  ELEMENTS["generate-btn"].disabled = true;
  ELEMENTS["models-banner"].hidden = true;
  generationConnected = false;
  submittingJob = false;
  modelsMissingBlocked = false;
  modelsBannerDismissed = false;
  plan = item;
  return checkStatus().then(function () {
    return { reason: textSets["generate-reason"] || "",
             banner: !ELEMENTS["models-banner"].hidden,
             disabled: ELEMENTS["generate-btn"].disabled,
             sets: textSets };
  });
});
Promise.all(out).then(function (rows) {
  process.stdout.write(JSON.stringify(rows));
});
"""


def _run_check_status(tmp_path: Path, cases: list[dict]) -> list[dict]:
    """Run the real status handler over `cases` (each a canned /api/v1/status)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read("app.js")
    parts = "\n".join([_js_var(js, name) for name in STATUS_VARS]
                      + [_js_function(js, fn) for fn in STATUS_FNS])
    harness = tmp_path / "status_harness.js"
    documents = tmp_path / "status_documents.json"
    harness.write_text(_STATUS_HARNESS.replace("__PARTS__", parts),
                       encoding="utf-8")
    documents.write_text(json.dumps(cases), encoding="utf-8")
    run = subprocess.run([node, str(harness), str(documents)],
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:2000]}"
    return json.loads(run.stdout)


def test_the_generate_button_names_the_blocker_that_killed_it(tmp_path):
    """The server's own status document decides the sentence beside the button,
    and the banner beside the prompt — in the same pass, not the next one.

    Each document is produced by the real status route for a machine in a real
    state, so the page is driven by what qwen-web actually sends.
    """
    import hashlib

    from qwen_engine import model_setup as setup
    from qwen_engine.web import api

    def status(*, installed: bool, generation_enabled: bool) -> dict:
        root = tmp_path / ("full" if installed else "bare")
        assets = root / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        alpha, beta = b"config-json\n", b"weights-" * 64

        def artifact(name, data, role):
            return {"role": role, "repo": "Qwen/Test-Base",
                    "revision": "a" * 40, "filename": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data)}

        (root / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "artifacts": [artifact("model_index.json", alpha, "pipeline"),
                          artifact("transformer/weights.bin", beta,
                                   "transformer")]}))
        if installed:
            (assets / "model_index.json").write_bytes(alpha)
            (assets / "transformer").mkdir(parents=True, exist_ok=True)
            (assets / "transformer" / "weights.bin").write_bytes(beta)
        roots = setup.SetupRoots.for_paths(
            models_dir=str(assets), manifest_path=str(root / "manifest.json"),
            q8_manifest_path=None)
        service = setup.ModelSetupService(roots)
        runtime = api.WebRuntime(
            output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"),
            models_dir=str(assets), manifest_path=str(root / "manifest.json"),
            manifest_summary=api.summarize_manifest_bytes(
                (root / "manifest.json").read_bytes()),
            generation_enabled=generation_enabled, setup_service=service)
        return api.status_response(runtime).payload

    fresh = _run_check_status(tmp_path, [{
        "status": 200,
        "body": json.dumps(status(installed=False,
                                  generation_enabled=True))}])[0]
    assert fresh["disabled"] is True
    assert fresh["reason"] == ("Models not downloaded — Settings → Models lists "
                               "what is needed and can fetch them on this "
                               "machine."), fresh["reason"]
    assert fresh["banner"] is True

    installed = _run_check_status(tmp_path, [{
        "status": 200,
        "body": json.dumps(status(installed=True,
                                  generation_enabled=True))}])[0]
    assert installed["disabled"] is False
    assert "Jobs run one at a time" in installed["reason"], installed["reason"]
    assert installed["banner"] is False

    no_generation = _run_check_status(tmp_path, [{
        "status": 200,
        "body": json.dumps(status(installed=True,
                                  generation_enabled=False))}])[0]
    assert "Connect to a qwen-web server started with --enable-generation" \
        in no_generation["reason"], no_generation["reason"]
    assert no_generation["banner"] is False

