"""The local-only Admin page (contract C4): wire contract, service, routes.

Three layers, each tested where it actually lives:

* the PURE builders and the document validator in ``web/admin.py`` — closed
  envelopes, typed refusals, no path and no address on the wire;
* the ``AdminService`` over the REAL user store in a tmp dir, so a write's
  answer is a document the file now holds;
* the REAL ``_Handler`` over an in-memory socket (no socket is created; the
  guarded runner forbids them), asserting the two reach rules that make the
  page local-only: 404 on a listener that may not serve it, and the CSRF
  header before any body work.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import types

import pytest

from qwen_engine.users import UserStore
from qwen_engine.web import admin, api, server

HOST = "127.0.0.1:8765"
ORIGIN = "http://127.0.0.1:8765"


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def store(tmp_path):
    return UserStore(str(tmp_path / "users.json"))


@pytest.fixture()
def service(store):
    return admin.AdminService(store)


def body(**fields) -> bytes:
    return json.dumps(fields).encode("utf-8")


def add_body(email, name=None, schema=admin.ADMIN_ADD_SCHEMA):
    fields = {"schema": schema, "email": email}
    if name is not None:
        fields["display_name"] = name
    return body(**fields)


def allow(**flags) -> dict:
    base = {"q8": True, "2k": False, "edit": False, "official_enhancer": False}
    base.update(flags)
    return base


# ------------------------------------------------------------ closed envelopes


def test_add_request_accepts_only_an_email_and_a_display_name():
    assert admin.build_add_request(add_body("Alice@Example.com ")) == {
        "email": "alice@example.com", "display_name": ""}
    assert admin.build_add_request(add_body("alice@example.com", "Alice")) == {
        "email": "alice@example.com", "display_name": "Alice"}


def test_add_request_refuses_permissions_and_unknown_keys():
    # Permissions are NOT part of an invitation: a new account starts with q8
    # only, and the tick boxes change it afterwards.
    for extra in ({"allow": allow()}, {"password": "x"}, {"sub": "123"}):
        with pytest.raises(api.ApiError) as caught:
            admin.build_add_request(
                body(schema=admin.ADMIN_ADD_SCHEMA, email="a@example.com", **extra))
        assert caught.value.status == 400
        assert caught.value.code == "invalid_envelope"


def test_add_request_refuses_a_wrong_schema_and_a_missing_key():
    with pytest.raises(api.ApiError) as caught:
        admin.build_add_request(add_body("a@example.com",
                                         schema="qwen_engine.web_admin.v1"))
    assert caught.value.status == 400
    with pytest.raises(api.ApiError) as caught:
        admin.build_add_request(body(schema=admin.ADMIN_ADD_SCHEMA))
    assert caught.value.status == 400
    assert "email" in caught.value.message


def test_an_admin_request_without_its_schema_tag_is_a_400_not_a_500():
    with pytest.raises(api.ApiError) as caught:
        admin.build_add_request(json.dumps({"email": "a@example.com"}).encode())
    assert caught.value.status == 400
    assert "schema" in caught.value.message


def test_add_request_refuses_a_non_address_with_a_typed_field_error():
    for bad in ("", "   ", "no-at-sign", "two@at@example.com", "has space@x.com",
                "a@example", "x" * 400 + "@example.com"):
        with pytest.raises(api.ApiError) as caught:
            admin.build_add_request(add_body(bad))
        assert caught.value.status == 422, bad
        assert caught.value.code == "invalid_user"
        assert caught.value.field_errors
        assert caught.value.field_errors[0][0] == "email"


def test_remove_request_requires_a_real_boolean():
    good = body(schema=admin.ADMIN_REMOVE_SCHEMA, email="a@example.com",
                delete_data=True)
    assert admin.build_remove_request(good) == {
        "email": "a@example.com", "delete_data": True}
    for value in (1, "true", None):
        bad = body(schema=admin.ADMIN_REMOVE_SCHEMA, email="a@example.com",
                   delete_data=value)
        with pytest.raises(api.ApiError) as caught:
            admin.build_remove_request(bad)
        assert caught.value.status == 400
    with pytest.raises(api.ApiError):
        admin.build_remove_request(body(schema=admin.ADMIN_REMOVE_SCHEMA,
                                        email="a@example.com"))


def test_permissions_request_must_carry_exactly_the_four_flags():
    envelope = body(schema=admin.ADMIN_PERMISSIONS_SCHEMA,
                    email="a@example.com", allow=allow(**{"2k": True}))
    assert admin.build_permissions_request(envelope)["allow"]["2k"] is True
    # A partial block would leave the page and the store disagreeing about the
    # flags it did not mention, so it is refused whole.
    partial = dict(allow())
    del partial["edit"]
    for bad_allow in (partial, {"q8": True, "2k": True, "edit": True,
                               "official_enhancer": True, "turbo": True},
                      {**allow(), "2k": "yes"}):
        with pytest.raises(api.ApiError) as caught:
            admin.build_permissions_request(
                body(schema=admin.ADMIN_PERMISSIONS_SCHEMA,
                     email="a@example.com", allow=bad_allow))
        assert caught.value.status == 400


def test_display_name_request_is_bounded_and_trimmed():
    envelope = body(schema=admin.ADMIN_DISPLAY_NAME_SCHEMA,
                    email="a@example.com", display_name="  Alice  ")
    assert admin.build_display_name_request(envelope)["display_name"] == "Alice"
    with pytest.raises(api.ApiError) as caught:
        admin.build_display_name_request(
            body(schema=admin.ADMIN_DISPLAY_NAME_SCHEMA, email="a@example.com",
                 display_name="n" * 81))
    assert caught.value.status == 422


def test_the_document_validator_refuses_a_widened_or_partial_row():
    document = {
        "schema": admin.ADMIN_SCHEMA, "cap": 5,
        "users": [{"email": "a@example.com", "display_name": "", "invited":
                   "2026-10-27T10:00:00Z", "first_sign_in": None,
                   "signed_in": False, "history": 0, "active": 0,
                   "allow": allow()}]}
    assert admin.validate_admin_document(document)["users"][0]["email"] \
        == "a@example.com"
    widened = json.loads(json.dumps(document))
    widened["users"][0]["password_hash"] = "x"
    with pytest.raises(api.ApiError) as caught:
        admin.validate_admin_document(widened)
    assert caught.value.status == 500
    assert caught.value.code == "admin_unavailable"
    duplicate = json.loads(json.dumps(document))
    duplicate["users"].append(dict(document["users"][0]))
    with pytest.raises(api.ApiError):
        admin.validate_admin_document(duplicate)


def test_the_document_never_carries_a_sub_or_a_path(store, tmp_path):
    store.add("alice@example.com")
    document = admin.admin_document(store)
    text = json.dumps(document)
    assert "sub" not in text
    assert str(tmp_path) not in text
    assert document["schema"] == admin.ADMIN_SCHEMA
    assert document["cap"] == admin.HISTORY_CAP
    row = document["users"][0]
    assert set(row) == set(admin._ADMIN_USER_KEYS)
    assert row["allow"] == allow()          # a new account starts with q8 only


# ------------------------------------------------------------------- service


def test_add_then_the_document_already_holds_the_account(service, store):
    response = service.add_response(add_body("Alice@Example.com", "Alice"))
    assert response.status == 200
    assert response.payload["users"][0]["email"] == "alice@example.com"
    assert response.payload["users"][0]["display_name"] == "Alice"
    assert store.get("alice@example.com") is not None


def test_permissions_and_display_name_write_through_and_answer_the_whole_document(
        service, store):
    service.add_response(add_body("alice@example.com"))
    response = service.post_response(
        "admin_permissions",
        body(schema=admin.ADMIN_PERMISSIONS_SCHEMA, email="alice@example.com",
             allow=allow(edit=True, **{"2k": True})))
    assert response.payload["users"][0]["allow"] == allow(edit=True,
                                                          **{"2k": True})
    assert store.permissions("alice@example.com")["edit"] is True
    response = service.post_response(
        "admin_display_name",
        body(schema=admin.ADMIN_DISPLAY_NAME_SCHEMA, email="alice@example.com",
             display_name="Alice B"))
    assert response.payload["users"][0]["display_name"] == "Alice B"


def test_removing_an_unknown_account_is_a_typed_404_not_a_silent_success(service):
    with pytest.raises(api.ApiError) as caught:
        service.remove_response(
            body(schema=admin.ADMIN_REMOVE_SCHEMA, email="nobody@example.com",
                 delete_data=False))
    assert caught.value.status == 404
    assert caught.value.code == "user_unknown"


def test_remove_deletes_finished_jobs_and_references_when_asked(service, store):
    store.add("alice@example.com")
    sub = "google-sub-1"
    store.bind_sub("alice@example.com", sub)
    calls = []

    class Record:
        def __init__(self, job_id, owner, state):
            self.job_id = job_id
            self.owner = owner
            self.state = types.SimpleNamespace(value=state)

    class Controller:
        def snapshots(self):
            return [Record("a" * 32, sub, "succeeded"),
                    Record("b" * 32, sub, "running"),
                    Record("c" * 32, "local", "succeeded")]

        def forget(self, job_id, delete_files=False):
            calls.append((job_id, delete_files))
            return 2

    service.generation = types.SimpleNamespace(
        controller=Controller(),
        runtime=types.SimpleNamespace(references_dir=None))
    response = service.remove_response(
        body(schema=admin.ADMIN_REMOVE_SCHEMA, email="alice@example.com",
             delete_data=True))
    # ONLY the finished job of THAT account: a running job is never killed
    # from the admin page, and the owner's own history is untouched.
    assert calls == [("a" * 32, True)]
    assert response.payload["users"] == []
    assert store.get("alice@example.com") is None


def test_remove_without_the_tick_keeps_the_files(service, store):
    store.add("alice@example.com")
    calls = []

    class Controller:
        def snapshots(self):
            return []

        def forget(self, job_id, delete_files=False):  # pragma: no cover
            calls.append(job_id)

    service.generation = types.SimpleNamespace(controller=Controller(),
                                               runtime=None)
    service.remove_response(body(schema=admin.ADMIN_REMOVE_SCHEMA,
                                 email="alice@example.com", delete_data=False))
    assert calls == []
    assert store.get("alice@example.com") is None


def test_usage_counts_come_from_the_job_store_per_owner():
    class Record:
        def __init__(self, owner, state):
            self.job_id = "x" * 32
            self.owner = owner
            self.state = types.SimpleNamespace(value=state)

    generation = types.SimpleNamespace(controller=types.SimpleNamespace(
        snapshots=lambda: [Record("alice@example.com", "succeeded"),
                           Record("alice@example.com", "queued"),
                           Record("bob@example.com", "failed")]))
    counts = admin._usage_by_owner(generation)
    assert counts == {"alice@example.com": (1, 1), "bob@example.com": (1, 0)}
    # No job store at all: every account reads as zero, which is the truth on
    # a planning-only server.
    assert admin._usage_by_owner(None) == {}


# --------------------------------------------------- the status document keys


def _listener(*, admin_service=None, admin_enabled=True, status_user=None,
              version="abc1234"):
    runtime = api.WebRuntime(manifest_summary=api.summarize_manifest_bytes(None))
    return types.SimpleNamespace(
        web_runtime=runtime, admin=admin_service, admin_enabled=admin_enabled,
        status_user=status_user, server_version=version,
        bound_ip="127.0.0.1", bound_port=8765)


def test_status_adds_admin_version_and_user_only_when_they_exist(service):
    payload = server.status_payload(_listener(admin_service=service), object())
    assert payload["admin"] is True
    assert payload["server_version"] == "abc1234"
    assert "user" not in payload          # the owner's page has no identity

    plain = server.status_payload(
        types.SimpleNamespace(web_runtime=api.WebRuntime(), server_version=None),
        object())
    # No admin service and no callable flag: a server that never heard of the
    # admin page reports no admin key at all, and the version falls back to the
    # checkout's own read.
    assert plain["admin"] is False
    assert isinstance(plain["server_version"], str) and plain["server_version"]


def test_a_remote_listener_reports_admin_false_and_its_user_block(service):
    remote = _listener(admin_service=service, admin_enabled=False,
                       status_user=lambda handler: {
        "email": "alice@example.com", "display_name": "Alice",
        "allow": allow(**{"2k": True}),
        "history": {"count": 5, "cap": 5, "will_replace_oldest": True}})
    payload = server.status_payload(remote, object())
    assert payload["admin"] is False
    assert payload["user"]["email"] == "alice@example.com"
    assert payload["user"]["history"]["will_replace_oldest"] is True
    # The admin flag follows the LISTENER, never a request header or a body:
    # the same service, two listeners, two answers.
    local = _listener(admin_service=remote.admin, admin_enabled=True)
    assert server.status_payload(local, object())["admin"] is True


def test_the_admin_flag_hook_is_asked_and_a_broken_hook_fails_closed(service):
    asked = []
    remote = _listener(admin_service=service,
                       admin_enabled=lambda handler: asked.append(handler) or False)
    assert server.status_payload(remote, "the-request")["admin"] is False
    assert asked == ["the-request"]

    def broken(handler):
        raise RuntimeError("hook failure")

    assert server.status_payload(
        _listener(admin_service=service, admin_enabled=broken), object())[
        "admin"] is False


# ---------------------------------------------------------------- the routes


def test_admin_routes_resolve_to_their_own_kinds():
    assert server.resolve_route("GET", "/api/v1/admin/users")[0] == "admin_users"
    for path, kind in (("/api/v1/admin/users/add", "admin_add"),
                       ("/api/v1/admin/users/remove", "admin_remove"),
                       ("/api/v1/admin/users/permissions", "admin_permissions"),
                       ("/api/v1/admin/users/display-name", "admin_display_name")):
        assert server.resolve_route("POST", path)[0] == kind
    with pytest.raises(api.ApiError) as caught:
        server.resolve_route("DELETE", "/api/v1/admin/users/add")
    assert caught.value.status == 405
    with pytest.raises(api.ApiError):
        server.resolve_route("POST", "/api/v1/admin/users")
    with pytest.raises(api.ApiError):
        server.resolve_route("GET", "/api/v1/admin/users/add")


def _response(raw: bytes, listener) -> str:
    """Drive the real _Handler over an in-memory socket (never a real one)."""
    handler = server._Handler.__new__(server._Handler)
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.connection = types.SimpleNamespace(
        getsockname=lambda: ("127.0.0.1", 8765))
    handler.server = listener
    handler.close_connection = False
    handler.request_version = "HTTP/1.1"
    handler.address_string = lambda: "127.0.0.1"
    handler.log_message = lambda *args, **kwargs: None
    handler.handle_one_request()
    return handler.wfile.getvalue().decode("latin-1")


def _listener_for(service, *, admin_enabled=True):
    listener = _listener(admin_service=service, admin_enabled=admin_enabled)
    listener.assets = {"app.js": b"/* asset */"}
    return listener


ADMIN_POST = (b"POST {path} HTTP/1.1\r\nHost: " + HOST.encode() + b"\r\n"
              b"Origin: " + ORIGIN.encode() + b"\r\n"
              b"X-Qwen-UI: 1\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: {length}\r\n\r\n")


def test_get_admin_users_answers_the_document_on_the_local_listener(service, store):
    store.add("alice@example.com")
    out = _response(f"GET /api/v1/admin/users HTTP/1.1\r\nHost: {HOST}\r\n\r\n"
                    .encode(), _listener_for(service))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 200"), head.split("\r\n")[0]
    document = json.loads(payload)
    assert document["schema"] == admin.ADMIN_SCHEMA
    assert document["users"][0]["email"] == "alice@example.com"


def test_a_listener_that_does_not_serve_admin_answers_the_ordinary_404(service):
    # "There is no admin page here" is exactly what a never-published route
    # says, and nothing more: a remote user learns nothing about the routes.
    out = _response(f"GET /api/v1/admin/users HTTP/1.1\r\nHost: {HOST}\r\n\r\n"
                    .encode(), _listener_for(service, admin_enabled=False))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 404"), head.split("\r\n")[0]
    assert '"code": "not_found"' in payload
    # A write is refused BEFORE a body byte is read: same 404, no envelope work.
    write = ADMIN_POST.replace(b"{path}", b"/api/v1/admin/users/add") \
        .replace(b"{length}", b"2") + b"{}"
    out = _response(write, _listener_for(service, admin_enabled=False))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 404"), head.split("\r\n")[0]
    assert '"code": "not_found"' in payload


def test_a_post_without_the_csrf_header_never_reaches_the_service(service, store):
    out = _response(
        (f"POST /api/v1/admin/users/add HTTP/1.1\r\nHost: {HOST}\r\n"
         f"Origin: {ORIGIN}\r\nContent-Type: application/json\r\n"
         f"Content-Length: 2\r\n\r\n").encode() + b"{}",
        _listener_for(service))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 403"), head.split("\r\n")[0]
    assert "ui_header_required" in payload
    assert store.list_users() == []


def test_post_add_answers_the_whole_refreshed_document(service, store):
    payload_bytes = add_body("alice@example.com", "Alice")
    raw = (ADMIN_POST.replace(b"{path}", b"/api/v1/admin/users/add")
           .replace(b"{length}", str(len(payload_bytes)).encode())
           + payload_bytes)
    out = _response(raw, _listener_for(service))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 200"), head.split("\r\n")[0]
    document = json.loads(payload)
    assert document["users"][0]["display_name"] == "Alice"
    assert store.get("alice@example.com") is not None


def test_a_malformed_admin_envelope_is_a_typed_refusal(service):
    payload_bytes = b'{"schema": "wrong", "email": "a@example.com"}'
    raw = (ADMIN_POST.replace(b"{path}", b"/api/v1/admin/users/add")
           .replace(b"{length}", str(len(payload_bytes)).encode())
           + payload_bytes)
    out = _response(raw, _listener_for(service))
    head, _, payload = out.partition("\r\n\r\n")
    assert head.startswith("HTTP/1.1 400"), head.split("\r\n")[0]
    assert "invalid_envelope" in payload


def test_the_admin_page_never_leaks_a_filesystem_path(service, store, tmp_path):
    store.add("alice@example.com")
    payload_bytes = body(schema=admin.ADMIN_REMOVE_SCHEMA,
                         email="nobody@example.com", delete_data=False)
    raw = (ADMIN_POST.replace(b"{path}", b"/api/v1/admin/users/remove")
           .replace(b"{length}", str(len(payload_bytes)).encode())
           + payload_bytes)
    out = _response(raw, _listener_for(service))
    assert str(tmp_path) not in out
    assert "users.json" not in out


# ------------------------------------------- executed: the Admin page in node

#: Extracted verbatim, in dependency order: the constants first (an initialiser
#: that names an earlier constant needs it already assigned), then the real
#: functions. Nothing here reimplements a product rule.
ADMIN_VARS = ("API_ADMIN_USERS", "API_ADMIN_ADD", "API_ADMIN_REMOVE",
              "API_ADMIN_PERMISSIONS", "API_ADMIN_DISPLAY_NAME", "ADMIN_SCHEMA",
              "ADMIN_ADD_REQUEST_SCHEMA", "ADMIN_REMOVE_REQUEST_SCHEMA",
              "ADMIN_PERMISSIONS_REQUEST_SCHEMA",
              "ADMIN_DISPLAY_NAME_REQUEST_SCHEMA", "ADMIN_DOCUMENT_KEYS",
              "ADMIN_USER_KEYS", "ADMIN_TEXT_KEYS", "ADMIN_COUNT_KEYS",
              "PERMISSION_KEYS", "PERMISSION_LABELS", "MAX_ADMIN_USERS",
              "INVALID", "ERROR_SCHEMA", "adminDoc", "adminBusy",
              "adminRemoveTarget", "adminEnabled")

ADMIN_FNS = ("setText", "hideIf", "pyStrip", "parseAdminDocument",
             "parseAdminUserRow", "adminFirstSignInText", "adminCell",
             "adminButton", "renderAdminTable", "renderAdminRow",
             "adminPermissionBox", "adminStatus", "adminRefusal", "postAdmin",
             "loadAdminDocument", "commitAdminPermissions",
             "commitAdminDisplayName", "submitAdminAdd",
             "updateAdminAddButton", "beginAdminRemove", "cancelAdminRemove",
             "confirmAdminRemove")

_ADMIN_HARNESS = """// Built by tests/test_web_admin.py: the real Admin page building and write
// functions, run over realistic documents with the smallest DOM they touch and
// apiRequest stubbed (its own contract is covered by the Python tests above).
var fs = require("node:fs");
__PARTS__

function makeElement(tag) {
  return {
    tagName: String(tag).toUpperCase(), children: [], className: "",
    textContent: "", type: "", value: "", checked: false, hidden: false,
    disabled: false, maxLength: 0, dataset: {}, attrs: {}, handlers: {},
    appendChild: function (child) { this.children.push(child); return child; },
    replaceChildren: function () { this.children = []; },
    setAttribute: function (name, value) { this.attrs[name] = String(value); },
    getAttribute: function (name) {
      return Object.prototype.hasOwnProperty.call(this.attrs, name)
        ? this.attrs[name] : null;
    },
    removeAttribute: function (name) { delete this.attrs[name]; },
    focus: function () { this.focused = true; },
    addEventListener: function (name, fn) { this.handlers[name] = fn; }
  };
}
var NODES = new Map();
function $(id) {
  if (!NODES.has(id)) NODES.set(id, makeElement("div"));
  return NODES.get(id);
}
var document = {
  getElementById: function (id) { return NODES.has(id) ? NODES.get(id) : null; },
  createElement: function (tag) { return makeElement(tag); },
  querySelectorAll: function () { return []; }
};

var calls = [];
var nextAnswer = {res: {ok: true, status: 200}, json: {}};
function apiRequest(url, opts, mode) {
  calls.push({url: url, method: opts.method, headers: opts.headers,
              body: opts.body, mode: mode});
  return Promise.resolve(nextAnswer);
}

function reset() {
  NODES.clear();
  calls = [];
  adminDoc = null; adminBusy = false; adminRemoveTarget = null;
  adminEnabled = true;
  nextAnswer = {res: {ok: true, status: 200}, json: {}};
  $("admin-users-body").children = [];
  $("admin-add-email").value = "";
  $("admin-add-name").value = "";
  $("admin-delete-data").checked = false;
  $("admin-remove-confirm").hidden = true;
}
function pump() { return new Promise(function (r) { setTimeout(r, 20); }); }

function row(over) {
  var base = {email: "alice@example.com", display_name: "Alice",
              invited: "2026-10-27T10:00:00Z", first_sign_in: null,
              signed_in: false, history: 0, active: 0,
              allow: {q8: true, "2k": false, edit: false,
                      official_enhancer: false}};
  Object.keys(over || {}).forEach(function (k) { base[k] = over[k]; });
  return base;
}
function doc(rows, cap) {
  return {schema: ADMIN_SCHEMA, cap: cap === undefined ? 5 : cap,
          users: rows};
}
// The table renders the PARSED document (camelCase keys), exactly as
// loadAdminDocument and postAdmin hand it over.
function pdoc(rows, cap) {
  var parsed = parseAdminDocument(doc(rows, cap));
  if (parsed === INVALID) throw new Error("harness document is not valid");
  return parsed;
}
function describe(node) {
  return {tag: node.tagName, type: node.type, text: node.textContent,
          checked: node.checked, disabled: node.disabled, value: node.value,
          className: node.className,
          ariaLabel: node.attrs["aria-label"] || "",
          maxLength: node.maxLength,
          children: node.children.map(describe)};
}
function cellsOf(tr) {
  return tr.children.map(function (cell) {
    var described = describe(cell);
    if (!described.children.length) delete described.children;
    return described;
  });
}
function snapshot() {
  var body = $("admin-users-body");
  return {
    rows: body.children.length,
    emails: body.children.map(function (tr) { return tr.dataset.email; }),
    cells: body.children.map(cellsOf),
    emptyHidden: $("admin-empty").hidden,
    capText: $("admin-cap").textContent,
    statusText: $("admin-status").textContent,
    statusHidden: $("admin-status").hidden,
    errorText: $("admin-error").textContent,
    addDisabled: $("admin-add-submit").disabled,
    addEmail: $("admin-add-email").value,
    addName: $("admin-add-name").value,
    confirmHidden: $("admin-remove-confirm").hidden,
    confirmQuestion: $("admin-remove-question").textContent,
    confirmQuestionFocused: $("admin-remove-question").focused === true,
    deleteDataChecked: $("admin-delete-data").checked,
    calls: calls
  };
}

var out = {};

/* ---- 1. the table is built from the document, as text ---- */
out.table = (function () {
  reset();
  renderAdminTable(pdoc([
    row({email: "alice@example.com", display_name: "Alice",
         first_sign_in: "2026-10-20T08:30:00Z", signed_in: true, history: 3,
         active: 1,
         allow: {q8: true, "2k": true, edit: false, official_enhancer: true}}),
    row({email: "bob@example.com", display_name: "", first_sign_in: null,
         signed_in: true, history: 0, active: 0})]));
  return snapshot();
})();

out.emptyTable = (function () {
  reset();
  renderAdminTable(pdoc([]));
  return snapshot();
})();

out.invalidDocuments = (function () {
  var result = {};
  function refused(name, mutate) {
    var copy = doc([row({})]);
    mutate(copy);
    result[name] = parseAdminDocument(copy) === INVALID;
  }
  refused("extraDocumentKey", function (v) { v.total = 1; });
  refused("missingDocumentKey", function (v) { delete v.cap; });
  refused("wrongSchema", function (v) { v.schema = "qwen_engine.web_status.v1"; });
  refused("zeroCap", function (v) { v.cap = 0; });
  refused("floatCap", function (v) { v.cap = 2.5; });
  refused("usersNotArray", function (v) { v.users = {}; });
  refused("extraRowKey", function (v) { v.users[0].password_hash = "x"; });
  refused("missingRowKey", function (v) { delete v.users[0].signed_in; });
  // Case is NOT re-checked here: the store lowercases on write and the server
  // validates its own document before sending, so a page-side rule would be a
  // second, weaker copy of one that already holds.
  result.upperEmailAccepted =
    parseAdminDocument(doc([row({email: "Alice@example.com"})])) !== INVALID;
  refused("spaceEmail", function (v) { v.users[0].email = "alice @x.com"; });
  refused("duplicateEmail", function (v) { v.users.push(row({})); });
  refused("stringCount", function (v) { v.users[0].history = "3"; });
  refused("negativeCount", function (v) { v.users[0].active = -1; });
  refused("badFirstSignIn", function (v) { v.users[0].first_sign_in = 7; });
  refused("stringSignedIn", function (v) { v.users[0].signed_in = "yes"; });
  refused("partialAllow", function (v) { delete v.users[0].allow.edit; });
  refused("stringFlag", function (v) { v.users[0].allow.q8 = 1; });
  result.allRefused = Object.keys(result).every(function (k) { return result[k]; });
  result.refusedKeys = Object.keys(result).filter(function (k) {
    return k !== "allRefused" && k !== "refusedKeys" && result[k] !== true;
  });
  result.good = !!parseAdminDocument(doc([row({})]));
  result.nullDocument = parseAdminDocument(null) === INVALID;
  result.arrayDocument = parseAdminDocument([]) === INVALID;
  return result;
})();

out.firstSignIn = (function () {
  return {
    stamped: adminFirstSignInText({firstSignIn: "2026-10-20T08:30:00Z",
                                   signedIn: true}),
    signedInNoStamp: adminFirstSignInText({firstSignIn: null, signedIn: true}),
    never: adminFirstSignInText({firstSignIn: null, signedIn: false}),
    junkStamp: adminFirstSignInText({firstSignIn: "yesterday", signedIn: true})
  };
})();

/* ---- 2. the status line: refusals are the server's bounded text ---- */
out.status = (function () {
  reset();
  adminStatus("Saved.", false);
  var ok = {text: $("admin-status").textContent,
            hidden: $("admin-status").hidden,
            error: $("admin-error").textContent};
  adminStatus("The write failed.", true);
  var failed = {text: $("admin-status").textContent,
                hidden: $("admin-status").hidden,
                error: $("admin-error").textContent};
  adminStatus("", false);
  var cleared = {text: $("admin-status").textContent,
                 hidden: $("admin-status").hidden};
  return {
    ok: ok, failed: failed, cleared: cleared,
    refusal: adminRefusal({code: "user_exists", message: "already invited"}, 409),
    refusalNoCode: adminRefusal({code: "Not A Code", message: "x"}, 500),
    refusalNoMessage: adminRefusal({code: "user_unknown"}, 404)
  };
})();

/* ---- 3. the write path: one CSRF header, the refreshed document back ---- */
function step(setup) {
  // Every scenario is driven through ONE sequential chain: the write path
  // clears the shared in-flight flag when its promise settles, so interleaved
  // scenarios would see another's `adminBusy`.
  return function () {
    reset();
    return Promise.resolve(setup()).then(pump);
  };
}
var sequence = Promise.resolve();
function run(name, setup, extra) {
  sequence = sequence.then(function () {
    return step(setup)().then(function () {
      out[name] = extra ? extra() : snapshot();
    });
  });
}

run("permissionWrite", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com", history: 2})]));
  var target = pdoc([row({email: "alice@example.com"})]).users[0];
  var box = {checked: true};
  nextAnswer = {res: {ok: true, status: 200},
                json: doc([row({email: "alice@example.com",
                                display_name: "Alice",
                                allow: {q8: true, "2k": true, edit: false,
                                        official_enhancer: false}})])};
  commitAdminPermissions(target, "2k", box);
  return undefined;
});

run("displayNameWrite", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com"})]));
  nextAnswer = {res: {ok: true, status: 200},
                json: doc([row({email: "alice@example.com",
                                display_name: "Alice B"})])};
  commitAdminDisplayName("alice@example.com", {value: "  Alice B  "});
});

run("addUser", function () {
  $("admin-add-email").value = "  Carol@Example.com ";
  $("admin-add-name").value = " Carol ";
  nextAnswer = {res: {ok: true, status: 200},
                json: doc([row({email: "carol@example.com",
                                display_name: "Carol"})])};
  submitAdminAdd();
});

run("addRefusedLocally", function () {
  $("admin-add-email").value = "not-an-address";
  submitAdminAdd();
});

run("removeWithData", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com"})]));
  beginAdminRemove("alice@example.com");
  out.removePanel = snapshot();
  $("admin-delete-data").checked = true;
  nextAnswer = {res: {ok: true, status: 200}, json: doc([])};
  confirmAdminRemove();
});

run("refusedWrite", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com"})]));
  nextAnswer = {res: {ok: false, status: 409},
                json: {schema: ERROR_SCHEMA, code: "user_exists",
                       message: "that account is already invited"}};
  commitAdminDisplayName("alice@example.com", {value: "Alice"});
});

run("networkWrite", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com"})]));
  nextAnswer = {network: true};
  commitAdminDisplayName("alice@example.com", {value: "Alice"});
});

run("badDocumentBack", function () {
  renderAdminTable(pdoc([row({email: "alice@example.com"})]));
  nextAnswer = {res: {ok: true, status: 200},
                json: {schema: ADMIN_SCHEMA, cap: 5, users: "nope"}};
  commitAdminDisplayName("alice@example.com", {value: "Alice"});
});

run("loadGood", function () {
  nextAnswer = {res: {ok: true, status: 200},
                json: doc([row({email: "alice@example.com", history: 4})])};
  loadAdminDocument();
});

run("loadHttpError", function () {
  nextAnswer = {res: {ok: false, status: 500},
                json: {schema: ERROR_SCHEMA, code: "admin_unavailable",
                       message: "malformed"}};
  loadAdminDocument();
});

run("loadNetwork", function () {
  nextAnswer = {network: true};
  loadAdminDocument();
});

sequence.then(function () {
  process.stdout.write(JSON.stringify(out));
});
"""


def _run_admin_harness(tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so app.js cannot be executed")
    js = _read_app_file("app.js")
    parts = "\n".join([_js_var(js, name) for name in ADMIN_VARS]
                      + [_js_function(js, fn) for fn in ADMIN_FNS])
    harness = tmp_path / "admin_harness.js"
    harness.write_text(_ADMIN_HARNESS.replace("__PARTS__", parts),
                       encoding="utf-8")
    run = subprocess.run([node, str(harness)], capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, f"node harness failed: {run.stderr[:3000]}"
    return json.loads(run.stdout)


def _read_app_file(name: str) -> str:
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[1] / "src" / "qwen_engine" / "web_static" / name
    assert path.is_file(), f"missing packaged static asset: {path}"
    return path.read_text(encoding="utf-8")


def _js_function(js: str, signature: str) -> str:
    import re as _re

    m = _re.search(rf"function {_re.escape(signature)}\([\s\S]*?\n  \}}", js)
    assert m, f"{signature} not found in app.js"
    return m.group(0)


def _js_var(js: str, name: str) -> str:
    import re as _re

    m = _re.search(rf"^  var {_re.escape(name)} =[\s\S]*?;$", js, _re.M)
    assert m, f"{name} not found in app.js"
    return m.group(0)


def test_the_table_shows_every_column_the_brief_names(tmp_path):
    out = _run_admin_harness(tmp_path)
    table = out["table"]
    assert table["rows"] == 2
    assert table["emails"] == ["alice@example.com", "bob@example.com"]
    alice, bob = table["cells"]
    # email, display name, first sign-in, history, active, permissions, remove
    assert len(alice) == 7, alice
    assert {k: alice[0][k] for k in ("tag", "text", "className")} == {
        "tag": "TD", "text": "alice@example.com", "className": "admin-email"}
    name = alice[1]["children"][0]
    assert name["tag"] == "INPUT" and name["value"] == "Alice"
    assert name["maxLength"] == 80
    assert name["ariaLabel"] == "Display name for alice@example.com"
    assert alice[2]["text"] == "2026-10-20"          # the store's own stamp
    assert alice[3]["text"] == "3"                   # jobs in history
    assert alice[4]["text"] == "1"                   # queued or running
    # Each permission is a label wrapping the tick box and its visible words.
    perms = alice[5]["children"]
    assert [label["tag"] for label in perms] == ["LABEL"] * 4
    boxes = [label["children"][0] for label in perms]
    assert [label["children"][1]["text"] for label in perms] == [
        "Q8", "2K sizes", "Image editing", "Official prompt enhancer"]
    assert [box["checked"] for box in boxes] == [True, True, False, True]
    assert [box["type"] for box in boxes] == ["checkbox"] * 4
    assert boxes[0]["ariaLabel"] == "Q8 for alice@example.com"
    assert alice[6]["children"][0]["text"] == "Remove"
    assert alice[6]["children"][0]["className"] == "btn-danger admin-remove-btn"
    # An account that never signed in says so; an empty name is an empty input.
    assert bob[1]["children"][0]["value"] == ""
    assert bob[2]["text"] == "Yes", "signed in without a stamp: the honest yes"
    assert table["emptyHidden"] is True
    assert table["capText"] == ("Each account keeps its 5 most recent images; "
                                "a new one replaces the oldest. 2 of 5 "
                                "accounts invited.")


def test_an_empty_document_says_so_and_the_cap_line_is_still_honest(tmp_path):
    out = _run_admin_harness(tmp_path)
    empty = out["emptyTable"]
    assert empty["rows"] == 0
    assert empty["emptyHidden"] is False
    assert "0 of 5 accounts invited" in empty["capText"]
    assert empty["statusText"] == "" and empty["errorText"] == ""
    assert empty["confirmHidden"] is True


def test_the_page_refuses_any_document_the_server_could_not_have_sent(tmp_path):
    out = _run_admin_harness(tmp_path)
    docs = out["invalidDocuments"]
    assert docs["refusedKeys"] == []
    assert docs["allRefused"] is True
    assert docs["good"] is True
    assert docs["nullDocument"] is True and docs["arrayDocument"] is True
    # Case is the STORE's business: it lowercases on write and the server
    # validates its own document, so the page does not re-implement that rule.
    assert docs["upperEmailAccepted"] is True


def test_the_first_sign_in_column_never_invents_a_date(tmp_path):
    out = _run_admin_harness(tmp_path)
    column = out["firstSignIn"]
    assert column["stamped"] == "2026-10-20"
    assert column["signedInNoStamp"] == "Yes"
    assert column["never"] == "Not yet"
    assert column["junkStamp"] == "Yes", "an unparsable stamp is not a date"


def test_saving_says_it_saved_and_a_refusal_says_what_the_server_said(tmp_path):
    out = _run_admin_harness(tmp_path)
    status = out["status"]
    assert status["ok"] == {"text": "Saved.", "hidden": False, "error": ""}
    assert status["failed"]["text"] == "" and status["failed"]["hidden"] is True
    assert status["failed"]["error"] == "The write failed."
    assert status["cleared"] == {"text": "", "hidden": True}
    assert status["refusal"] == "user_exists: already invited"
    assert status["refusalNoCode"] == "The server refused the change (HTTP 500)."
    assert status["refusalNoMessage"] == "user_unknown: no message"


def test_a_permission_tick_posts_the_whole_block_and_shows_the_new_table(tmp_path):
    out = _run_admin_harness(tmp_path)
    wrote = out["permissionWrite"]
    (call,) = wrote["calls"]
    assert call["url"] == "/api/v1/admin/users/permissions"
    assert call["method"] == "POST"
    assert call["headers"]["X-Qwen-UI"] == "1"
    assert call["headers"]["Content-Type"] == "application/json"
    # The whole four-flag block, because a partial one is refused server-side:
    # the untouched flags come from the row the table was built from.
    assert json.loads(call["body"]) == {
        "schema": "qwen_engine.web_admin_permissions.v1",
        "email": "alice@example.com",
        "allow": {"q8": True, "2k": True, "edit": False,
                  "official_enhancer": False}}
    assert wrote["statusText"] == "2K sizes enabled for alice@example.com."
    assert wrote["statusHidden"] is False
    boxes = [label["children"][0] for label in wrote["cells"][0][5]["children"]]
    assert [box["checked"] for box in boxes] == [True, True, False, False], \
        "the table is the server's answer, not the tick the person made"


def test_a_display_name_is_trimmed_saved_and_the_field_is_cleared(tmp_path):
    out = _run_admin_harness(tmp_path)
    wrote = out["displayNameWrite"]
    (call,) = wrote["calls"]
    assert call["url"] == "/api/v1/admin/users/display-name"
    assert json.loads(call["body"]) == {
        "schema": "qwen_engine.web_admin_display_name.v1",
        "email": "alice@example.com", "display_name": "Alice B"}
    assert wrote["statusText"] == "Display name saved for alice@example.com."
    assert wrote["cells"][0][1]["children"][0]["value"] == "Alice B"


def test_add_user_sends_one_invitation_and_clears_the_form(tmp_path):
    out = _run_admin_harness(tmp_path)
    added = out["addUser"]
    (call,) = added["calls"]
    assert call["url"] == "/api/v1/admin/users/add"
    payload = json.loads(call["body"])
    assert payload == {"schema": "qwen_engine.web_admin_add.v1",
                       "email": "carol@example.com", "display_name": "Carol"}
    assert "allow" not in payload, "permissions are never part of an invitation"
    assert added["statusText"] == ("Invited carol@example.com. A new account "
                                   "starts with Q8 only.")
    assert added["addEmail"] == "" and added["addName"] == ""
    assert added["emails"] == ["carol@example.com"]


def test_a_local_typo_never_reaches_the_server(tmp_path):
    out = _run_admin_harness(tmp_path)
    refused = out["addRefusedLocally"]
    assert refused["calls"] == [], "a local typo never reaches the server"
    assert refused["errorText"] == \
        "Type the email address of the person to invite."


def test_remove_is_confirmed_in_the_page_and_the_tick_authorizes_the_files(
        tmp_path):
    out = _run_admin_harness(tmp_path)
    panel = out["removePanel"]
    assert panel["confirmHidden"] is False
    assert panel["confirmQuestion"] == ("Remove alice@example.com? They lose "
                                        "access to this studio at once.")
    assert panel["confirmQuestionFocused"] is True
    assert panel["deleteDataChecked"] is False, "files are never deleted unticked"
    removed = out["removeWithData"]
    (call,) = removed["calls"]
    assert call["url"] == "/api/v1/admin/users/remove"
    assert json.loads(call["body"]) == {
        "schema": "qwen_engine.web_admin_remove.v1",
        "email": "alice@example.com", "delete_data": True}
    assert removed["statusText"] == ("Removed alice@example.com, and deleted "
                                     "their images and references.")
    assert removed["confirmHidden"] is True
    assert removed["rows"] == 0 and removed["emptyHidden"] is False
    assert removed["deleteDataChecked"] is False


def test_a_failed_write_keeps_the_table_and_names_the_refusal(tmp_path):
    out = _run_admin_harness(tmp_path)
    refused = out["refusedWrite"]
    assert refused["errorText"] == "user_exists: that account is already invited"
    assert refused["statusHidden"] is True
    unreachable = out["networkWrite"]
    assert unreachable["errorText"] == ("The server is unreachable, so nothing "
                                        "was saved.")
    bad = out["badDocumentBack"]
    assert bad["errorText"] == ("The server answered with an unexpected "
                                "document, so the table may be out of date. "
                                "Reload the page.")
    # A refusal never invents a table: the rows that were on screen are gone
    # only because the harness reset between scenarios, and the refusal path
    # must not render a document it could not parse.
    for name in ("refusedWrite", "networkWrite", "badDocumentBack"):
        assert out[name]["rows"] == 0, name


def test_opening_the_admin_tab_reads_the_list_and_its_failures_are_typed(tmp_path):
    out = _run_admin_harness(tmp_path)
    good = out["loadGood"]
    (call,) = good["calls"]
    assert call["url"] == "/api/v1/admin/users"
    assert call["method"] == "GET"
    assert call["headers"] == {"Accept": "application/json", "X-Qwen-UI": "1"}
    assert call["mode"] == "json"
    assert good["rows"] == 1 and good["statusHidden"] is True
    assert good["cells"][0][3]["text"] == "4"
    assert out["loadHttpError"]["errorText"] == \
        "The account list could not be read (HTTP 500)."
    assert out["loadNetwork"]["errorText"] == \
        "The server is unreachable, so the accounts could not be read."
