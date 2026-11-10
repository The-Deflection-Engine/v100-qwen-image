"""Guarded R3/R4 wire tests for POST /api/v1/rewrite (no sockets, no torch).

The FULL request path runs over the in-memory socket adapter used by
``test_web_server_guard`` — Host/Origin/header policy, routing, body read,
wire-envelope validation, the service, and response framing. The backend is an
injected transport callable, so nothing here opens a socket.
"""
from __future__ import annotations

import io
import json
import types

import pytest

from qwen_engine.web import api
from qwen_engine.web import rewrite_service as rs
from qwen_engine.web import server

ENDPOINT = "http://127.0.0.1:8079/v1/chat/completions"
#: A syntactically valid reference id (the store resolves identities).
ID_A = "a" * 64


def answer(prompt: str = "A red bicycle.", **extra) -> bytes:
    obj = {"rewritten_prompt": prompt, **extra}
    return json.dumps({"choices": [{"message": {"role": "assistant",
                                                "content": json.dumps(obj)}}]}).encode()


def service(reply_body, **overrides) -> rs.RewriteService:
    settings = rs.RewriteSettings(
        backend="local-llm", endpoint=ENDPOINT, model="Qwen3.8",
        prompts={"t2i": "system t2i", "edit": "system edit"},
        timeout=overrides.pop("timeout", 30.0), **overrides)
    return rs.RewriteService(settings, transport=lambda _request: reply_body)


def call(raw: bytes, rewrite=None, runtime=None) -> str:
    """Drive the real handler over an in-memory socket (no socket created)."""
    h = server._Handler.__new__(server._Handler)
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.connection = types.SimpleNamespace(getsockname=lambda: ("127.0.0.1", 8765))
    h.server = types.SimpleNamespace(
        bound_ip="127.0.0.1", bound_port=8765,
        assets={"app.js": b"console.log('x')"},
        web_runtime=runtime or api.WebRuntime(), rewrite=rewrite)
    h.close_connection = False
    h.request_version = "HTTP/1.1"
    h.address_string = lambda: "127.0.0.1"
    h.log_message = lambda *a, **k: None
    h.handle_one_request()
    return h.wfile.getvalue().decode("latin-1")


def status_line(out: str) -> str:
    return out.split("\r\n")[0]


def body_of(out: str) -> dict:
    return json.loads(out.partition("\r\n\r\n")[2])


def request_object(**overrides) -> dict:
    base = {"schema": api.REWRITE_REQUEST_SCHEMA, "task": "t2i",
            "prompt": "a cat", "draft_revision": 7, "image_ids": []}
    base.update(overrides)
    return base


def post(obj: dict, rewrite=None, runtime=None, *, length=None,
         content_type="application/json", ui="1") -> str:
    raw = json.dumps(obj).encode()
    n = len(raw) if length is None else length
    head = (f"POST {server.REWRITE_ROUTE} HTTP/1.1\r\n"
            "Host: 127.0.0.1:8765\r\n")
    if ui is not None:
        head += f"X-Qwen-UI: {ui}\r\n"
    head += f"Content-Type: {content_type}\r\nContent-Length: {n}\r\n\r\n"
    return call(head.encode() + raw, rewrite, runtime)


# --------------------------------------------------------------- routing


def test_route_is_post_only_and_advertises_post():
    out = call(b"GET /api/v1/rewrite HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n\r\n")
    assert " 405 " in status_line(out)
    assert "Allow: POST" in out
    assert body_of(out)["code"] == "method_not_allowed"


def test_route_is_known_so_it_is_405_not_404():
    for method in ("HEAD", "PUT", "DELETE", "OPTIONS"):
        out = call(f"{method} /api/v1/rewrite HTTP/1.1\r\n"
                   "Host: 127.0.0.1:8765\r\n\r\n".encode())
        assert " 405 " in status_line(out), method
        assert "not_found" not in out, method


def test_resolve_route_returns_rewrite_kind():
    assert server.resolve_route("POST", server.REWRITE_ROUTE) == (
        "rewrite", server.REWRITE_ROUTE)


# ------------------------------------------------------- availability


def test_disabled_server_answers_typed_503_and_touches_no_body():
    # No service at all: the handler still refuses typed, never 500s.
    out = post(request_object(), rewrite=None)
    assert " 503 " in status_line(out)
    assert body_of(out)["code"] == "rewrite_disabled"


def test_default_status_block_is_the_canonical_disabled_block():
    block = api.status_response(api.WebRuntime()).payload["rewrite"]
    assert block == dict(api.REWRITE_DISABLED_BLOCK)
    # The amendment adds the settings pointer to the canonical block: the page
    # has to know where the choice lives even when nothing is configured.
    assert set(block) == {"state", "code", "backend", "reason", "settings"}


def test_disabled_service_refuses_each_draft_kind_with_the_missing_fact():
    """Nothing is configured on this server, so a draft of either kind is
    refused with the thing the user could act on (422, not 503: there is no
    rewriter here to be switched off). The Off SWITCH is the 503 answer."""
    svc = rs.RewriteService(rs.RewriteSettings.disabled())
    assert svc.available is False
    assert svc.active_by_task() == {"t2i": None, "edit": None}
    assert svc.status_block()["code"] == "rewrite_disabled"
    for task, words, images in (("t2i", "text-to-image", []),
                                ("edit", "image-edit", [ID_A])):
        out = post(request_object(task=task, image_ids=images), rewrite=svc)
        assert " 422 " in status_line(out), task
        assert body_of(out)["code"] == "task_unsupported"
        assert words in body_of(out)["message"]
    switched_off = rs.RewriteService(
        rs.RewriteSettings.disabled(),
        preferences=rs.RewritePreferences(
            choices={"t2i": "local-llm", "edit": "local-llm"}, enabled=False))
    for task, images in (("t2i", []), ("edit", [ID_A])):
        out = post(request_object(task=task, image_ids=images),
                   rewrite=switched_off)
        assert " 503 " in status_line(out), task
        assert body_of(out)["code"] == "rewrite_disabled"


def test_enabled_status_block_names_only_the_backend():
    svc = service(answer())
    runtime = api.WebRuntime(rewrite_block=svc.status_block())
    block = api.status_response(runtime).payload["rewrite"]
    # Which rewriter answers EACH draft kind, which ones this server can run,
    # and where the choice is changed: a route plus two labels — never an
    # address, never a path, never a model identifier.
    assert block == {"state": "available", "code": "rewrite_available",
                     "backend": "local-llm", "reason": "",
                     "backends": [{"id": "local-llm",
                                   "label": "OpenAI-compatible endpoint",
                                   "tasks": ["t2i", "edit"],
                                   "uses_gpu": False}],
                     "active_by_task": {"t2i": "local-llm",
                                        "edit": "local-llm"},
                     "settings": {"href": api.REWRITE_SETTINGS_HREF,
                                  "saved": False}}
    # The endpoint is operator-private and must never reach the browser.
    assert ENDPOINT not in json.dumps(api.status_response(runtime).payload)
    # And it still SERIALIZES: ``WebRuntime`` stores the block frozen, so a
    # validator that echoed back tuples of mappingproxy would turn every
    # status poll into a 500.
    assert json.loads(json.dumps(
        api.status_response(runtime).payload))["rewrite"]["backends"]


# ------------------------------------------------------------- success


def test_success_round_trip_carries_revision_and_original_prompt():
    out = post(request_object(prompt="a cat on a mat", draft_revision=42),
               rewrite=service(answer("A cat resting on a woven mat.")))
    assert " 200 " in status_line(out)
    payload = body_of(out)
    assert payload["schema"] == api.WEB_REWRITE_SCHEMA
    assert payload["draft_revision"] == 42
    assert payload["original_prompt"] == "a cat on a mat"
    assert payload["rewritten_prompt"] == "A cat resting on a woven mat."
    assert payload["parse_ok"] is True
    assert payload["backend"] == "local-llm"


def test_success_includes_ratio_hint_verbatim():
    out = post(request_object(), rewrite=service(answer("A bicycle.", wh_ratio="4:3")))
    assert body_of(out)["ratio"] == {"kind": "wh_ratio", "value": "4:3"}


def test_unparsed_answer_keeps_the_draft_out_of_the_response():
    out = post(request_object(prompt="secret draft text"),
               rewrite=service(json.dumps({"choices": [{"message": {
                   "content": "no json here"}}]}).encode()))
    payload = body_of(out)
    assert payload["parse_ok"] is False
    assert payload["rewritten_prompt"] == ""
    assert payload["unparsed_answer"] == "no json here"
    assert "secret draft text" not in json.dumps(
        {k: v for k, v in payload.items() if k != "original_prompt"})


# ------------------------------------------------------------- refusals


@pytest.mark.parametrize("obj,code,status", [
    (request_object(schema="qwen_engine.web_rewrite_request.v2"),
     "invalid_envelope", 400),
    ({"task": "t2i"}, "invalid_envelope", 400),
    ({**request_object(), "extra": 1}, "invalid_envelope", 400),
    (request_object(task="upscale"), "task_unsupported", 422),
    (request_object(prompt=""), "invalid_envelope", 400),
    (request_object(prompt="  \n "), "invalid_envelope", 400),
    (request_object(prompt="x" * 8001), "prompt_too_long", 422),
    (request_object(draft_revision=-1), "invalid_envelope", 400),
    (request_object(draft_revision="7"), "invalid_envelope", 400),
    (request_object(draft_revision=True), "invalid_envelope", 400),
    (request_object(image_ids=[1]), "invalid_envelope", 400),
    (request_object(image_ids=["a"]), "invalid_envelope", 400),
    # A t2i draft has no <imageN> for the rewriter to name.
    (request_object(image_ids=[ID_A]), "task_unsupported", 422),
    # An edit rewrite without references is refused, not downgraded to t2i.
    (request_object(task="edit"), "reference_required", 422),
    (request_object(task="edit", image_ids=[ID_A] * 11),
     "too_many_references", 422),
])
def test_bad_requests_are_typed_and_never_reach_the_backend(obj, code, status):
    called = []
    svc = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint=ENDPOINT,
                           model="m", prompts={"t2i": "s", "edit": "s"},
                           timeout=5.0),
        transport=lambda r: called.append(r) or answer())
    out = post(obj, rewrite=svc)
    assert f" {status} " in status_line(out)
    assert body_of(out)["code"] == code
    assert called == [], "an invalid request must never call the backend"


def test_prompt_length_limit_is_the_shared_constant():
    longest = "x" * 8000
    out = post(request_object(prompt=longest), rewrite=service(answer()))
    assert " 200 " in status_line(out), body_of(out)


def test_too_long_message_quotes_length_never_content():
    out = post(request_object(prompt="s" * 8001), rewrite=service(answer()))
    message = body_of(out)["message"]
    assert "8001" in message and "8000" in message
    assert "ssss" not in message


# -------------------------------------------------------------- framing


def test_body_cap_is_the_derived_rewrite_cap_not_the_64k_default():
    assert api.MAX_REWRITE_REQUEST_BYTES == 8000 * 4 + 1024
    assert api.MAX_REWRITE_REQUEST_BYTES < server.MAX_BODY_BYTES
    payload = b"x" * (api.MAX_REWRITE_REQUEST_BYTES + 1)
    raw = (f"POST {server.REWRITE_ROUTE} HTTP/1.1\r\n"
           "Host: 127.0.0.1:8765\r\nX-Qwen-UI: 1\r\n"
           "Content-Type: application/json\r\n"
           f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload
    out = call(raw, service(answer()))
    assert " 413 " in status_line(out)
    assert body_of(out)["code"] == "request_too_large"


def test_rewrite_cap_overrides_only_this_route():
    # The same length is still legal on /api/v1/plan (64 KiB JSON cap).
    big = b"x" * (api.MAX_REWRITE_REQUEST_BYTES + 1)
    raw = (b"POST /api/v1/plan HTTP/1.1\r\nHost: 127.0.0.1:8765\r\n"
           b"X-Qwen-UI: 1\r\nContent-Type: application/json\r\n"
           b"Content-Length: %d\r\n\r\n" % len(big)) + big
    out = call(raw)
    assert " 413 " not in status_line(out), "the plan route keeps its own cap"


@pytest.mark.parametrize("headers,code", [
    # origin / cross-site policy applies to this route like every other
    ({"Origin": "http://evil.example"}, "foreign_origin_denied"),
    ({"Sec-Fetch-Site": "cross-site"}, "cross_site_denied"),
    ({"Sec-Fetch-Site": "same-site"}, "cross_site_denied"),
    # the write header is required
    ({"X-Qwen-UI": None}, "ui_header_required"),
    ({"X-Qwen-UI": "0"}, "ui_header_required"),
    # JSON only
    ({"Content-Type": "application/octet-stream"}, "unsupported_media_type"),
    ({"Content-Type": "text/plain"}, "unsupported_media_type"),
])
def test_route_keeps_the_shared_header_policy(headers, code):
    raw = json.dumps(request_object()).encode()
    head = (f"POST {server.REWRITE_ROUTE} HTTP/1.1\r\n"
            "Host: 127.0.0.1:8765\r\n")
    for name, value in (("Origin", None), ("Sec-Fetch-Site", None),
                        ("X-Qwen-UI", "1"), ("Content-Type", "application/json")):
        override = headers.get(name, value) if name in headers else value
        if override is not None:
            head += f"{name}: {override}\r\n"
    head += f"Content-Length: {len(raw)}\r\n\r\n"
    out = call(head.encode() + raw, service(answer()))
    assert body_of(out)["code"] == code, out


def test_route_refuses_transfer_encoding_and_expect():
    raw = json.dumps(request_object()).encode()
    for extra, code, status in ((b"Transfer-Encoding: chunked\r\n",
                                 "transfer_encoding_unsupported", 400),
                                (b"Expect: 100-continue\r\n",
                                 "expectation_failed", 417)):
        out = call((f"POST {server.REWRITE_ROUTE} HTTP/1.1\r\n"
                    "Host: 127.0.0.1:8765\r\nX-Qwen-UI: 1\r\n"
                    "Content-Type: application/json\r\n").encode() + extra
                   + f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw,
                   service(answer()))
        assert f" {status} " in status_line(out)
        assert body_of(out)["code"] == code


def test_success_response_carries_the_defensive_header_set():
    out = post(request_object(), rewrite=service(answer()))
    head = out.partition("\r\n\r\n")[0]
    for name, value in server._DEFENSIVE_HEADERS:
        assert f"{name}: {value}" in head, name
    assert "Cache-Control" not in head or "no-store" in head


# ------------------------------------------------------------ the block


def test_status_block_is_additive_and_does_not_bump_the_schema():
    payload = api.status_response(api.WebRuntime()).payload
    assert payload["schema"] == api.WEB_STATUS_SCHEMA
    # C5's version and C4's admin flag come from the listener, not from this
    # pure runtime document (see ``server.read_server_version``).
    assert set(payload) == {"schema", "planning_available",
                            "generation_available", "gates", "manifest",
                            "generation_blockers", "rewrite"}
    # The four gates stay closed: rewriting is optional and never gates.
    assert set(payload["gates"]) == {"adapter", "assets", "calibration",
                                     "hardware"}


def test_rewrite_never_appears_in_generation_blockers():
    svc = rs.RewriteService(rs.RewriteSettings.disabled())
    runtime = api.WebRuntime(rewrite_block=svc.status_block())
    payload = api.status_response(runtime).payload
    assert not any("rewrite" in blocker for blocker in payload["generation_blockers"])


def test_empty_rewrite_block_is_refused_at_construction():
    with pytest.raises(ValueError):
        api.WebRuntime(rewrite_block={})


def test_status_block_wider_than_four_keys_is_refused_at_construction():
    # The endpoint is operator-private: a block carrying it is rejected when
    # the runtime is built, so it can never reach the browser at all.
    with pytest.raises(ValueError):
        api.WebRuntime(rewrite_block={"state": "available",
                                      "code": "rewrite_available",
                                      "backend": "local-llm",
                                      "reason": "",
                                      "endpoint": ENDPOINT})


def test_a_block_naming_a_rewriter_for_a_draft_it_cannot_answer_is_refused():
    """``active_by_task`` is a capability claim, and it is checked when the
    runtime is BUILT: a status page saying the enhancer answers an edit draft
    while its own row lists only ``t2i`` would offer the user a choice this
    server cannot honour, and the browser has no way to notice."""
    official = {"id": "pe-llamacpp",
                "label": api.REWRITE_BACKEND_LABELS["pe-llamacpp"],
                "tasks": ["t2i"], "uses_gpu": True}
    with pytest.raises(ValueError):
        api.WebRuntime(rewrite_block={
            "state": "available", "code": "rewrite_available",
            "backend": "pe-llamacpp", "reason": "", "backends": [official],
            "active_by_task": {"t2i": "pe-llamacpp", "edit": "pe-llamacpp"}})
    # A real split — official for the text draft, endpoint for the edit draft —
    # is legal and reaches the browser unchanged.
    runtime = api.WebRuntime(rewrite_block={
        "state": "available", "code": "rewrite_available",
        "backend": "pe-llamacpp", "reason": "",
        "backends": [official,
                     {"id": "local-llm",
                      "label": api.REWRITE_BACKEND_LABELS["local-llm"],
                      "tasks": ["t2i", "edit"], "uses_gpu": False}],
        "active_by_task": {"t2i": "pe-llamacpp", "edit": "local-llm"}})
    block = api.status_response(runtime).payload["rewrite"]
    assert block["active_by_task"] == {"t2i": "pe-llamacpp",
                                       "edit": "local-llm"}
    assert block["backend"] == "pe-llamacpp"
