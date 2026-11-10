"""C1/C3/C5: the remote listener over an in-memory socket.

Every request is driven through the REAL ``_Handler.handle_one_request`` (the
full stdlib request-line path) over ``BytesIO`` adapters, so no socket is
created and the guarded suite's network ban stays intact. Two listeners are
distinguished exactly the way production does: ``server.remote`` is
``RemoteSettings`` on the remote one and ``None`` on the local one.

What is pinned here is the CONTRACT, not the implementation: the token gate and
its bare 401, the allowlist 403, the owner-only routes answering the same 404 an
unknown target gets, per-user job/image/reference isolation, the cap-5 reaper,
the one-live-job queue limit, every C3 feature gate, remote rewrite routing, and
the additive status block D3 parses.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import types
from pathlib import Path

import pytest

from qwen_engine import jobs, reference_store
from qwen_engine.config import GenerationConfig
from qwen_engine.core import GenerationResult
from qwen_engine.web import api, rewrite_service as rs, server
from qwen_engine.web import remote as remote_module
from qwen_engine.web.generation import WebGeneration
from qwen_engine.web.admin import AdminService
from qwen_engine.web.server import RemoteSettings

PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j2ioAAAAASUVORK5CYII=')
#: Two different one-pixel PNGs, so each remote upload is its own content id.
REFERENCE_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
TOKEN = b"remote-proxy-secret-value-0001"
ORIGIN = "https://studio.example.com"
TOKEN_HEADER = "X-Lightseed-Proxy-Token"
EMAIL_HEADER = "X-Lightseed-User-Email"
SUB_HEADER = "X-Lightseed-User-Sub"
ALICE = ("alice@example.com", "107691503500061507151")
BOB = ("bob@example.com", "107691503500061507152")


# --------------------------------------------------------------------- utils

def envelope(**values):
    return json.dumps({"schema": api.WEB_PLAN_REQUEST_SCHEMA,
                       "config": {"prompt": "red cube",
                                  "seed": "18446744073709551615", **values}}).encode()


def save(context):
    for phase in ("load", "encode", "denoise", "decode", "save"):
        context.report_phase(phase)
    path = Path(context.config.output_dir) / "result.png"
    path.write_bytes(PNG)
    return GenerationResult(str(path))


def status_line(out: str) -> str:
    return out.split("\r\n")[0]


def head_of(out: str) -> dict[str, str]:
    lines = out.split("\r\n")
    headers = {}
    for line in lines[1:]:
        if not line:
            break
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return headers


def body_of(out: str) -> dict:
    return json.loads(out.partition("\r\n\r\n")[2])


def code_of(out: str) -> str:
    # The web error payload is FLAT: {"schema", "code", "message"}.
    return body_of(out)["code"]


def request(path: str, *, method: str = "GET", headers=(), body: bytes = b"") -> bytes:
    lines = [f"{method} {path} HTTP/1.1"]
    lines.extend(headers)
    if body or method == "POST":
        lines.append(f"Content-Length: {len(body)}")
    lines.append("")
    lines.append("")
    return "\r\n".join(lines).encode() + body


def remote_headers(identity=None, *, host=f"studio.example.com", origin=ORIGIN,
                   token: bytes | str | None = TOKEN, ui: bool = True) -> list[str]:
    out = [f"Host: {host}", "Sec-Fetch-Site: same-origin"]
    if origin is not None:
        out.append(f"Origin: {origin}")
    if token is not None:
        value = token.decode() if isinstance(token, bytes) else token
        out.append(f"{TOKEN_HEADER}: {value}")
    if identity is not None:
        email, sub = identity
        out.append(f"{EMAIL_HEADER}: {email}")
        out.append(f"{SUB_HEADER}: {sub}")
    if ui:
        out.append("X-Qwen-UI: 1")
    return out


LOCAL_HEADERS = ["Host: 127.0.0.1:8765", "X-Qwen-UI: 1"]


def post_json(path: str, obj, headers) -> bytes:
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return request(path, method="POST",
                   headers=list(headers) + ["Content-Type: application/json"],
                   body=body)


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    references = tmp_path / "references"
    references.mkdir()
    return api.WebRuntime(
        output_dir=str(root), cache_dir=str(tmp_path / "cache"),
        models_dir=str(tmp_path / "models"),
        manifest_path=str(tmp_path / "manifest.json"),
        # A Q8 job needs its sidecar path on the runtime, exactly as the
        # server passes one at startup; without it every quant=q8 request is
        # refused by the adapter policy before any permission could be read.
        q8_manifest_path=str(tmp_path / "q8-manifest.json"),
        references_dir=str(references),
        manifest_summary={"configured": True, "valid": True, "schema_version": 1,
                          "artifact_count": 24, "verification": "not_performed"},
        generation_enabled=True)


@pytest.fixture
def users(tmp_path):
    from qwen_engine.users import UserStore, users_path
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    store = UserStore(users_path(str(cache)))
    for email, _sub in (ALICE, BOB):
        store.add(email)
    return store


@pytest.fixture
def started():
    """Every generation service this test built, closed afterwards."""
    services = []
    yield services
    for service in services:
        service.close()


@pytest.fixture
def generation(runtime, started):
    service = WebGeneration(runtime, executor=save)
    started.append(service)
    return service


@pytest.fixture
def remote(users, runtime, generation):
    return RemoteSettings(token=TOKEN, origin=ORIGIN, users=users,
                          web_runtime=runtime, generation=generation)


def drive(raw: bytes, *, remote=None, runtime=None, generation=None,
          rewrite=None, assets=None, admin=None) -> str:
    """Handle one raw request; ``remote=None`` is the LOCAL port."""
    handler = server._Handler.__new__(server._Handler)
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.connection = types.SimpleNamespace(
        getsockname=lambda: ("127.0.0.1", 8765))
    # The listener IS the request's context: a remote handler's runtime and
    # generation service are the ones that listener was built with.
    if runtime is None:
        runtime = remote.web_runtime if remote is not None else api.WebRuntime()
    if generation is None and remote is not None:
        generation = remote.generation
    handler.server = types.SimpleNamespace(
        bound_ip="127.0.0.1", bound_port=8765,
        assets=assets if assets is not None else {"app.js": b"console.log('x')"},
        web_runtime=runtime, rewrite=rewrite, generation=generation,
        remote=remote, admin=admin)
    handler.close_connection = False
    handler.request_version = "HTTP/1.1"
    handler.address_string = lambda: "127.0.0.1"
    handler.log_message = lambda *a, **k: None
    handler.handle_one_request()
    return handler.wfile.getvalue().decode("latin-1")


# --------------------------------------------------------------- token (C1)

def test_missing_proxy_token_is_a_bare_401(remote):
    out = drive(request("/api/v1/status", headers=remote_headers(token=None)),
                remote=remote)
    assert " 401 " in status_line(out)
    payload = body_of(out)
    assert payload["code"] == "unauthorized"
    # No detail: a caller without the secret learns nothing, not even which
    # check refused it.
    assert "status\" " not in out and "generation_available" not in out


def test_wrong_proxy_token_is_the_same_bare_401(remote):
    out = drive(request("/api/v1/status",
                        headers=remote_headers(token=b"wrong-token-000000000")),
                remote=remote)
    assert " 401 " in status_line(out)
    assert code_of(out) == "unauthorized"


def test_duplicated_token_header_is_refused(remote):
    headers = remote_headers() + [f"{TOKEN_HEADER}: {TOKEN.decode()}"]
    out = drive(request("/api/v1/status", headers=headers), remote=remote)
    assert " 401 " in status_line(out)


def test_token_is_checked_before_the_host_and_origin_policy(remote):
    # A foreign origin AND no token: the answer must be the token 401, never a
    # host/origin verdict that would confirm the listener's expectations.
    out = drive(request("/api/v1/status",
                        headers=remote_headers(token=None,
                                               origin="https://evil.example.com")),
                remote=remote)
    assert " 401 " in status_line(out)


def test_right_token_and_identity_get_the_remote_status_document(remote):
    out = drive(request("/api/v1/status", headers=remote_headers(ALICE)),
                remote=remote)
    assert " 200 " in status_line(out)
    payload = body_of(out)
    assert payload["admin"] is False
    assert payload["user"]["email"] == ALICE[0]
    assert set(payload["user"]["allow"]) == {"q8", "2k", "edit", "official_enhancer"}
    assert payload["user"]["allow"] == {"q8": True, "2k": False, "edit": False,
                                        "official_enhancer": False}
    assert payload["user"]["history"] == {"count": 0, "cap": 5,
                                          "will_replace_oldest": False}
    assert "server_version" in payload


def test_the_static_shell_is_served_without_any_identity(remote):
    # The page itself has to load to say "you're not invited" — the proxy token
    # is still required, the user identity is not.
    out = drive(request("/", headers=remote_headers()), remote=remote,
                assets={"index.html": b"<!doctype html>"})
    assert " 200 " in status_line(out)
    assert "<!doctype html>" in out


def test_every_non_static_route_needs_identity_headers(remote):
    out = drive(request("/api/v1/jobs", headers=remote_headers()), remote=remote)
    assert " 401 " in status_line(out)
    assert code_of(out) == "unauthorized"


# ------------------------------------------------- identity is remote-only

def test_identity_headers_are_ignored_on_the_local_port(runtime):
    # Port 8765 never trusts the proxy headers: there is no allowlist and the
    # owner is the local owner whatever the request claims.
    headers = ["Host: 127.0.0.1:8765", f"{EMAIL_HEADER}: {ALICE[0]}",
               f"{SUB_HEADER}: {ALICE[1]}"]
    out = drive(request("/api/v1/status", headers=headers), runtime=runtime)
    assert " 200 " in status_line(out)
    payload = body_of(out)
    # This bare server has no admin service, so it offers no Admin tab; the
    # point here is that the claimed identity produced no user block.
    assert payload["admin"] is False
    assert "user" not in payload


def test_a_spoofed_identity_on_the_local_port_never_reaches_another_owner(
        runtime, users, started):
    service = WebGeneration(runtime, executor=save)
    started.append(service)
    record = service.controller.submit(GenerationConfig(), owner=ALICE[1])
    done = service.controller.wait(record.job_id)

    headers = ["Host: 127.0.0.1:8765", f"{SUB_HEADER}: {ALICE[1]}"]
    out = drive(request(f"/api/v1/jobs/{done.job_id}", headers=headers),
                runtime=runtime, generation=service)
    # The local port is the owner's own history: an invited person's job is
    # not part of it, whatever header the request carries.
    assert " 200 " not in status_line(out)


def test_the_owners_history_holds_only_the_owners_jobs(runtime, started):
    service = WebGeneration(runtime, executor=save)
    started.append(service)
    mine = service.controller.wait(
        service.controller.submit(GenerationConfig()).job_id)
    theirs = service.controller.wait(
        service.controller.submit(GenerationConfig(), owner=ALICE[1]).job_id)
    local = ["Host: 127.0.0.1:8765"]

    listed = body_of(drive(request("/api/v1/jobs", headers=local),
                           runtime=runtime, generation=service))
    assert listed["jobs"] == [mine.job_id]
    out = drive(request(f"/api/v1/jobs/{mine.job_id}", headers=local),
                runtime=runtime, generation=service)
    assert " 200 " in status_line(out)
    out = drive(request(f"/api/v1/jobs/{theirs.job_id}/image", headers=local),
                runtime=runtime, generation=service)
    assert " 200 " not in status_line(out)


def test_the_owners_delete_all_leaves_invited_peoples_history(runtime, started):
    service = WebGeneration(runtime, executor=save)
    started.append(service)
    service.controller.wait(service.controller.submit(GenerationConfig()).job_id)
    theirs = service.controller.wait(
        service.controller.submit(GenerationConfig(), owner=ALICE[1]).job_id)
    body = json.dumps({"delete_files": True}).encode()
    out = drive(request("/api/v1/jobs/forget-all", method="POST",
                        headers=["Host: 127.0.0.1:8765",
                                 "Origin: http://127.0.0.1:8765", "X-Qwen-UI: 1",
                                 "Content-Type: application/json"], body=body),
                runtime=runtime, generation=service)
    assert " 200 " in status_line(out), out
    assert body_of(out)["forgotten"] == 1
    assert [r.job_id for r in service.controller.snapshots()] == [theirs.job_id]


def test_an_unknown_account_is_not_invited(remote):
    out = drive(request("/api/v1/status",
                        headers=remote_headers(("mallory@example.com", "999"))),
                remote=remote)
    assert " 403 " in status_line(out)
    assert code_of(out) == "not_invited"


def test_auto_invite_adds_a_new_account_with_q8_only(remote, users):
    from dataclasses import replace
    out = drive(request("/api/v1/status",
                        headers=remote_headers(("carol@example.com", "333"))),
                remote=replace(remote, auto_invite=True))
    assert " 200 " in status_line(out), out
    row = users.get("carol@example.com")
    assert row is not None and row["sub"] == "333"
    assert row["allow"] == {"q8": True, "2k": False, "edit": False,
                            "official_enhancer": False}


def test_auto_invite_keeps_out_someone_the_admin_removed(remote, users):
    from dataclasses import replace
    assert users.remove(BOB[0]) is True
    out = drive(request("/api/v1/status", headers=remote_headers(BOB)),
                remote=replace(remote, auto_invite=True))
    assert " 403 " in status_line(out)
    assert code_of(out) == "not_invited"
    assert users.get(BOB[0]) is None


def test_a_recycled_address_must_present_the_bound_account_id(remote, users):
    users.bind_sub(ALICE[0], ALICE[1])
    out = drive(request("/api/v1/status",
                        headers=remote_headers((ALICE[0], "999999999"))),
                remote=remote)
    assert " 403 " in status_line(out)
    assert code_of(out) == "not_invited"


def test_a_duplicated_identity_header_is_refused_not_merged(remote):
    headers = remote_headers(ALICE) + [f"{EMAIL_HEADER}: {BOB[0]}"]
    out = drive(request("/api/v1/status", headers=headers), remote=remote)
    assert " 401 " in status_line(out)


# ------------------------------------------- owner-only routes do not exist

@pytest.mark.parametrize("path,method", [
    ("/api/v1/settings/rewrite", "GET"),
    ("/api/v1/settings/rewrite/test", "POST"),
    ("/api/v1/models", "GET"),
    ("/api/v1/models/download", "POST"),
    ("/api/v1/models/download/cancel", "POST"),
    ("/api/v1/jobs/forget-all", "POST"),
])
def test_owner_only_routes_answer_the_same_404_an_unknown_target_gets(
        remote, path, method):
    headers = remote_headers(ALICE) + ["Content-Type: application/json"]
    raw = request(path, method=method, headers=headers,
                  body=b"{}" if method == "POST" else b"")
    out = drive(raw, remote=remote)
    assert " 404 " in status_line(out), out
    assert code_of(out) == "not_found"
    # Byte-for-byte the refusal an unpublished target gets: nothing in the
    # answer separates "forbidden here" from "does not exist".
    unknown = drive(request("/api/v1/nowhere",
                            headers=remote_headers(ALICE)), remote=remote)
    assert out.partition("\r\n\r\n")[2] == unknown.partition("\r\n\r\n")[2]


def test_forbidden_route_is_refused_before_the_identity_is_even_resolved(remote):
    # No identity headers, forbidden route: 404 (route absence) — never 401,
    # which would confirm the route exists on some other listener.
    out = drive(request("/api/v1/models", headers=remote_headers()),
                remote=remote)
    assert " 404 " in status_line(out)


# ------------------------------------------------------- per-user isolation

def run_job(service, owner, prompt="red cube"):
    body = envelope(prompt=prompt)
    record = service.controller.submit(GenerationConfig(prompt=prompt), owner=owner)
    return service.controller.wait(record.job_id)


def test_jobs_are_listed_per_owner(remote, generation):
    alice_job = run_job(generation, ALICE[1])
    bob_job = run_job(generation, BOB[1])

    out = drive(request("/api/v1/jobs", headers=remote_headers(ALICE)),
                remote=remote)
    assert " 200 " in status_line(out)
    listed = body_of(out)
    assert listed["jobs"] == [alice_job.job_id]
    assert listed["history"] == {"count": 1, "cap": 5, "will_replace_oldest": False}

    out = drive(request("/api/v1/jobs", headers=remote_headers(BOB)), remote=remote)
    assert body_of(out)["jobs"] == [bob_job.job_id]


def test_a_live_job_is_listed_but_does_not_count_against_the_history_cap(
        remote, runtime, started):
    release = threading.Event()

    def gated(context):
        if context.config.prompt == "second":
            release.wait(5)
        return save(context)

    service = WebGeneration(runtime, executor=gated)
    started.append(service)
    done = run_job(service, ALICE[1], prompt="first")
    live = service.controller.submit(GenerationConfig(prompt="second"),
                                     owner=ALICE[1])
    out = drive(request("/api/v1/jobs", headers=remote_headers(ALICE)),
                remote=remote, generation=service)
    listed = body_of(out)
    assert set(listed["jobs"]) == {done.job_id, live.job_id}
    # The cap counts FINISHED jobs: a queued one must not make the page warn
    # about an eviction the reaper would not perform.
    assert listed["history"] == {"count": 1, "cap": 5,
                                "will_replace_oldest": False}
    release.set()
    service.controller.wait(live.job_id, timeout_s=5)


def test_another_accounts_job_image_and_forget_all_answer_404(remote, generation):
    bob_job = run_job(generation, BOB[1])
    image = Path(bob_job.config.output_dir) / bob_job.artifact.filename
    assert image.is_file()

    out = drive(request(f"/api/v1/jobs/{bob_job.job_id}",
                        headers=remote_headers(ALICE)), remote=remote)
    assert " 404 " in status_line(out)
    assert code_of(out) == "job_unavailable"

    out = drive(request(f"/api/v1/jobs/{bob_job.job_id}/image",
                        headers=remote_headers(ALICE)), remote=remote)
    assert " 404 " in status_line(out)
    assert code_of(out) == "image_unavailable"

    out = drive(post_json(f"/api/v1/jobs/{bob_job.job_id}/forget", {},
                          remote_headers(ALICE)), remote=remote)
    assert " 404 " in status_line(out)

    out = request(f"/api/v1/jobs/{bob_job.job_id}/cancel", method="POST",
                  headers=remote_headers(ALICE) + ["Content-Type: application/json"],
                  body=b"{}")
    out = drive(out, remote=remote)
    assert " 404 " in status_line(out)

    # Nothing above touched Bob's entry or its file.
    assert generation.controller.get(bob_job.job_id).terminal
    assert image.is_file()
    assert drive(request(f"/api/v1/jobs/{bob_job.job_id}",
                         headers=remote_headers(BOB)),
                 remote=remote).startswith("HTTP/1.1 200")


def upload(remote, runtime, identity, raw=REFERENCE_PNG):
    out = drive(request("/api/v1/references", method="POST",
                        headers=remote_headers(identity) + [
                            "Content-Type: application/octet-stream"],
                        body=raw), remote=remote, runtime=runtime)
    assert " 201 " in status_line(out), out
    return body_of(out)["id"]


def test_references_are_owner_scoped_but_the_blob_stays_shared(runtime, remote):
    alice_id = upload(remote, runtime, ALICE)
    bob_id = upload(remote, runtime, BOB, raw=PNG)
    assert alice_id != bob_id

    out = drive(request(f"/api/v1/references/{alice_id}",
                        headers=remote_headers(ALICE)), remote=remote)
    assert " 200 " in status_line(out)
    # Same fixed 404 an unknown id gets: an id cannot be confirmed by guessing.
    for path in (f"/api/v1/references/{alice_id}",
                 f"/api/v1/references/{alice_id}/thumb"):
        out = drive(request(path, headers=remote_headers(BOB)), remote=remote)
        assert " 404 " in status_line(out), (path, out)
        assert code_of(out) == "reference_unknown"


def test_identical_bytes_uploaded_twice_share_one_blob_and_store_owner_edges(
        runtime, remote):
    alice_id = upload(remote, runtime, ALICE)
    bob_id = upload(remote, runtime, BOB)
    assert alice_id == bob_id  # content-addressed, one blob
    assert reference_store.owners_for(runtime.references_dir, alice_id) == tuple(
        sorted((ALICE[1], BOB[1])))
    for identity in (ALICE, BOB):
        out = drive(request(f"/api/v1/references/{alice_id}",
                            headers=remote_headers(identity)), remote=remote)
        assert " 200 " in status_line(out)


def test_a_rewrite_may_not_describe_another_accounts_reference(
        runtime, remote, users):
    # An edit draft needs the edit permission; the point here is OWNERSHIP, so
    # both people are granted editing and only the id differs.
    users.set_permissions(ALICE[0], edit=True)
    users.set_permissions(BOB[0], edit=True)
    bob_id = upload(remote, runtime, BOB)
    service = rs.RewriteService(
        rs.RewriteSettings(backend="local-llm", endpoint="http://127.0.0.1:8079/v1/chat/completions",
                           model="Qwen3.8",
                           prompts={"t2i": "system t2i", "edit": "system edit"},
                           timeout=30.0, references_dir=runtime.references_dir),
        transport=lambda _request: b'{"choices":[{"message":{"content":'
                                   b'"{\\"rewritten_prompt\\": \\"x\\"}"}}]}')
    try:
        request_obj = {"schema": api.REWRITE_REQUEST_SCHEMA, "task": "edit",
                       "prompt": "a cat", "draft_revision": 7,
                       "image_ids": [bob_id]}
        out = drive(post_json("/api/v1/rewrite", request_obj,
                              remote_headers(BOB)), remote=remote,
                    runtime=runtime, rewrite=service)
        assert " 200 " in status_line(out), out
        out = drive(post_json("/api/v1/rewrite", request_obj,
                              remote_headers(ALICE)), remote=remote,
                    runtime=runtime, rewrite=service)
        assert " 404 " in status_line(out)
        assert code_of(out) == "reference_unknown"
    finally:
        service.close()


def test_a_submitted_job_is_tagged_with_the_account_id(remote, generation):
    out = drive(post_json("/api/v1/jobs", envelope(), remote_headers(ALICE)),
                remote=remote, generation=generation)
    assert " 202 " in status_line(out), out
    job_id = body_of(out)["job_id"]
    record = generation.controller.get(job_id)
    assert record.owner == ALICE[1]
    # The submit document is the pre-run snapshot: the job has not been taken
    # by the worker yet, so it reads as pending (or already running).
    assert body_of(out)["queue_position"] in (0, 1)
    generation.controller.wait(job_id)
    assert generation.controller.queue_position(job_id) is None


def test_a_local_job_keeps_the_unchanged_local_document(runtime, generation):
    out = drive(post_json("/api/v1/jobs", envelope(), LOCAL_HEADERS),
                runtime=runtime, generation=generation)
    assert " 202 " in status_line(out)
    payload = body_of(out)
    # No history block on the local port: the cap does not exist there.
    assert "history" not in payload
    assert generation.controller.get(payload["job_id"]).owner == "local"


def test_a_record_written_before_ownership_loads_as_the_local_owner(tmp_path):
    # The upgrade path, and the one place a bug would be silent: a job.json
    # written by a release that had no ``owner`` field must come back as the
    # person at the machine's job, never be dropped (History would empty) and
    # never become visible to a remote account.
    import hashlib

    from qwen_engine.config import GenerationConfig
    from qwen_engine.jobs import JobController

    root = tmp_path / "images"
    root.mkdir()
    job_id = "a" * 32
    job_dir = root / job_id
    job_dir.mkdir()
    image = b"\x89PNG\r\n\x1a\n" + b"pixels"
    (job_dir / "image.png").write_bytes(image)
    record = {
        "schema": "qwen_engine.job_record.v1",
        "job_id": job_id,
        "state": "succeeded",
        "error_code": None,
        "cancel_requested": False,
        "created_ns": 1_700_000_000_000_000_000,
        "finished_ns": 1_700_000_001_000_000_000,
        "config": GenerationConfig(prompt="old").to_dict(),
        "artifact": {"filename": "image.png", "bytes": len(image),
                     "sha256": hashlib.sha256(image).hexdigest()},
    }
    assert "owner" not in record
    (job_dir / "job.json").write_text(json.dumps(record), encoding="utf-8")

    controller = JobController(root, lambda context: None)
    restored = controller.snapshots()
    assert [snapshot.job_id for snapshot in restored] == [job_id]
    assert restored[0].owner == "local"
    assert [snapshot.job_id for snapshot in controller.snapshots(owner="local")] == [job_id]
    assert controller.snapshots(owner=ALICE[1]) == ()


# ------------------------------------------------------------ queue limit

def test_a_second_live_job_is_refused_queue_limit(remote, runtime, started):
    gate = __import__("threading").Event()

    def blocked(context):
        assert gate.wait(5)
        return save(context)

    service = WebGeneration(runtime, executor=blocked)
    started.append(service)
    remote = RemoteSettings(token=TOKEN, origin=ORIGIN,
                            users=remote.users, web_runtime=runtime,
                            generation=service)
    try:
        first = drive(post_json("/api/v1/jobs", envelope(), remote_headers(ALICE)),
                      remote=remote, generation=service)
        assert " 202 " in status_line(first), first
        job_id = body_of(first)["job_id"]

        second = drive(post_json("/api/v1/jobs", envelope(), remote_headers(ALICE)),
                       remote=remote, generation=service)
        assert " 409 " in status_line(second)
        assert code_of(second) == "queue_limit"
        assert len(service.controller.snapshots(owner=ALICE[1])) == 1

        # A DIFFERENT account is unaffected by that limit.
        other = drive(post_json("/api/v1/jobs", envelope(), remote_headers(BOB)),
                      remote=remote, generation=service)
        assert " 202 " in status_line(other), other
    finally:
        gate.set()
        service.controller.wait(job_id, timeout_s=5)


# --------------------------------------------------------- history cap (C3)

def test_the_sixth_success_replaces_the_oldest_and_deletes_its_files(
        remote, generation):
    assert remote_module.REMOTE_HISTORY_CAP == 5
    done = [run_job(generation, ALICE[1], prompt=f"job {n}") for n in range(6)]
    generation.controller.wait(done[-1].job_id, timeout_s=5)

    kept = [record.job_id for record in generation.controller.snapshots(owner=ALICE[1])]
    oldest = done[0]
    assert len(kept) == 5
    assert oldest.job_id not in kept
    assert kept == [record.job_id for record in done[1:]]
    # The evicted job's OWN directory and image are gone from disk.
    assert not Path(oldest.config.output_dir).exists()
    # The survivors' files are untouched.
    for record in done[1:]:
        assert (Path(record.config.output_dir) / record.artifact.filename).is_file()
    # And its id now answers 404 for its owner.
    out = drive(request(f"/api/v1/jobs/{oldest.job_id}",
                        headers=remote_headers(ALICE)), remote=remote)
    assert " 404 " in status_line(out)


def test_the_cap_does_not_touch_the_local_owner(remote, generation):
    for n in range(6):
        run_job(generation, "local", prompt=f"local {n}")
    assert generation.controller.finished_for("local") == 6
    for record in generation.controller.snapshots(owner="local"):
        assert (Path(record.config.output_dir) / record.artifact.filename).is_file()


def test_a_failed_job_never_evicts_an_older_success(remote, runtime, started):
    def mixed(context):
        if context.config.prompt == "bad":
            raise RuntimeError("boom")
        return save(context)

    service = WebGeneration(runtime, executor=mixed)
    started.append(service)
    successes = [run_job(service, ALICE[1], prompt=f"ok {n}") for n in range(5)]
    failed = service.controller.submit(GenerationConfig(prompt="bad"), owner=ALICE[1])
    service.controller.wait(failed.job_id, timeout_s=5)
    assert service.controller.get(failed.job_id).state is jobs.JobState.FAILED
    for record in successes:
        assert service.controller.get(record.job_id) is not None
        assert (Path(record.config.output_dir) / record.artifact.filename).is_file()


def test_the_submit_document_warns_before_the_next_success_replaces(
        remote, generation):
    for n in range(4):
        run_job(generation, ALICE[1], prompt=f"fill {n}")
    out = drive(post_json("/api/v1/jobs", envelope(prompt="fifth"),
                          remote_headers(ALICE)), remote=remote,
                generation=generation)
    # Four finished + this one live: the cap is not reached yet.
    assert body_of(out)["history"] == {"count": 4, "cap": 5,
                                       "will_replace_oldest": False}
    job_id = body_of(out)["job_id"]
    generation.controller.wait(job_id, timeout_s=5)

    out = drive(post_json("/api/v1/jobs", envelope(prompt="sixth"),
                          remote_headers(ALICE)), remote=remote,
                generation=generation)
    history = body_of(out)["history"]
    assert history == {"count": 5, "cap": 5, "will_replace_oldest": True}
    sixth = body_of(out)["job_id"]
    generation.controller.wait(sixth, timeout_s=5)


# ---------------------------------------------------------- feature gates

@pytest.mark.parametrize("values,feature", [
    ({"width": 1536, "height": 1024}, "2k"),
    ({"quant": "q8"}, "q8"),
    ({"references": ["0" * 64]}, "edit"),
])
def test_a_missing_permission_refuses_the_feature_by_name(remote, users,
                                                          generation,
                                                          values, feature):
    # Revoked EXPLICITLY, so the test pins the rule rather than today's
    # DEFAULT_PERMISSIONS (q8 starts granted).
    users.set_permissions(ALICE[0], **{feature: False})
    out = drive(post_json("/api/v1/jobs", envelope(**values),
                          remote_headers(ALICE)), remote=remote,
                generation=generation)
    assert " 403 " in status_line(out), out
    assert code_of(out) == "feature_not_allowed"
    message = body_of(out)["message"]
    assert feature in message
    # One grammatical sentence naming the permission to ask the host for: the
    # page prints it verbatim, so a broken one reaches people.
    assert message == remote_module.denial(feature)
    assert message.endswith(f"needs the '{feature}' permission; ask the host")
    assert not generation.controller.snapshots()


def test_an_edit_needs_the_edit_permission(remote, runtime, generation):
    reference_id = upload(remote, runtime, ALICE)
    out = drive(post_json("/api/v1/jobs", envelope(references=[reference_id]),
                          remote_headers(ALICE)), remote=remote,
                generation=generation)
    assert " 403 " in status_line(out)
    assert body_of(out)["message"] == remote_module.denial("edit")


def test_a_granted_permission_lets_the_same_request_through(remote, users,
                                                            generation):
    users.set_permissions(ALICE[0], **{"2k": True, "edit": True, "q8": True})
    out = drive(post_json("/api/v1/jobs",
                          envelope(width=1536, height=1024, quant="q8",
                                   vae_tiled_decode=True),
                          remote_headers(ALICE)), remote=remote,
                generation=generation)
    assert " 202 " in status_line(out), out
    job_id = body_of(out)["job_id"]
    generation.controller.wait(job_id, timeout_s=5)


def test_the_local_owner_has_every_feature_without_a_gate(runtime, generation):
    # tiled decode is the adapter's OWN envelope rule above 4096 latent
    # tokens, not a permission: the local owner has no C3 gate at all.
    out = drive(post_json("/api/v1/jobs",
                          envelope(width=1536, height=1024, vae_tiled_decode=True),
                          LOCAL_HEADERS),
                runtime=runtime, generation=generation)
    assert " 202 " in status_line(out), out
    job_id = body_of(out)["job_id"]
    generation.controller.wait(job_id, timeout_s=5)


# ------------------------------------------------------- remote rewriting

def rewrite_service(runtime, transport, *, endpoint=True):
    from qwen_engine.web import rewrite_service as rs_module
    settings = rs_module.RewriteSettings(
        backend="local-llm",
        endpoint="http://127.0.0.1:8079/v1/chat/completions",
        model="Qwen3.8",
        prompts={"t2i": "system t2i", "edit": "system edit"},
        timeout=30.0, references_dir=runtime.references_dir)
    return rs_module.RewriteService(settings, transport=transport)


def rewrite_answer() -> bytes:
    return json.dumps({"choices": [{"message": {
        "content": json.dumps({"rewritten_prompt": "A red bicycle."})}}]}).encode()


def rewrite_request_obj(**overrides) -> dict:
    base = {"schema": api.REWRITE_REQUEST_SCHEMA, "task": "t2i",
            "prompt": "a cat", "draft_revision": 7, "image_ids": []}
    base.update(overrides)
    return base


def test_a_remote_draft_is_forced_onto_the_endpoint_even_when_settings_say_official(
        runtime, remote):
    seen = []

    def transport(request):
        seen.append(request)
        return rewrite_answer()

    service = rewrite_service(runtime, transport)
    try:
        # Settings claim the official enhancer answers text drafts; the remote
        # user has no `official_enhancer` permission, so C3 overrides it.
        service._preferences = service._preferences.__class__(
            choices={"t2i": "pe-llamacpp", "edit": "local-llm"}, enabled=True)
        out = drive(post_json("/api/v1/rewrite", rewrite_request_obj(),
                              remote_headers(ALICE)), remote=remote,
                    runtime=runtime, rewrite=service)
        assert " 200 " in status_line(out), out
        assert body_of(out)["backend"] == "local-llm"
        assert len(seen) == 1
    finally:
        service.close()


def test_an_unconfigured_remote_rewriter_is_a_503_rewrite_unavailable(runtime, remote):
    service = rs.RewriteService(rs.RewriteSettings.disabled(),
                                transport=lambda _request: rewrite_answer())
    try:
        out = drive(post_json("/api/v1/rewrite", rewrite_request_obj(),
                              remote_headers(ALICE)), remote=remote,
                    runtime=runtime, rewrite=service)
        # Disabled outright is still a typed refusal; the remote path never
        # silently falls back to a rewriter the account may not use.
        assert " 503 " in status_line(out), out
        assert code_of(out) in {"rewrite_disabled", "rewrite_unavailable"}
    finally:
        service.close()


def test_an_edit_rewrite_needs_the_edit_permission_on_the_remote_port(
        runtime, remote, users):
    reference_id = upload(remote, runtime, ALICE)
    service = rewrite_service(runtime, lambda _request: rewrite_answer())
    try:
        out = drive(post_json("/api/v1/rewrite",
                              rewrite_request_obj(task="edit",
                                                  image_ids=[reference_id]),
                              remote_headers(ALICE)), remote=remote,
                    runtime=runtime, rewrite=service)
        assert " 403 " in status_line(out)
        assert code_of(out) == "feature_not_allowed"

        users.set_permissions(ALICE[0], edit=True)
        out = drive(post_json("/api/v1/rewrite",
                              rewrite_request_obj(task="edit",
                                                  image_ids=[reference_id]),
                              remote_headers(ALICE)), remote=remote,
                    runtime=runtime, rewrite=service)
        assert " 200 " in status_line(out), out
    finally:
        service.close()


# ------------------------------------------------- status document (C5/C4)

def test_the_remote_status_document_has_exactly_the_contracted_user_block(remote):
    payload = body_of(drive(request("/api/v1/status", headers=remote_headers(ALICE)),
                            remote=remote))
    assert set(payload["user"]) == {"email", "display_name", "allow", "history"}
    assert payload["user"]["display_name"] == ""
    assert set(payload["user"]["history"]) == {"count", "cap", "will_replace_oldest"}
    assert payload["admin"] is False
    assert isinstance(payload["server_version"], str) and payload["server_version"]


def test_the_status_history_count_is_this_accounts_finished_jobs(remote, generation):
    for owner, count in ((ALICE[1], 2), (BOB[1], 3)):
        for n in range(count):
            run_job(generation, owner, prompt=f"{owner} {n}")
    payload = body_of(drive(request("/api/v1/status", headers=remote_headers(ALICE)),
                            remote=remote))
    assert payload["user"]["history"]["count"] == 2


def test_the_local_status_document_reports_admin_and_no_user_block(runtime, users):
    payload = body_of(drive(request("/api/v1/status",
                                    headers=["Host: 127.0.0.1:8765"]),
                            runtime=runtime, admin=AdminService(users)))
    assert payload["admin"] is True
    assert "user" not in payload
    assert isinstance(payload["server_version"], str)


def test_the_admin_routes_do_not_exist_on_the_remote_port(remote, users):
    # Even a remote listener that was (wrongly) handed the admin service must
    # answer the admin page with the same 404 as an unknown route.
    out = drive(request("/api/v1/admin/users", headers=remote_headers(ALICE)),
                remote=remote, admin=AdminService(users))
    assert " 404 " in status_line(out)
    assert body_of(out)["code"] == "not_found"


def test_server_version_is_a_short_sha_or_unknown():
    version = server.read_server_version()
    assert version == "unknown" or all(c in "0123456789abcdef" for c in version)
    assert 7 <= len(version) <= 40


# ------------------------------------------------------ transport behaviour

def test_head_is_not_routed_and_says_so_with_an_allow_header(remote):
    # The remote listener exposes no HEAD variant of any route: a probe gets
    # the typed 405 naming the method that does exist, never a 200 that would
    # confirm the target, and never a body.
    head = drive(request("/api/v1/status", method="HEAD",
                         headers=remote_headers(ALICE)), remote=remote)
    assert " 405 " in status_line(head)
    assert head_of(head)["allow"] == "GET"
    assert head.partition("\r\n\r\n")[2] == ""
    # The same target answers 200 for the method the route accepts.
    get = drive(request("/api/v1/status", headers=remote_headers(ALICE)),
                remote=remote)
    assert " 200 " in status_line(get)
    # A static asset behaves identically: it is not a HEAD-capable route.
    assert " 405 " in status_line(drive(
        request("/app.js", method="HEAD", headers=remote_headers(ALICE)),
        remote=remote))


def test_an_expect_header_is_refused_417_before_the_body_is_read(remote):
    headers = remote_headers(ALICE) + ["Content-Type: application/json",
                                       "Expect: 100-continue"]
    out = drive(request("/api/v1/jobs", method="POST", headers=headers,
                        body=envelope()), remote=remote)
    assert " 417 " in status_line(out)
    assert code_of(out) == "expectation_failed"


def test_a_missing_ui_header_refuses_a_remote_write(remote, generation):
    headers = remote_headers(ALICE, ui=False) + ["Content-Type: application/json"]
    out = drive(request("/api/v1/jobs", method="POST", headers=headers,
                        body=envelope()), remote=remote, generation=generation)
    assert " 403 " in status_line(out)
    assert code_of(out) == "ui_header_required"
    assert not generation.controller.snapshots()


def test_a_foreign_origin_is_refused_even_with_a_valid_token(remote):
    out = drive(request("/api/v1/status",
                        headers=remote_headers(ALICE,
                                               origin="https://evil.example.com")),
                remote=remote)
    assert " 403 " in status_line(out)
    assert code_of(out) == "foreign_origin_denied"


def test_the_remote_host_is_not_checked_against_the_listener_address(remote):
    # Railway's browser talks to the public domain while the socket is the
    # tunnel: any well-formed Host is legal, and a malformed one is not.
    ok = drive(request("/api/v1/status",
                       headers=remote_headers(ALICE, host="studio.example.com")),
               remote=remote)
    assert " 200 " in status_line(ok)
    bad = drive(request("/api/v1/status",
                        headers=remote_headers(ALICE, host="a b")), remote=remote)
    assert " 400 " in status_line(bad)
    assert code_of(bad) == "invalid_host"


@pytest.mark.parametrize("sub", ["local", "1" * (remote_module.MAX_SUB_CHARS + 1)])
def test_an_account_id_that_is_not_a_legal_remote_owner_is_refused(sub):
    # "local" is the owner id of the person at the machine; an account id spelled
    # that way must never be handed the owner's history.
    headers = {remote_module.EMAIL_HEADER: ["alice@example.com"],
               remote_module.SUB_HEADER: [sub]}
    with pytest.raises(api.ApiError) as caught:
        remote_module.parse_identity(headers)
    assert caught.value.status == 401


def navigation_headers(identity):
    """The proxy's headers on a browser page load: no Origin, no same-origin mark."""
    return [line for line in remote_headers(identity)
            if not line.startswith(("Sec-Fetch-Site:", "Origin:"))]


def test_a_cross_site_page_navigation_reaches_the_remote_page(remote):
    # The redirect back from Google sign-in (and a link in an email) is a
    # cross-site top-level navigation; the remote page must still load.
    navigation = ["Sec-Fetch-Site: cross-site", "Sec-Fetch-Mode: navigate",
                  "Sec-Fetch-Dest: document"]
    out = drive(request("/", headers=navigation_headers(ALICE) + navigation),
                remote=remote, assets={"index.html": b"<!doctype html>"})
    assert " 200 " in status_line(out)


@pytest.mark.parametrize("path, extra", [
    ("/api/v1/status", ["Sec-Fetch-Mode: navigate", "Sec-Fetch-Dest: document"]),
    ("/", ["Sec-Fetch-Mode: no-cors", "Sec-Fetch-Dest: iframe"]),
    ("/app.js", ["Sec-Fetch-Mode: navigate", "Sec-Fetch-Dest: document"]),
])
def test_cross_site_requests_other_than_a_page_load_stay_refused(remote, path, extra):
    out = drive(request(path, headers=navigation_headers(ALICE)
                        + ["Sec-Fetch-Site: cross-site"] + extra), remote=remote)
    assert " 403 " in status_line(out)
    assert body_of(out)["code"] == "cross_site_denied"


def test_the_local_port_still_refuses_a_cross_site_page_load(runtime):
    out = drive(request("/", headers=["Host: 127.0.0.1:8765", "Sec-Fetch-Site: cross-site",
                                      "Sec-Fetch-Mode: navigate",
                                      "Sec-Fetch-Dest: document"]), runtime=runtime)
    assert " 403 " in status_line(out)
