"""Guarded tests for first-run model setup: the catalogue, the job, the routes.

No sockets, no network, no ML. Every download is driven either by an injected
``fetch`` seam or by the REAL ``tools/fetch_models.py`` publish/verify code fed by
a fake transport, so the reuse of that tool is what is actually under test. The
bytes are tiny and live in ``tmp_path``.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

from qwen_engine import _jsonsafe
from qwen_engine import model_setup as setup
from qwen_engine.web import api, server
from qwen_engine.web.generation import WebGeneration

REPO = "Qwen/Test-Model"
REVISION = "d" * 40


# ---------------------------------------------------------------------------
# A small model tree that behaves like the real layout
# ---------------------------------------------------------------------------


def artifact(filename: str, data: bytes, *, role: str = "processor",
             repo: str = REPO, revision: str = REVISION) -> dict:
    return {"role": role, "repo": repo, "revision": revision,
            "filename": filename, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def write_manifest(path, artifacts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}))


@pytest.fixture
def tree(tmp_path):
    """A ``models/`` parent with a two-file base manifest, a Q8 sidecar and a
    Turbo manifest, and NONE of the assets present. The directory is named
    ``assetroot`` so a test can prove no absolute path ever reaches a document.
    """
    parent = tmp_path / "assetroot"
    (parent / "assets").mkdir(parents=True)
    small = b"config-json\n"                 # 12 bytes
    big = b"weights-" * 64                   # 512 bytes
    write_manifest(parent / "manifest.json", [
        artifact("model_index.json", small, role="pipeline"),
        artifact("transformer/weights.bin", big, role="transformer")])
    write_manifest(parent / "turbo" / "manifest.json", [
        artifact("transformer/turbo.bin", big, role="transformer",
                 repo="Qwen/Test-Turbo", revision="e" * 40)])
    (parent / "q8_manifest.json").write_text(json.dumps({
        "schema": "qwen_engine.q8_artifact.v1", "repo": "unsloth/Test-GGUF",
        "revision": REVISION, "filename": "q8/test-Q8_0.gguf",
        "sha256": hashlib.sha256(big).hexdigest(), "bytes": len(big)}))
    return parent


def roots_for(parent, **overrides):
    kwargs = {"models_dir": str(parent / "assets"),
              "manifest_path": str(parent / "manifest.json"),
              "q8_manifest_path": str(parent / "q8_manifest.json")}
    kwargs.update(overrides)
    return setup.SetupRoots.for_paths(**kwargs)


def service_for(parent, **kwargs) -> setup.ModelSetupService:
    return setup.ModelSetupService(roots_for(parent), **kwargs)


def full_state(service) -> dict:
    """The exact document the route publishes: the service's own state plus the
    three keys only the transport knows. Validators are tested against THIS, not
    against a hand-made lookalike."""
    return {**service.state(), "can_download": True,
            "server_command": api.SETUP_SERVER_COMMAND,
            "notices": list(service.notices())}


LOOP = ("127.0.0.1", 43210)
REMOTE = ("100.90.1.2", 43211)
TUNNEL = ("172.16.4.9", 43212)


def download_body(ids) -> bytes:
    return json.dumps({"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                       "ids": list(ids)}).encode()


def wait_idle(service, limit: float = 5.0) -> dict:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if service.job_state()["state"] != setup.STATE_RUNNING:
            return service.job_state()
        time.sleep(0.02)
    raise AssertionError("the download job never finished")


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


def test_catalogue_lists_every_pinned_item(tree):
    items = setup.catalogue(roots_for(tree))
    ids = [item.id for item in items]
    assert ids == list(setup.ITEM_IDS)
    base = next(item for item in items if item.id == setup.BASE)
    assert base.required is True
    assert [f.dest for f in base.files] == ["model_index.json",
                                           "transformer/weights.bin"]
    turbo = next(item for item in items if item.id == setup.TURBO)
    assert turbo.required is False and turbo.files[0].repo == "Qwen/Test-Turbo"
    assert turbo.files[0].revision == "e" * 40


def test_q8_item_only_exists_when_the_sidecar_is_fetchable(tree):
    good = setup.catalogue(roots_for(tree))
    q8 = next(item for item in good if item.id == setup.Q8)
    assert q8.files[0].bytes == 512 and q8.files[0].dest == "q8/test-Q8_0.gguf"

    broken = tree / "q8_manifest.json"
    broken.write_text(json.dumps({"schema": "qwen_engine.q8_artifact.v1",
                                 "repo": "unsloth/Test-GGUF",
                                 "filename": "q8/test-Q8_0.gguf"}))
    ids = [item.id for item in setup.catalogue(roots_for(tree))]
    assert setup.Q8 not in ids
    assert setup.BASE in ids


def test_enhancer_items_are_marked_as_needing_a_binary(tree):
    items = setup.catalogue(roots_for(tree))
    enhancers = [item for item in items if item.id in (setup.PE_TEXT,
                                                       setup.PE_EDIT)]
    assert len(enhancers) == 2
    assert all(item.extra == setup.NEEDS_BINARY for item in enhancers)
    assert all(not item.required for item in enhancers)
    edit = next(item for item in enhancers if item.id == setup.PE_EDIT)
    assert sorted(f.dest for f in edit.files) == sorted(
        [setup.PE_I2I_FILE, setup.PE_I2I_MMPROJ_FILE])


def test_enhancer_pins_match_the_values_root_hashed():
    assert setup.PE_T2I_SHA256 == ("57808ab9fa2959498ef107d442afc81435b71427"
                                   "ecf7ddf13fc616860af196b1")
    assert setup.PE_I2I_SHA256 == ("77458dff91be9bd6b773d5cf95c60a3d9dbd264a"
                                    "6a699d617f867172646cf6cf")
    assert setup.PE_I2I_MMPROJ_SHA256 == ("8dedb71dbc3092dc47de9108ad373d68a1"
                                          "2854e2527d59bd9399601738f3bce1")
    assert setup.PE_T2I_REVISION == "b74e77d01fe90c8537e21d1c2f7b9bc886a17bf2"
    assert setup.PE_I2I_REVISION == "a7a26d38f4a13483ee5a740e48be9dd6f563f6bd"
    for digest in (setup.PE_T2I_SHA256, setup.PE_I2I_SHA256,
                   setup.PE_I2I_MMPROJ_SHA256):
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_catalogue_document_never_carries_a_location(tree):
    document = service_for(tree).state()
    text = json.dumps(document)
    for forbidden in (REPO, REVISION, "https://", "huggingface.co",
                      str(tree), str(tree.parent), "/home/", "/tmp/"):
        assert forbidden not in text, forbidden
    assert all(isinstance(item["dest"], str)
               and not item["dest"].startswith("/")
               and ".." not in item["dest"] for item in document["items"])
    assert set(document) == {"schema", "items", "bytes_needed", "free_bytes",
                             "required_missing", "job"}


def test_scan_reports_missing_partial_and_installed(tree):
    service = service_for(tree)
    base = next(item for item in service.items if item.id == setup.BASE)
    assert setup.scan(base)[0] == setup.MISSING
    (tree / "assets" / "model_index.json").write_bytes(b"config-json\n")
    status, missing, needed = setup.scan(base)
    assert status == setup.PARTIAL
    assert missing == ["transformer/weights.bin"]
    assert needed == 512
    (tree / "assets" / "transformer" / "weights.bin").parent.mkdir(
        parents=True, exist_ok=True)
    (tree / "assets" / "transformer" / "weights.bin").write_bytes(b"weights-" * 64)
    assert setup.scan(base)[0] == setup.INSTALLED


def test_a_wrongly_sized_file_is_not_installed(tree):
    """A half-written file must not be counted: it is what a crash leaves."""
    (tree / "assets" / "model_index.json").write_bytes(b"config-json\n")
    (tree / "assets" / "transformer").mkdir(parents=True, exist_ok=True)
    (tree / "assets" / "transformer" / "weights.bin").write_bytes(b"weights-")
    base = next(item for item in setup.catalogue(roots_for(tree))
                if item.id == setup.BASE)
    status, missing, needed = setup.scan(base)
    assert status == setup.PARTIAL
    assert missing == ["transformer/weights.bin"] and needed == 512


def test_nothing_correct_on_disk_is_missing_not_partial(tree):
    """One truncated file is not a partial download of a two-file model."""
    (tree / "assets" / "model_index.json").write_bytes(b"config-js")
    base = next(item for item in setup.catalogue(roots_for(tree))
                if item.id == setup.BASE)
    status, _missing, _needed = setup.scan(base)
    assert status == setup.MISSING


def test_required_missing_and_byte_totals(tree):
    service = service_for(tree)
    state = service.state()
    assert state["required_missing"] is True
    rows = {row["id"]: row for row in state["items"]}
    assert state["bytes_needed"] == sum(row["bytes_needed"]
                                        for row in state["items"])
    assert rows[setup.BASE]["bytes_needed"] == 12 + 512
    assert rows[setup.PE_TEXT]["bytes_needed"] == setup.PE_T2I_BYTES
    assert rows[setup.Q8]["bytes_needed"] == 512
    for name, data in (("model_index.json", b"config-json\n"),
                       ("transformer/weights.bin", b"weights-" * 64)):
        target = tree / "assets" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert service.state()["required_missing"] is False


def test_notices_are_empty_on_a_complete_checkout(tree):
    assert service_for(tree).notices() == []
    (tree / "turbo" / "manifest.json").unlink()
    notices = service_for(tree).notices()
    assert len(notices) == 1 and "Turbo" in notices[0]
    assert setup.BASE in [item.id for item in setup.catalogue(roots_for(tree))]


def test_catalogue_survives_an_unreadable_manifest(tree):
    """A manifest the server cannot read is a notice, never a false "installed" —
    and never a false "nothing required is missing" either: the base model is
    exactly the thing that manifest describes."""
    (tree / "manifest.json").write_text("{not json")
    service = service_for(tree)
    ids = [item.id for item in service.items]
    assert setup.BASE not in ids and setup.TURBO in ids
    assert any("base" in notice for notice in service.notices())
    assert service.required_missing() is True


# ---------------------------------------------------------------------------
# The download job
# ---------------------------------------------------------------------------

BASE_FILES = ("model_index.json", "transformer/weights.bin")


def blobs_for(tree) -> dict:
    """The exact bytes the fixture's manifest pins, keyed by destination."""
    return {"model_index.json": b"config-json\n",
            "transformer/weights.bin": b"weights-" * 64,
            "transformer/turbo.bin": b"weights-" * 64,
            "q8/test-Q8_0.gguf": b"weights-" * 64}


def small_items(service, ids):
    """Trim the catalogue to ``ids``: the pinned 9 GiB enhancer weights are not
    something a unit test materialises, and the job logic does not need them."""
    service.items = tuple(item for item in service.items if item.id in ids)
    return service


def make_fetch(blobs, *, raises=None, on_each=None):
    """A fake of the fetch seam: writes the pinned bytes, reports the bytes it
    wrote, and can fail or stop like a real stream."""
    calls = []

    def fetch(setup_file, dest, on_bytes):
        calls.append(setup_file.dest)
        if on_each is not None:
            on_each(setup_file, calls)
        if raises is not None:
            raise raises
        data = blobs[setup_file.dest]
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(str(dest) + ".partial")
        with open(partial, "wb") as handle:
            handle.write(data)
            on_bytes(len(data))
        os.replace(partial, dest)
    fetch.calls = calls
    return fetch


def test_a_job_fetches_every_pinned_file_and_publishes_it(tree):
    service = small_items(service_for(tree, fetch=make_fetch(blobs_for(tree))),
                          (setup.BASE,))
    job = service.start([])
    assert job["state"] == setup.STATE_RUNNING
    assert [row["item"] for row in job["files"]] == ["base", "base"]
    assert [row["name"] for row in job["files"]] == list(BASE_FILES)
    done = wait_idle(service)
    assert done["state"] == setup.STATE_COMPLETED, done["error"]
    assert done["error"] is None
    assert (tree / "assets" / "model_index.json").read_bytes() == b"config-json\n"
    assert (tree / "assets" / "transformer" / "weights.bin").read_bytes() == (
        b"weights-" * 64)
    assert not list((tree / "assets").glob("*.partial"))
    assert service.state()["required_missing"] is False


def test_required_items_are_added_to_every_request(tree):
    fetch = make_fetch(blobs_for(tree))
    service = small_items(service_for(tree, fetch=fetch), (setup.BASE, setup.TURBO))
    assert service.start([])["ids"] == [setup.BASE]
    service.cancel()
    wait_idle(service)
    job = service.start([setup.TURBO])
    assert sorted(job["ids"]) == sorted([setup.BASE, setup.TURBO])
    wait_idle(service)


def test_an_unknown_id_refuses_the_whole_request(tree):
    fetch = make_fetch(blobs_for(tree))
    service = small_items(service_for(tree, fetch=fetch), (setup.BASE,))
    with pytest.raises(setup.UnknownItem):
        service.start(["not-a-model"])
    assert fetch.calls == []
    assert service.job_state()["state"] == setup.STATE_IDLE


def test_a_request_for_an_item_this_server_cannot_offer_refuses(tree):
    service = small_items(service_for(tree, fetch=make_fetch(blobs_for(tree))),
                          (setup.BASE,))
    with pytest.raises(setup.UnknownItem):
        service.start([setup.Q8])
    assert service.job_state()["state"] == setup.STATE_IDLE


def test_only_one_job_runs_at_a_time(tree):
    gate = threading.Event()
    entered = threading.Event()
    blobs = blobs_for(tree)

    def slow(setup_file, dest, on_bytes):
        entered.set()
        assert gate.wait(5.0), "the test never released the first file"
        make_fetch(blobs)(setup_file, dest, on_bytes)

    service = small_items(service_for(tree, fetch=slow), (setup.BASE,))
    threads_before = threading.active_count()
    service.start([])
    assert entered.wait(5.0)
    with pytest.raises(setup.DownloadBusy):
        service.start([])
    assert threading.active_count() == threads_before + 1
    gate.set()
    assert wait_idle(service)["state"] == setup.STATE_COMPLETED
    assert threading.active_count() == threads_before


def test_a_job_that_would_not_fit_is_refused_before_anything_starts(tree):
    fetch = make_fetch(blobs_for(tree))
    service = small_items(service_for(tree, fetch=fetch,
                                      disk_free=lambda _path: 1024),
                          (setup.BASE,))
    with pytest.raises(setup.NotEnoughSpace):
        service.start([])
    assert fetch.calls == []
    assert service.job_state()["state"] == setup.STATE_IDLE
    assert not (tree / "assets" / "model_index.json").exists()


def test_progress_counts_bytes_per_file_and_overall(tree):
    blobs = blobs_for(tree)
    snapshots = []

    def fetch(setup_file, dest, on_bytes):
        data = blobs[setup_file.dest]
        half = len(data) // 2
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(str(dest) + ".partial", "wb") as handle:
            handle.write(data[:half])
            on_bytes(half)
            handle.flush()
            snapshots.append([(row["name"], row["done"], row["total"])
                              for row in service.job_state()["files"]])
            handle.write(data[half:])
            on_bytes(len(data) - half)
        os.replace(str(dest) + ".partial", dest)

    service = small_items(service_for(tree, fetch=fetch), (setup.BASE,))
    service.start([])
    done = wait_idle(service)
    assert any(row[1] == 6 and row[2] == 12 for row in snapshots[0]
               if row[0] == "model_index.json")
    assert done["bytes_done"] == done["bytes_total"] == 12 + 512
    assert all(row["done"] <= row["total"] for row in done["files"])
    assert sum(row["total"] for row in done["files"]) == 12 + 512


def test_cancel_stops_the_job_and_keeps_the_partial_for_a_resume(tree):
    blobs = blobs_for(tree)
    started = threading.Event()
    release = threading.Event()

    def fetch(setup_file, dest, on_bytes):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(str(dest) + ".partial", "wb") as handle:
            handle.write(blobs[setup_file.dest][:8])
            on_bytes(8)
            handle.flush()
            started.set()
            assert release.wait(5.0), "the test never released the stream"
            if service._cancel.is_set():
                raise setup.DownloadCancelled("the download was stopped")
        os.replace(str(dest) + ".partial", dest)

    service = small_items(service_for(tree, fetch=fetch), (setup.BASE,))
    service.start([])
    assert started.wait(5.0)
    answer = service.cancel()
    assert answer["state"] == setup.STATE_RUNNING, "cancel answers with the job as it is"
    release.set()
    done = wait_idle(service)
    assert done["state"] == setup.STATE_CANCELLED
    assert done["error"] is None
    partial = tree / "assets" / "model_index.json.partial"
    assert partial.exists() and partial.stat().st_size == 8
    assert not (tree / "assets" / "model_index.json").exists()


def test_cancelling_when_nothing_runs_is_a_no_op(tree):
    service = service_for(tree, fetch=make_fetch(blobs_for(tree)))
    assert service.cancel()["state"] == setup.STATE_IDLE


def test_a_cancelled_job_leaves_the_next_job_free_to_finish(tree):
    blobs = blobs_for(tree)
    stop = {"now": True}

    def fetch(setup_file, dest, on_bytes):
        if stop["now"]:
            raise setup.DownloadCancelled("the download was stopped")
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = str(dest) + ".partial"
        with open(partial, "wb") as handle:
            handle.write(blobs[setup_file.dest])
            on_bytes(len(blobs[setup_file.dest]))
        os.replace(partial, dest)

    service = small_items(service_for(tree, fetch=fetch), (setup.BASE,))
    service.start([])
    assert wait_idle(service)["state"] == setup.STATE_CANCELLED
    stop["now"] = False
    assert service.start([])["state"] == setup.STATE_RUNNING
    assert wait_idle(service)["state"] == setup.STATE_COMPLETED
    assert (tree / "assets" / "transformer" / "weights.bin").exists()


def test_a_prompt_that_is_not_the_pinned_text_is_deleted(tree):
    """The md5 pin ``rewrite.py`` checks is enforced here too, and the wrong file
    must not be left where the rewriter would read it."""
    blobs = blobs_for(tree)
    service = service_for(tree)
    service.items = tuple(item for item in service.items
                          if item.id in (setup.BASE, setup.PROMPTS))

    def fetch(setup_file, dest, on_bytes):
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (b"sorry, wrong instructions\n"
                if setup_file.dest.startswith("system_prompt")
                else blobs[setup_file.dest])
        dest.write_bytes(data)
        on_bytes(len(data))

    service._fetch = fetch
    assert service.start([setup.PROMPTS])["state"] == setup.STATE_RUNNING
    done = wait_idle(service)
    assert done["state"] == setup.STATE_FAILED
    assert done["error"]["code"] == setup.VerificationFailed.code
    assert "prompt" in done["error"]["message"]
    assert not (tree / "rewrite" / "system_prompt_t2i.txt").exists()


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------


def ready_service(tree):
    """A service whose base assets are already on disk."""
    service = small_items(service_for(tree, fetch=make_fetch(blobs_for(tree))),
                          (setup.BASE,))
    service.start([])
    wait_idle(service)
    return service


def test_the_three_routes_are_routed_and_the_others_refused():
    assert server.resolve_route("GET", "/api/v1/models") == (
        "models", "/api/v1/models")
    assert server.resolve_route("POST", "/api/v1/models/download")[0] == (
        "models_download")
    assert server.resolve_route("POST", "/api/v1/models/download/cancel")[0] == (
        "models_cancel")
    for method, target, allow in (
            ("POST", "/api/v1/models", "GET"),
            ("GET", "/api/v1/models/download", "POST"),
            ("GET", "/api/v1/models/download/cancel", "POST")):
        with pytest.raises(api.ApiError) as caught:
            server.resolve_route(method, target)
        assert caught.value.status == 405
        assert dict(caught.value.headers)["Allow"] == allow


def test_a_read_is_allowed_from_any_client_but_says_what_it_will_not_do(tree):
    service = service_for(tree)
    near = server.models_get_response(service, LOOP)
    far = server.models_get_response(service, ("100.90.1.2", 43211))
    assert near.status == far.status == 200
    assert near.payload["can_download"] is True
    assert far.payload["can_download"] is False
    assert json.dumps(far.payload).count('"can_download": false') == 1


def test_every_write_refuses_a_client_that_is_not_on_this_machine(tree):
    service = service_for(tree)
    for address in (("100.90.1.2", 1), ("172.16.4.9", 1), ("10.0.0.5", 1),
                    ("192.168.1.9", 1), ("8.8.8.8", 1), ("fe80::1", 1), None):
        with pytest.raises(api.ApiError) as caught:
            server.models_download_response(service, download_body([]), address)
        assert caught.value.status == 403
        assert caught.value.code == "setup_server_only"
        message = caught.value.message
        assert "fetch_models.py" in message and "--fetch" in message
        assert "://" not in message and "//" not in message
        assert "/home/" not in message and "/tmp/" not in message
        assert str(tree) not in message
        with pytest.raises(api.ApiError) as cancel:
            server.models_cancel_response(service, b"{}", address)
        assert cancel.value.status == 403
    for address in (("127.0.0.1", 1), ("::1", 1)):
        assert server.models_cancel_response(service, b"{}", address).status == 200


def test_the_download_response_carries_the_running_job(tree):
    service = small_items(service_for(tree, fetch=make_fetch(blobs_for(tree))),
                          (setup.BASE,))
    answer = server.models_download_response(service, download_body([]), LOOP)
    assert answer.status == 202
    assert answer.payload["schema"] == setup.JOB_SCHEMA
    assert answer.payload["state"] == setup.STATE_RUNNING
    assert wait_idle(service)["state"] == setup.STATE_COMPLETED


def test_every_refusal_keeps_its_own_status_code(tree):
    service = service_for(tree)
    with pytest.raises(api.ApiError) as unknown:
        server.models_download_response(service, download_body(["nope"]), LOOP)
    assert (unknown.value.status, unknown.value.code) == (422, "unknown_model")

    small = small_items(service_for(tree, fetch=make_fetch(blobs_for(tree)),
                                    disk_free=lambda _p: 8), (setup.BASE,))
    with pytest.raises(api.ApiError) as space:
        server.models_download_response(small, download_body([]), LOOP)
    assert (space.value.status, space.value.code) == (507, "insufficient_space")

    gate = threading.Event()
    entered = threading.Event()
    blobs = blobs_for(tree)

    def hold(setup_file, dest, on_bytes):
        entered.set()
        assert gate.wait(5.0)
        make_fetch(blobs)(setup_file, dest, on_bytes)

    busy = small_items(service_for(tree, fetch=hold), (setup.BASE,))
    assert server.models_download_response(busy, download_body([]),
                                           LOOP).status == 202
    assert entered.wait(5.0)
    with pytest.raises(api.ApiError) as clash:
        server.models_download_response(busy, download_body([]), LOOP)
    assert (clash.value.status, clash.value.code) == (409, "setup_busy")
    gate.set()
    assert wait_idle(busy)["state"] == setup.STATE_COMPLETED


def test_a_server_with_no_setup_service_says_so(tree):
    for call in (lambda: server.models_get_response(None, LOOP),
                 lambda: server.models_download_response(None, download_body([]),
                                                         LOOP),
                 lambda: server.models_cancel_response(None, b"{}", LOOP)):
        with pytest.raises(api.ApiError) as caught:
            call()
        assert caught.value.status == 503
        assert caught.value.code == "models_unavailable"


def test_cancel_takes_only_an_empty_object(tree):
    service = service_for(tree)
    assert server.models_cancel_response(service, b"{}", LOOP).status == 200
    for body in (b'{"ids":[]}', b"[]", b"{\"schema\":\"x\"}"):
        with pytest.raises(api.ApiError) as caught:
            server.models_cancel_response(service, body, LOOP)
        assert caught.value.status == 400
        assert caught.value.code == "invalid_envelope"


def test_the_download_envelope_is_closed_and_takes_ids_only():
    ok = api.build_models_download_request(
        json.dumps({"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                    "ids": ["pe-t2i", "pe-t2i"]}).encode())
    assert ok == {"ids": ["pe-t2i"]}
    for body in (
            json.dumps({"ids": []}).encode(),
            json.dumps({"schema": "qwen_engine.web_models.v1",
                        "ids": []}).encode(),
            json.dumps({"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                        "ids": [], "model_dir": "/tmp/x"}).encode(),
            json.dumps({"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                        "ids": ["base"], "repo": "evil/repo"}).encode(),
            json.dumps({"schema": setup.DOWNLOAD_REQUEST_SCHEMA,
                        "ids": "base"}).encode(),
            b"[", b"null"):
        with pytest.raises(api.ApiError) as caught:
            api.build_models_download_request(body)
        assert caught.value.status in (400, 422)
        if b"model_dir" in body or b"repo" in body:
            assert "no repository, revision, URL" in caught.value.message


def test_a_catalogue_with_an_absolute_destination_is_refused(tree):
    """The destination the page is shown is display text, and an absolute path
    in it means a server that leaked its layout: refused, not sanitised."""
    document = full_state(service_for(tree))
    document["items"][0]["dest"] = "/home/operator/models/assets"
    with pytest.raises(api.ApiError) as caught:
        api.validate_models_state(document)
    assert (caught.value.status, caught.value.code) == (500, "models_unavailable")
    assert any("absolute path" in str(pair) for pair in caught.value.field_errors)


def test_the_state_validator_is_closed_over_every_document(tree):
    document = full_state(service_for(tree))
    document["can_download"] = False
    document["job"] = {"schema": setup.JOB_SCHEMA, "state": setup.STATE_RUNNING,
                       "ids": ["base"],
                       "files": [{"item": "base", "name": "model_index.json",
                                  "done": 4, "total": 12}],
                       "bytes_done": 4, "bytes_total": 12, "error": None}
    checked = api.validate_models_state(document)
    assert set(checked) == {"schema", "items", "bytes_needed", "free_bytes",
                            "required_missing", "can_download",
                            "server_command", "job", "notices"}
    assert checked["job"]["files"][0]["done"] == 4

    for tamper in (
            lambda doc: doc.update({"extra_key": 1}),
            lambda doc: doc.update({"server_command": "rm -rf /"}),
            lambda doc: doc["items"][0].update({"repo": "evil/repo"}),
            lambda doc: doc["items"][0].update({"id": "not-a-model"}),
            lambda doc: doc["items"][0].update({"installed": "maybe"}),
            lambda doc: doc["items"][0]["files"][0].update({"bytes": -5}),
            lambda doc: doc["items"][0]["files"][0].update(
                {"name": "../../etc/passwd"}),
            lambda doc: doc["items"][0].update({"dest": "models/../../etc"}),
            lambda doc: doc["items"][0].update({"missing_files": ["../.ssh/id"]}),
            lambda doc: doc["job"].update(
                {"files": [{"item": "base", "name": "../x", "done": 1,
                            "total": 4}]}),
            lambda doc: doc.update({"bytes_needed": -1}),
            lambda doc: doc.update({"required_missing": False}),
            lambda doc: doc["job"].update({"bytes_done": 13}),
            lambda doc: doc["job"]["files"][0].update({"done": 99}),
            lambda doc: doc["job"].update({"ids": ["not-a-model"]}),
            lambda doc: doc["job"].update({"state": "almost"}),
            lambda doc: doc["job"].update({"error": {"code": "x"}}),
            lambda doc: doc.pop("job")):
        broken = json.loads(json.dumps(document))
        tamper(broken)
        with pytest.raises(api.ApiError):
            api.validate_models_state(broken)


def test_the_validator_accepts_the_frozen_document_the_runtime_holds(tree):
    """``WebRuntime`` freezes its mappings; a validator that only accepted plain
    dicts would 500 on every poll."""
    document = json.loads(json.dumps(full_state(service_for(tree))))
    frozen = _jsonsafe.freeze_json(document, "models", [])
    checked = api.validate_models_state(frozen)
    assert isinstance(checked, dict) and isinstance(checked["items"], list)
    json.dumps(checked)


def test_generate_is_gated_while_the_required_assets_are_missing(tree):
    paths = {"output_dir": str(tree / "out"), "cache_dir": str(tree / "cache"),
             "models_dir": str(tree / "assets"),
             "manifest_path": str(tree / "manifest.json")}
    summary = api.summarize_manifest_bytes(
        (tree / "manifest.json").read_bytes())
    missing = api.WebRuntime(**paths, manifest_summary=summary,
                             generation_enabled=True,
                             setup_service=service_for(tree))
    assert api.MODELS_MISSING in api._blockers(missing)
    payload = api.status_response(missing).payload
    assert api.MODELS_MISSING in payload["generation_blockers"]
    assert payload["generation_available"] is False
    assert set(payload["gates"]) == set(api._gates())
    # The runtime document is exactly what it always was: C5's version and C4's
    # admin flag are supplied by the LISTENER (the local and remote routes),
    # never baked in here — that keeps ``status_response`` pure and the schema
    # tag un-bumped.
    assert set(payload) == {"schema", "planning_available",
                            "generation_available", "gates", "manifest",
                            "generation_blockers", "rewrite"}

    installed = api.WebRuntime(**paths, manifest_summary=summary,
                               generation_enabled=True,
                               setup_service=ready_service(tree))
    assert api.MODELS_MISSING not in api._blockers(installed)

    unwired = api.WebRuntime(**paths, manifest_summary=summary,
                             generation_enabled=True)
    assert api.MODELS_MISSING not in api._blockers(unwired)


def test_a_setup_service_object_that_is_not_one_is_dropped(tree):
    """A half-wired runtime must not be able to emit a half-true blocker list."""
    runtime = api.WebRuntime(manifest_summary=api.summarize_manifest_bytes(None),
                             generation_enabled=True, setup_service=object())
    assert runtime.setup_service is None
    assert api.MODELS_MISSING not in api._blockers(runtime)


# ---------------------------------------------------------------------------
# The reuse of tools/fetch_models.py: the real publish/verify code, fake bytes
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, data: bytes, start: int, headers: dict) -> None:
        self._data = data[start:]
        self.status = 206 if start else 200
        self.headers = headers

    def read(self, size: int) -> bytes:
        chunk, self._data = self._data[:size], self._data[size:]
        return chunk

    def close(self) -> None:
        pass


class _FakeTransport:
    """Serves bytes for a pinned URL; records the resume offset it was asked for."""

    def __init__(self, blobs_by_name, *, tamper=False):
        self.blobs = blobs_by_name
        self.tamper = tamper
        self.starts: list[int] = []

    def get(self, url: str, start: int):
        name = url.rsplit("/", 1)[-1]
        self.starts.append(start)
        data = self.blobs[name]
        if self.tamper:
            data = b"tampered-" + data
        headers = {}
        if start:
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        return _FakeStream(data, start, headers)


@pytest.fixture
def real_fetcher(monkeypatch):
    """The REAL ``tools/fetch_models.py`` publish/verify path with a fake
    transport: this is the code the download job actually runs."""
    module = setup.load_fetch_tool()

    def install(transport):
        monkeypatch.setattr(module, "UrllibTransport", lambda *a, **k: transport)
        return transport
    return install


def base_transport(**kwargs):
    return _FakeTransport({"model_index.json": b"config-json\n",
                           "weights.bin": b"weights-" * 64}, **kwargs)


def test_the_default_fetch_publishes_verified_bytes(tree, real_fetcher):
    transport = real_fetcher(base_transport())
    service = small_items(service_for(tree), (setup.BASE,))
    service.start([])
    done = wait_idle(service)
    assert done["state"] == setup.STATE_COMPLETED, done["error"]
    assert (tree / "assets" / "transformer" / "weights.bin").read_bytes() == (
        b"weights-" * 64)
    assert transport.starts == [0, 0]


def test_a_sha256_mismatch_is_reported_and_never_published(tree, real_fetcher):
    transport = real_fetcher(base_transport(tamper=True))
    service = small_items(service_for(tree), (setup.BASE,))
    service.start([])
    done = wait_idle(service)
    assert done["state"] == setup.STATE_FAILED
    assert done["error"]["code"] == setup.VerificationFailed.code
    assert not (tree / "assets" / "transformer" / "weights.bin").exists()
    assert transport.starts


def test_an_existing_file_is_never_overwritten(tree, real_fetcher):
    (tree / "assets" / "transformer").mkdir(parents=True, exist_ok=True)
    (tree / "assets" / "transformer" / "weights.bin").write_bytes(
        b"someone elses bytes!" * 32)
    real_fetcher(base_transport())
    service = small_items(service_for(tree), (setup.BASE,))
    service.start([])
    done = wait_idle(service)
    assert done["state"] == setup.STATE_FAILED
    assert (tree / "assets" / "transformer" / "weights.bin").read_bytes() == (
        b"someone elses bytes!" * 32)


def test_a_partial_is_resumed_from_where_it_stopped(tree, real_fetcher):
    (tree / "assets" / "transformer").mkdir(parents=True, exist_ok=True)
    (tree / "assets" / "transformer" / "weights.bin.partial").write_bytes(
        b"weights-" * 8)
    transport = real_fetcher(base_transport())
    service = small_items(service_for(tree), (setup.BASE,))
    service.start([])
    done = wait_idle(service)
    assert done["state"] == setup.STATE_COMPLETED, done["error"]
    assert 64 in transport.starts, transport.starts
    assert (tree / "assets" / "transformer" / "weights.bin").stat().st_size == 512


def test_a_server_without_a_fetch_tool_refuses_before_starting_a_thread(
        tree, monkeypatch):
    def unavailable() -> None:
        raise setup.FetcherUnavailable("the fetch tool is not here")

    monkeypatch.setattr(setup, "load_fetch_tool", unavailable)
    service = small_items(service_for(tree), (setup.BASE,))
    with pytest.raises(setup.FetcherUnavailable):
        service.start([])
    assert service.job_state()["state"] == setup.STATE_IDLE


def test_a_corrupt_download_is_retried_by_the_next_press_and_then_finishes(
        tree, real_fetcher, monkeypatch):
    """A bad checksum is not published and not silently retried forever: the job
    fails once, the catalogue still says the file is wanted, and the user's next
    Download press fetches that one file again and completes."""
    transport = real_fetcher(base_transport(tamper=True))
    service = small_items(service_for(tree), (setup.BASE,))
    service.start([])
    first = wait_idle(service)
    assert first["state"] == setup.STATE_FAILED
    assert first["error"]["code"] == setup.VerificationFailed.code
    assert transport.starts == [0], "the corrupt file was fetched once"
    target = tree / "assets" / "transformer" / "weights.bin"
    assert not target.exists()
    # The catalogue is honest about what the failed job left behind.
    base = next(row for row in service.state()["items"] if row["id"] == setup.BASE)
    assert base["installed"] == setup.MISSING
    assert set(base["missing_files"]) == set(BASE_FILES)
    assert base["bytes_needed"] > 0

    # The second press sees a healthy server: exactly the missing files come
    # back, and the item ends installed.
    real_fetcher(base_transport())
    service.start([])
    second = wait_idle(service)
    assert second["state"] == setup.STATE_COMPLETED, second["error"]
    assert target.stat().st_size == 512
    assert service.state()["required_missing"] is False


def test_a_server_with_no_models_directory_at_all_starts_and_says_what_it_needs(
        tmp_path):
    """A fresh checkout has no ``models/``. Starting must succeed, the catalogue
    must still be readable (empty, with the reason), and the page must be told
    what to fetch — never a crash on a missing directory."""
    empty = tmp_path / "models"          # deliberately never created
    service = setup.ModelSetupService(setup.SetupRoots.for_paths(
        models_dir=str(empty / "assets"),
        manifest_path=str(empty / "manifest.json"),
        q8_manifest_path=str(empty / "q8_manifest.json")))
    runtime = api.WebRuntime(output_dir=str(tmp_path / "out"),
                             cache_dir=str(tmp_path / "cache"),
                             models_dir=str(empty / "assets"),
                             manifest_path=str(empty / "manifest.json"),
                             manifest_summary=api.summarize_manifest_bytes(None),
                             generation_enabled=True, setup_service=service)

    payload = api.status_response(runtime).payload
    assert api.MODELS_MISSING in payload["generation_blockers"]
    assert payload["generation_available"] is False
    # The rewriter, the manifest block and the rest of the page still answer.
    assert payload["planning_available"] is True
    assert payload["manifest"]["configured"] is False

    models = server.models_get_response(service, LOOP).payload
    # The items pinned in code are still offered; the manifest-derived ones are
    # not, and that is said out loud rather than shown as "nothing needed".
    assert setup.BASE not in [row["id"] for row in models["items"]]
    assert setup.PE_TEXT in [row["id"] for row in models["items"]]
    assert models["required_missing"] is True
    assert models["can_download"] is True
    # A catalogue nobody can read is said out loud, not silently omitted.
    assert any("manifest" in notice for notice in models["notices"]), \
        models["notices"]
    # A download request cannot invent an item this catalogue does not offer,
    # required or not: the id is unknown here, so the whole request is refused
    # before any byte is fetched.
    with pytest.raises(api.ApiError) as caught:
        server.models_download_response(service, download_body(["base"]), LOOP)
    assert (caught.value.status, caught.value.code) == (422, "unknown_model")
    assert service.job_state()["state"] == setup.STATE_IDLE


# ---------------------------------------------------------------------------
# The real startup path, because the whole point is that a machine with no
# models can start the server at all. Root's check ran
#   python -m qwen_engine.web --enable-generation --models-dir <EMPTY> …
# and it exited with "generation output storage or worker could not be
# initialized": models_missing is a status blocker, and WebGeneration refused
# ANY blocker, so the first-run page could never be served. These tests run
# server.main itself; only the listening socket is faked, because the guarded
# suite forbids AF_INET — every decision (argparse, the manifest read, the
# catalogue, the blockers, the real WebGeneration construction) is production
# code.
# ---------------------------------------------------------------------------


class _Listener:
    """Stand-in for the bind: records what main() wired onto the server."""

    def __init__(self, address, handler):
        self.server_address = (address[0], address[1] or 54321)
        self.bound_ip, self.bound_port = address[0], address[1] or 54321
        self.closed = False
        self.generation = None
        self.web_runtime = None

    def serve_forever(self, *args, **kwargs):
        raise KeyboardInterrupt   # the SIGTERM a real operator would send

    def server_close(self):
        self.closed = True


def _startup_argv(tmp_path, models_dir) -> list[str]:
    return ["--host", "127.0.0.1", "--port", "0", "--enable-generation",
            "--manifest", str(models_dir / "manifest.json"),
            "--models-dir", str(models_dir / "models"),
            "--output-dir", str(tmp_path / "out"),
            "--cache-dir", str(tmp_path / "cache")]


def test_a_fresh_install_starts_a_generation_server(tmp_path, monkeypatch, capsys):
    """Empty models dir + --enable-generation: main() must reach serve_forever
    with a live generation worker, not exit with an initialisation error."""
    import hashlib
    import json

    from qwen_engine.web import server as web_server

    models = tmp_path / "fresh"
    (models / "models").mkdir(parents=True)      # exists, and holds nothing
    small = b"config-json\n"
    (models / "manifest.json").write_text(json.dumps({"schema_version": 1,
                                                      "artifacts": [
        {"role": "pipeline", "repo": "Qwen/Test-Base", "revision": "a" * 40,
         "filename": "model_index.json",
         "sha256": hashlib.sha256(small).hexdigest(), "bytes": len(small)}]}))

    opened: list[_Listener] = []

    def listener(address, handler):
        instance = _Listener(address, handler)
        opened.append(instance)
        return instance

    monkeypatch.setattr(web_server, "_Server", listener)
    assert web_server.main(_startup_argv(tmp_path, models)) == 0

    assert len(opened) == 1 and opened[0].closed
    # THE regression: the worker was built even though the weights are absent.
    assert opened[0].generation is not None, "generation worker was not built"
    assert opened[0].web_runtime is not None
    assert api.MODELS_MISSING in api._blockers(opened[0].web_runtime)
    # And `--enable-generation`'s own refusal did not fire.
    err = capsys.readouterr().err
    assert "could not be initialized" not in err
    assert "generation requires valid manifest metadata" not in err
    # The operator is told what is missing, and the size named is the required
    # item's own size, not the whole catalogue's.
    assert "model setup: missing Qwen-Image-2.1 (required)" in err, err
    assert "1.2 GiB" not in err  # 12 B here; anything GiB-scale would be total


def test_a_fresh_install_still_refuses_a_real_startup_blocker(tmp_path, monkeypatch,
                                                             capsys):
    """Loosening models_missing must not loosen the others: generation enabled
    with no usable manifest is still a refusal to start, not a half-server."""
    from qwen_engine.web import server as web_server

    models = tmp_path / "broken"
    models.mkdir(parents=True)
    opened: list[_Listener] = []

    def listener(address, handler):
        instance = _Listener(address, handler)
        opened.append(instance)
        return instance

    monkeypatch.setattr(web_server, "_Server", listener)
    argv = _startup_argv(tmp_path, models)
    argv[argv.index("--manifest") + 1] = str(models / "no-such-manifest.json")
    assert web_server.main(argv) == 2
    assert opened == [], "a refused startup must leave no listener open"
    assert "generation requires valid manifest metadata" in capsys.readouterr().err


def test_a_fresh_install_refuses_a_job_until_the_weights_arrive(tmp_path):
    """The page disables Generate; the server says the same thing to a client
    that posts anyway, and unblocks the moment the files are on disk."""
    roots = setup.SetupRoots.for_paths(models_dir=str(tmp_path / "assets"),
                                       manifest_path=str(tmp_path / "none.json"),
                                       q8_manifest_path=None)
    out = tmp_path / "out"
    out.mkdir()          # main() creates this before building the worker
    service = setup.ModelSetupService(roots)
    runtime = api.WebRuntime(output_dir=str(out),
                             cache_dir=str(tmp_path / "cache"),
                             models_dir=str(tmp_path / "assets"),
                             manifest_path=str(tmp_path / "none.json"),
                             manifest_summary={"configured": True, "valid": True,
                                               "schema_version": 1,
                                               "artifact_count": 1,
                                               "verification": "not_performed"},
                             generation_enabled=True, setup_service=service)
    body = json.dumps({"schema": api.WEB_PLAN_REQUEST_SCHEMA,
                       "config": {"prompt": "a red cube", "seed": "7"}}).encode()
    generation = WebGeneration(runtime)
    try:
        with pytest.raises(api.ApiError) as caught:
            generation.submit(body)
        assert (caught.value.status, caught.value.code) == (503, "models_missing")
        assert "Settings" in caught.value.message
    finally:
        generation.close()
