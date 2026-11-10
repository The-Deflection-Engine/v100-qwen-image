"""Reference store contract: content addressing, caps, no-clobber, typed errors.

Real Pillow encodes (PNG/JPEG/WebP, RGB/RGBA/L) — the store is the upload
trust boundary, so every refusal is exercised against real image bytes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from qwen_engine import reference_store as rs

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def encode_image(width=24, height=16, mode="RGB", fmt="PNG"):
    color = {"RGB": (200, 30, 30), "RGBA": (200, 30, 30, 128), "L": 90}[mode]
    image = Image.new(mode, (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# put_bytes / resolve round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,fmt", [
    ("RGB", "PNG"), ("RGBA", "PNG"), ("L", "PNG"),
    ("RGB", "JPEG"), ("RGB", "WEBP"), ("RGBA", "WEBP"),
])
def test_put_returns_content_id_and_resolve_round_trips_bytes(tmp_path, mode, fmt):
    raw = encode_image(mode=mode, fmt=fmt)
    reference_id = rs.put_bytes(tmp_path, raw)
    assert reference_id == sha(raw)
    paths = rs.resolve(tmp_path, [reference_id])
    assert paths == (tmp_path / f"{reference_id}.img",)
    assert paths[0].read_bytes() == raw
    assert rs.read_blob(tmp_path, reference_id) == raw


def test_sidecar_describes_the_stored_image(tmp_path):
    raw = encode_image(width=31, height=17, mode="RGBA", fmt="PNG")
    reference_id = rs.put_bytes(tmp_path, raw)
    doc = rs.read_sidecar(tmp_path, reference_id)
    assert doc == {
        "schema": rs.REFERENCE_SIDECAR_SCHEMA,
        "id": reference_id, "bytes": len(raw), "format": "png",
        "width": 31, "height": 17, "mode": "RGBA", "has_alpha": True,
    }
    rgb = rs.put_bytes(tmp_path, encode_image(mode="RGB", fmt="PNG"))
    assert rs.read_sidecar(tmp_path, rgb)["has_alpha"] is False
    gray = rs.put_bytes(tmp_path, encode_image(mode="L", fmt="PNG"))
    assert rs.read_sidecar(tmp_path, gray)["mode"] == "L"


def test_thumbnail_is_bounded_png(tmp_path):
    raw = encode_image(width=900, height=300, fmt="PNG")
    reference_id = rs.put_bytes(tmp_path, raw)
    thumb = rs.read_thumb(tmp_path, reference_id)
    assert thumb.startswith(PNG_MAGIC)
    image = Image.open(io.BytesIO(thumb))
    assert max(image.size) <= rs.THUMB_MAX_DIM


def test_put_is_idempotent_and_content_addressed(tmp_path):
    raw_a, raw_b = encode_image(), encode_image(width=40)
    first = rs.put_bytes(tmp_path, raw_a)
    stat_before = os.stat(tmp_path / f"{first}.img")
    assert rs.put_bytes(tmp_path, raw_a) == first
    assert os.stat(tmp_path / f"{first}.img") == stat_before  # no rewrite
    second = rs.put_bytes(tmp_path, raw_b)
    assert rs.list_ids(tmp_path) == tuple(sorted([first, second]))
    assert rs.usage(tmp_path) == (2, len(raw_a) + len(raw_b))


# ---------------------------------------------------------------------------
# input refusal: format, caps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,match", [
    (b"", "non-empty bytes"),
    (b"GIF89a" + b"\x00" * 64, "PNG, JPEG or WebP"),
    (b"BMP" + b"\x00" * 64, "PNG, JPEG or WebP"),
    (b"\xff\xd8\xff\xe0not really jpeg", "opened|decoded|corrupt|cannot"),
])
def test_non_images_refuse_typed(tmp_path, raw, match):
    with pytest.raises(rs.ReferenceStoreError, match=match):
        rs.put_bytes(tmp_path, raw)
    assert list(tmp_path.iterdir()) == []


def test_byte_cap_is_a_refusal_not_a_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MAX_REFERENCE_BYTES", 64)
    raw = encode_image(fmt="PNG")
    assert len(raw) > 64
    with pytest.raises(rs.ReferenceStoreError, match="exceeds 64 bytes"):
        rs.put_bytes(tmp_path, raw)
    assert list(tmp_path.iterdir()) == []


def test_dimension_caps_refuse_before_full_decode(tmp_path, monkeypatch):
    tall = encode_image(width=1, height=4097, fmt="PNG")
    assert 1 * 4097 < rs.MAX_REFERENCE_PIXELS  # dimension rule, not pixel budget
    with pytest.raises(rs.ReferenceStoreError, match="at most 4096 px"):
        rs.put_bytes(tmp_path, tall)
    monkeypatch.setattr(rs, "MAX_REFERENCE_PIXELS", 100)
    with pytest.raises(rs.ReferenceStoreError, match="pixel budget"):
        rs.put_bytes(tmp_path, encode_image(width=20, height=20, fmt="PNG"))
    assert list(tmp_path.iterdir()) == []


def test_store_caps_refuse_instead_of_evicting(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "MAX_STORE_BLOBS", 2)
    ids = [rs.put_bytes(tmp_path, encode_image(width=w)) for w in (10, 11)]
    with pytest.raises(rs.ReferenceStoreError, match="store is full"):
        rs.put_bytes(tmp_path, encode_image(width=12))
    assert rs.list_ids(tmp_path) == tuple(sorted(ids))  # nothing evicted
    # a duplicate of stored content is accepted even when the store is full
    assert rs.put_bytes(tmp_path, encode_image(width=10)) == ids[0]
    monkeypatch.setattr(rs, "MAX_STORE_BLOBS", 512)
    monkeypatch.setattr(rs, "MAX_STORE_BYTES", 10)
    with pytest.raises(rs.ReferenceStoreError, match="store is full"):
        rs.put_bytes(tmp_path, encode_image(width=13))


# ---------------------------------------------------------------------------
# integrity: tamper, substitution, id grammar
# ---------------------------------------------------------------------------


BAD_IDS = ["", "x" * 64, "0" * 63, "0" * 64 + "g", "../" + "0" * 62 + "ff",
           "A" * 64, "0" * 63 + "A", "0" * 64 + ".img",
           "/etc/passwd", None, 7]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_id_grammar_is_strict_everywhere(bad):
    with pytest.raises(rs.ReferenceStoreError):
        rs.path_for("/tmp/store", bad)
    with pytest.raises(rs.ReferenceStoreError):
        rs.resolve("/tmp/store", [bad])
    with pytest.raises(rs.ReferenceStoreError):
        rs.read_blob("/tmp/store", bad)
    with pytest.raises(rs.ReferenceStoreError):
        rs.read_sidecar("/tmp/store", bad)


def test_resolve_refuses_a_bare_string_of_ids(tmp_path):
    with pytest.raises(rs.ReferenceStoreError, match="tuple/list"):
        rs.resolve(tmp_path, "0" * 64)


def test_tampered_blob_fails_closed_everywhere(tmp_path):
    raw = encode_image()
    reference_id = rs.put_bytes(tmp_path, raw)
    blob = tmp_path / f"{reference_id}.img"
    blob.write_bytes(encode_image(width=25))
    with pytest.raises(rs.ReferenceStoreError, match="no longer matches its id"):
        rs.read_blob(tmp_path, reference_id)
    with pytest.raises(rs.ReferenceStoreError, match="no longer matches its id"):
        rs.resolve(tmp_path, [reference_id])
    with pytest.raises(rs.ReferenceStoreError, match="no longer matches its id"):
        rs.put_bytes(tmp_path, raw)  # dedupe path re-validates before reuse


def test_symlinked_blob_refused(tmp_path):
    raw = encode_image()
    reference_id = rs.put_bytes(tmp_path, raw)
    blob = tmp_path / f"{reference_id}.img"
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(raw)
    blob.unlink()
    blob.symlink_to(outside)
    with pytest.raises(rs.ReferenceStoreError, match="cannot read"):
        rs.read_blob(tmp_path, reference_id)
    # usage() counts regular files only; the symlink is not stored bytes
    assert rs.usage(tmp_path) == (0, 0)


def test_missing_blob_is_a_typed_miss(tmp_path):
    with pytest.raises(rs.ReferenceStoreError, match="cannot read"):
        rs.resolve(tmp_path, ["a" * 64])


def test_sidecar_identity_must_match_the_request(tmp_path):
    reference_id = rs.put_bytes(tmp_path, encode_image())
    sidecar = tmp_path / f"{reference_id}.json"
    forged = json.loads(sidecar.read_text())
    forged["id"] = "b" * 64
    sidecar.write_text(json.dumps(forged, sort_keys=True) + "\n")
    with pytest.raises(rs.ReferenceStoreError, match="disagrees"):
        rs.read_sidecar(tmp_path, reference_id)


def test_list_ids_and_usage_ignore_foreign_names(tmp_path):
    reference_id = rs.put_bytes(tmp_path, encode_image())
    (tmp_path / "deadbeef.img").write_bytes(b"junk")
    (tmp_path / ("f" * 64 + ".img.tmp-123")).write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("hi")
    assert rs.list_ids(tmp_path) == (reference_id,)
    expected_bytes = (tmp_path / f"{reference_id}.img").stat().st_size
    assert rs.usage(tmp_path) == (1, expected_bytes)


def test_is_reference_id_and_path_for_stay_off_the_filesystem(tmp_path):
    good = "a" * 64
    assert rs.is_reference_id(good) and not rs.is_reference_id("A" * 64)
    assert not rs.is_reference_id(".." + "/a" * 64)
    assert not rs.is_reference_id(None)
    path = rs.path_for(tmp_path, good)
    assert path == tmp_path / f"{good}.img"
    assert not path.exists()  # path_for never touches the filesystem


def test_root_must_be_an_existing_directory(tmp_path):
    with pytest.raises(rs.ReferenceStoreError):
        rs.list_ids(17)
    with pytest.raises(rs.ReferenceStoreError):
        rs.put_bytes("", encode_image())
    with pytest.raises(rs.ReferenceStoreError, match="existing directory"):
        rs.list_ids(tmp_path / "never-created")


# ---------------------------------------------------------------------------
# Owner index (C3): a shared, content-addressed store with per-owner edges
# ---------------------------------------------------------------------------


def _shared(tmp_path):
    raw = encode_image()
    return raw, rs.put_bytes(tmp_path, raw)


def test_an_edge_is_idempotent_and_scoped_to_its_owner(tmp_path):
    _, reference_id = _shared(tmp_path)
    assert rs.owners_for(tmp_path, reference_id) == ()
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    rs.add_owner(tmp_path, "sub-alice", reference_id)   # idempotent
    assert rs.owners_for(tmp_path, reference_id) == ("sub-alice",)
    assert rs.owner_ids(tmp_path, "sub-alice") == (reference_id,)
    assert rs.owner_ids(tmp_path, "sub-bob") == ()


def test_an_edge_to_bytes_the_store_does_not_hold_is_refused(tmp_path):
    with pytest.raises(rs.ReferenceStoreError, match="not stored"):
        rs.add_owner(tmp_path, "sub-alice", "0" * 64)


@pytest.mark.parametrize("owner", ["", "a b", "a\nb", 17, None, "x" * 201])
def test_an_owner_id_has_one_grammar(tmp_path, owner):
    _, reference_id = _shared(tmp_path)
    with pytest.raises(rs.ReferenceStoreError):
        rs.add_owner(tmp_path, owner, reference_id)


def test_owner_ids_skip_an_id_whose_blob_was_removed(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    os.unlink(tmp_path / (reference_id + rs.BLOB_SUFFIX))
    # The edge survives on disk, but an id the store cannot serve is not
    # offered to the page: the UI would only get a 404 from it.
    assert rs.owner_ids(tmp_path, "sub-alice") == ()


def test_forgetting_one_owner_keeps_a_blob_another_owner_still_holds(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    rs.add_owner(tmp_path, "sub-bob", reference_id)
    released, deleted = rs.forget_owner(tmp_path, "sub-alice")
    assert released == (reference_id,) and deleted == 0
    assert rs.owners_for(tmp_path, reference_id) == ("sub-bob",)
    assert rs.owner_ids(tmp_path, "sub-bob") == (reference_id,)
    assert rs.read_blob(tmp_path, reference_id) == encode_image()
    assert (tmp_path / (reference_id + rs.BLOB_SUFFIX)).exists()


def test_forgetting_the_last_owner_deletes_blob_sidecar_and_thumbnail(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    released, deleted = rs.forget_owner(tmp_path, "sub-alice")
    assert released == (reference_id,) and deleted == 3
    for suffix in (rs.BLOB_SUFFIX, rs.SIDECAR_SUFFIX, rs.THUMB_SUFFIX):
        assert not (tmp_path / (reference_id + suffix)).exists()
    assert rs.owners_for(tmp_path, reference_id) == ()
    assert rs.list_ids(tmp_path) == ()


def test_forgetting_an_owner_with_no_edges_is_a_no_op(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-bob", reference_id)
    assert rs.forget_owner(tmp_path, "sub-alice") == ((), 0)
    assert rs.owners_for(tmp_path, reference_id) == ("sub-bob",)


def test_forgetting_treats_a_hand_deleted_blob_as_already_gone(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    os.unlink(tmp_path / (reference_id + rs.SIDECAR_SUFFIX))
    released, deleted = rs.forget_owner(tmp_path, "sub-alice")
    assert released == (reference_id,) and deleted == 2
    assert not (tmp_path / (reference_id + rs.BLOB_SUFFIX)).exists()


def test_the_owner_index_is_a_closed_document(tmp_path):
    _, reference_id = _shared(tmp_path)
    rs.add_owner(tmp_path, "sub-alice", reference_id)
    path = rs._owners_path(tmp_path)
    assert path.exists()
    document = json.loads(path.read_text())
    assert document["schema"] == rs.OWNERS_SCHEMA
    assert document["owners"] == {reference_id: ["sub-alice"]}


@pytest.mark.parametrize("payload", [
    b"not json", b"[]", b'{"schema": "wrong", "owners": {}}',
    b'{"schema": "%s", "owners": []}' % rs.OWNERS_SCHEMA.encode(),
    b'{"schema": "%s", "owners": {"%s": "sub-a"}}' % (
        rs.OWNERS_SCHEMA.encode(), b"0" * 64),
])
def test_a_corrupt_owner_index_is_a_typed_refusal(tmp_path, payload):
    _, reference_id = _shared(tmp_path)
    rs._owners_path(tmp_path).write_bytes(payload)
    with pytest.raises(rs.ReferenceStoreError):
        rs.owners_for(tmp_path, reference_id)


def test_the_owner_index_refuses_an_id_with_too_many_owners(tmp_path, monkeypatch):
    _, reference_id = _shared(tmp_path)
    monkeypatch.setattr(rs, "MAX_OWNERS_PER_ID", 2)
    rs.add_owner(tmp_path, "sub-a", reference_id)
    rs.add_owner(tmp_path, "sub-b", reference_id)
    with pytest.raises(rs.ReferenceStoreError, match="owners"):
        rs.add_owner(tmp_path, "sub-c", reference_id)
