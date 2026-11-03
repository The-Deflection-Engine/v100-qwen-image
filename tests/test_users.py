"""C2: the invited-person allowlist (users.json) — shape, atomicity, 0600, refusals."""
from __future__ import annotations

import json
import os
import stat

import pytest

from qwen_engine import users as users_module
from qwen_engine.users import (USERS_SCHEMA, UserStore, UserStoreError,
                               normalise_email, users_path)


@pytest.fixture
def store(tmp_path):
    (tmp_path / "cache").mkdir()
    return UserStore(users_path(str(tmp_path / "cache")))


def read_document(store):
    with open(store.path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


# ----------------------------------------------------------------- C2 shape

def test_a_fresh_store_invites_nobody_and_creates_no_file(store):
    assert store.path.endswith("users.json")
    assert store.list_users() == []
    assert store.is_invited("alice@example.com") is False
    assert not os.path.exists(store.path)


def test_add_writes_the_documented_document_at_0600(store):
    invited = store.add("  Alice@Example.COM ")
    assert set(invited) == {"email", "sub", "added", "display_name", "allow"}
    assert invited["email"] == "alice@example.com"
    assert invited["sub"] is None
    assert invited["display_name"] == ""
    # C2 defaults: q8 on, everything else off.
    assert invited["allow"] == {"q8": True, "2k": False, "edit": False,
                                "official_enhancer": False}
    document = read_document(store)
    assert set(document) == {"schema", "users"}
    assert document["schema"] == USERS_SCHEMA
    assert set(document["users"]) == {"alice@example.com"}
    record = document["users"]["alice@example.com"]
    assert set(record) == {"sub", "added", "display_name", "allow"}
    assert record["sub"] is None
    assert record["allow"] == {"q8": True, "2k": False, "edit": False,
                               "official_enhancer": False}
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600, oct(mode)


def test_the_file_replaces_atomically_leaving_no_temp_files(store):
    store.add("alice@example.com")
    store.add("bob@example.com")
    store.set_permissions("bob@example.com", edit=True)
    directory = os.path.dirname(store.path)
    assert sorted(os.listdir(directory)) == ["users.json"]


def test_add_is_idempotent_and_keeps_permissions_and_binding(store):
    store.add("alice@example.com")
    store.set_permissions("alice@example.com", edit=True)
    assert store.bind_sub("alice@example.com", "123") is True
    again = store.add("ALICE@example.com")
    assert again["allow"]["edit"] is True
    assert again["sub"] == "123"


def test_permissions_are_a_typed_refusal_for_unknown_flags_and_non_bools(store):
    store.add("alice@example.com")
    for flags in ({"turbo": True}, {"q8": "yes"}, {"2k": 1}):
        with pytest.raises(UserStoreError):
            store.set_permissions("alice@example.com", **flags)
    with pytest.raises(UserStoreError):
        store.set_permissions("nobody@example.com", q8=True)


def test_permissions_turn_things_on_and_off(store):
    store.add("alice@example.com")
    assert store.permissions("alice@example.com")["edit"] is False
    store.set_permissions("alice@example.com", edit=True, **{"2k": True})
    assert store.permissions("alice@example.com") == {
        "q8": True, "2k": True, "edit": True, "official_enhancer": False}
    # An unknown address is all-denied, never an exception.
    assert store.permissions("nobody@example.com") == {
        "q8": False, "2k": False, "edit": False, "official_enhancer": False}


def test_display_name_is_bounded_printable_text(store):
    store.add("alice@example.com")
    assert store.set_display_name("alice@example.com",
                                  "  Alice   A ")["display_name"] == "Alice A"
    with pytest.raises(UserStoreError):
        store.set_display_name("alice@example.com", "x" * 500)
    with pytest.raises(UserStoreError):
        store.set_display_name("alice@example.com", "bad\x07name")
    with pytest.raises(UserStoreError):
        store.set_display_name("nobody@example.com", "Alice")


def test_remove_reports_whether_the_address_was_on_the_list(store):
    store.add("alice@example.com")
    assert store.remove("alice@example.com") is True
    assert store.remove("alice@example.com") is False
    with pytest.raises(UserStoreError):
        store.remove("alice@example.com", delete_data="yes")
    assert store.list_users() == []


# -------------------------------------------------------------- bind_sub

def test_bind_sub_records_the_first_account_id_and_refuses_a_different_one(store):
    store.add("alice@example.com")
    assert store.bind_sub("alice@example.com", "123") is True
    assert store.bind_sub("alice@example.com", "123") is True
    # A recycled address must never reach the previous person's data.
    assert store.bind_sub("alice@example.com", "456") is False
    assert read_document(store)["users"]["alice@example.com"]["sub"] == "123"
    with pytest.raises(UserStoreError):
        store.bind_sub("nobody@example.com", "123")


def test_email_for_sub_finds_the_bound_address(store):
    store.add("alice@example.com")
    store.bind_sub("alice@example.com", "123")
    assert store.email_for_sub("123") == "alice@example.com"
    assert store.email_for_sub("999") is None


# ----------------------------------------------------------- persistence

def test_a_new_store_reads_the_file_back(store):
    store.add("alice@example.com")
    store.set_permissions("alice@example.com", **{"2k": True, "official_enhancer": True})
    store.set_display_name("alice@example.com", "Alice")
    store.bind_sub("alice@example.com", "123")
    reopened = UserStore(store.path)
    invited = reopened.get("alice@example.com")
    assert invited is not None
    assert invited["display_name"] == "Alice"
    assert invited["sub"] == "123"
    assert invited["allow"] == {"q8": True, "2k": True, "edit": False,
                                "official_enhancer": True}


@pytest.mark.parametrize("payload", [
    b"",                                                        # empty file
    b"not json",                                                # not JSON
    b'{"schema": "qwen_engine.users.v1"}',                      # missing users
    b'{"schema": "other", "users": {}}',                        # unknown schema
    b'{"schema": "qwen_engine.users.v1", "users": [], "x": 1}',  # not an object
    b'{"schema": "qwen_engine.users.v1", "users": {"a@b.c": {}}}',  # bad record
    (b'{"schema": "qwen_engine.users.v1", "users": {"Alice@b.c": '
     b'{"sub": null, "added": "2026-10-27T10:00:00Z", "display_name": "", '
     b'"allow": {"q8": true, "2k": false, "edit": false, "official_enhancer": false}}}}'),
])
def test_a_corrupt_or_foreign_file_is_a_typed_refusal_never_a_crash(tmp_path, payload):
    path = tmp_path / "users.json"
    path.write_bytes(payload)
    with pytest.raises(UserStoreError):
        UserStore(str(path))


def test_an_unset_cache_directory_invites_nobody_and_refuses_writes():
    store = UserStore(None)
    assert store.path is None
    assert store.is_invited("alice@example.com") is False
    with pytest.raises(UserStoreError):
        store.add("alice@example.com")


def test_email_normalization_is_lowercase_and_trimmed():
    # Leading/trailing whitespace (a trailing newline included) is not part of
    # the address: the stored key and the allowlist lookup use this one form.
    assert normalise_email("  Alice@Example.COM\n") == "alice@example.com"


@pytest.mark.parametrize("bad", ["", "   ", "no-at-sign", "a@", "@b.c", "a b@c.d",
                                 "alice@example", "ünicode@example.com"])
def test_unusable_addresses_are_typed_refusals(bad):
    with pytest.raises(UserStoreError):
        normalise_email(bad)


def test_list_users_is_sorted_by_email(store):
    store.add("zoe@example.com")
    store.add("alice@example.com")
    assert [u["email"] for u in store.list_users()] == ["alice@example.com",
                                                        "zoe@example.com"]


def test_the_store_refuses_a_list_beyond_its_bound(store, monkeypatch):
    monkeypatch.setattr(users_module, "MAX_USERS", 1)
    store.add("alice@example.com")
    with pytest.raises(UserStoreError):
        store.add("bob@example.com")
    assert len(store.list_users()) == 1


# ------------------------------------------------------- the C2 surface (D3)

def test_the_store_answers_the_exact_surface_the_admin_routes_read(store):
    """D3's admin.py reads rows as dicts and imports these names.

    Pinned here because it is a cross-worktree contract, not a local choice:
    the admin page's document builder does ``entry["sub"]``,
    ``entry["added"]``, ``entry.get("first_sign_in")`` and imports
    ``normalise_email``/``normalise_display_name``/``PERMISSIONS``/``MAX_USERS``.
    """
    assert users_module.PERMISSIONS == ("q8", "2k", "edit", "official_enhancer")
    assert users_module.MAX_USERS == 500
    # The admin page's display-name input carries maxLength=80.
    assert users_module.MAX_DISPLAY_NAME_CHARS == 80
    assert callable(users_module.normalise_email)
    assert callable(users_module.normalise_display_name)

    created = store.add("alice@example.com")
    listed = store.list_users()
    assert isinstance(listed, list) and all(isinstance(row, dict) for row in listed)
    for row in listed:
        assert set(row) == {"email", "sub", "added", "display_name", "allow"}
    # ``list_users`` is sorted and each row is a COPY of the stored record.
    listed[0]["allow"]["edit"] = True
    assert store.permissions("alice@example.com")["edit"] is False
    assert store.get("alice@example.com") == created


def test_a_display_name_the_admin_page_accepts_is_accepted_here(store):
    store.add("alice@example.com")
    # 80 characters is exactly the page's limit: accepted, not refused.
    assert store.set_display_name("alice@example.com", "n" * 80)["display_name"] == "n" * 80
    with pytest.raises(UserStoreError):
        store.set_display_name("alice@example.com", "n" * 81)


def test_bind_sub_reports_a_mismatch_without_ever_rebinding(store):
    """C2's rule is "first sign-in binds, later ones must match"; a mismatch
    must be a distinct answer, never a silent re-bind."""
    store.add("alice@example.com")
    assert store.bind_sub("alice@example.com", "123") is True
    assert store.bind_sub("alice@example.com", "456") is False
    # The refusal keeps the FIRST binding, so the previous person's data stays
    # unreachable through the recycled address.
    assert store.get("alice@example.com")["sub"] == "123"


def test_removal_blocks_auto_add_until_invited_again(store):
    store.add("ann@example.com")
    assert store.remove("ann@example.com") is True
    assert store.is_blocked("ann@example.com")
    assert store.auto_add("ann@example.com") is False
    assert not store.is_invited("ann@example.com")
    # The block survives a restart.
    reloaded = UserStore(store.path)
    assert reloaded.is_blocked("ann@example.com")
    # Inviting by hand lifts it.
    reloaded.add("ann@example.com")
    assert not reloaded.is_blocked("ann@example.com")
    assert reloaded.is_invited("ann@example.com")
    assert "blocked" not in read_document(reloaded)


def test_auto_add_invites_a_new_address_with_defaults(store):
    assert store.auto_add("New@Example.com") is True
    row = store.get("new@example.com")
    assert row["allow"] == dict(users_module.DEFAULT_ALLOW)


def test_a_file_listing_an_address_as_both_invited_and_blocked_is_refused(store):
    store.add("ann@example.com")
    document = read_document(store)
    document["blocked"] = ["ann@example.com"]
    with open(store.path, "w") as handle:
        json.dump(document, handle)
    with pytest.raises(UserStoreError):
        UserStore(store.path)
