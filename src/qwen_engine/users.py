"""Invited people and their feature permissions (owner: the local admin).

The file is ``<cache>/users.json``, mode 0600, written atomically (temp file in
the same directory + ``os.replace``). It holds the allowlist of invited emails
and the per-user feature flags the server enforces on the remote port; it holds
no secrets and no job data (jobs and references carry their own owner tag).

Every value that reaches the wire or the disk is validated here, so the server
never has to trust the file: a document whose schema tag, key set, email shape
or flag types disagree with this module's contract is a typed
:class:`UserStoreError` at load, never a silently half-adopted allowlist.

With automatic invitations on (``--remote-auto-invite``), anyone who gets
through Google sign-in is added on first visit. Removing a person then also
records them under ``blocked`` so they are not re-added on their next visit;
inviting the address again by hand lifts the block.

Identity is the Google account ``sub``: the first sign-in records it
(:meth:`UserStore.bind_sub`) and every later sign-in must present the same one,
so an email that changes hands at the provider can never inherit the previous
person's stored jobs.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

USERS_SCHEMA = "qwen_engine.users.v1"
USERS_FILENAME = "users.json"

#: Closed feature-flag set, in wire order. ``q8`` is on for a new invite
#: because the quantized checkpoint is the ordinary way to run on 16 GB;
#: everything that costs noticeably more (2K, editing, the official enhancer
#: that occupies the whole card) is opt-in per person.
PERMISSIONS = ("q8", "2k", "edit", "official_enhancer")
DEFAULT_ALLOW: Mapping[str, bool] = MappingProxyType(
    {"q8": True, "2k": False, "edit": False, "official_enhancer": False})
#: What a caller that asks about an email nobody invited gets: every feature
#: denied. Fail closed — the server refuses a non-invited request before it
#: reads permissions, and this keeps a future caller from granting one by
#: accident.
_UNKNOWN_PERMISSIONS: Mapping[str, bool] = MappingProxyType(
    {flag: False for flag in PERMISSIONS})

#: The admin page's input carries ``maxLength = 80``; the store accepts
#: exactly what the page lets a person type.
MAX_DISPLAY_NAME_CHARS = 80
MAX_SUB_CHARS = 255
MAX_EMAIL_CHARS = 254
#: A bounded allowlist: this is one person's invite list, not a directory.
MAX_USERS = 500
#: The document is a few hundred bytes per invited person; anything far past
#: this is corruption, not a big list.
MAX_USERS_BYTES = 1 << 20

_USER_KEYS = frozenset({"sub", "added", "display_name", "allow"})
#: Lowercase ASCII local part @ lowercase ASCII domain with at least one dot.
#: Deliberately strict: the email is an identity key, and a normalized form
#: that can be spelled two ways would be two different people.
_EMAIL_RE = re.compile(r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+"
                       r"(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
                       r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
                       r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\Z")
_ADDED_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


class UserStoreError(ValueError):
    """Typed refusal of the user store (bad email, bad flag, corrupt file).

    Messages are fixed diagnostics: a stored document may hold arbitrary text
    and no offending value is echoed back into one.
    """


def normalise_email(value: Any) -> str:
    """Trim + lowercase, or a typed refusal. The ONE normalizer: the store,
    the remote identity check and the admin routes all call it, so an address
    can never be spelled two ways between them."""
    if not isinstance(value, str):
        raise UserStoreError("an email address must be a string")
    text = value.strip().lower()
    if not text or len(text) > MAX_EMAIL_CHARS or not text.isascii() \
            or _EMAIL_RE.fullmatch(text) is None:
        raise UserStoreError("that is not a usable email address")
    return text


def _normalise_sub(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_SUB_CHARS:
        raise UserStoreError("a provider account id must be a bounded non-empty string")
    if not value.isprintable() or any(ch.isspace() for ch in value):
        raise UserStoreError("a provider account id must be printable without whitespace")
    return value


def normalise_display_name(value: Any) -> str:
    if not isinstance(value, str):
        raise UserStoreError("a display name must be a string")
    text = " ".join(value.split())
    if len(text) > MAX_DISPLAY_NAME_CHARS:
        raise UserStoreError(
            f"a display name is at most {MAX_DISPLAY_NAME_CHARS} characters")
    if any(not ch.isprintable() for ch in text):
        raise UserStoreError("a display name must be printable text")
    return text


def users_path(cache_dir: str | None) -> str | None:
    """Where the allowlist lives, or ``None`` when this server has no cache
    directory (then nobody is invited and the remote port has no users)."""
    if not cache_dir:
        return None
    return os.path.join(cache_dir, USERS_FILENAME)


def _now_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row(email: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """One invited person as a fresh, JSON-able dict (C2's documented row).

    A copy, not a view: the store keeps one mutable document under its lock,
    and a caller that mutated a row must never reach back into it. ``allow`` is
    rebuilt in :data:`PERMISSIONS` order so the wire order is fixed.
    """
    return {"email": email, "sub": record["sub"], "added": record["added"],
            "display_name": record["display_name"],
            "allow": {flag: bool(record["allow"][flag]) for flag in PERMISSIONS}}


def _validate_record(email: str, record: Any) -> dict[str, Any]:
    if type(record) is not dict or set(record) != _USER_KEYS:
        raise UserStoreError("a user record has an unexpected field set")
    sub = record["sub"]
    if sub is not None:
        sub = _normalise_sub(sub)
    added = record["added"]
    if not isinstance(added, str) or _ADDED_RE.fullmatch(added) is None:
        raise UserStoreError("a user record's added stamp is not UTC ISO-8601")
    display_name = normalise_display_name(record["display_name"])
    allow = record["allow"]
    if type(allow) is not dict or set(allow) != set(PERMISSIONS):
        raise UserStoreError("a user record's permission set is unknown")
    for flag in PERMISSIONS:
        if type(allow[flag]) is not bool:
            raise UserStoreError("a user permission is not a boolean")
    if _EMAIL_RE.fullmatch(email) is None or email != email.strip().lower():
        raise UserStoreError("a stored email is not in its normalized form")
    return {"sub": sub, "added": added, "display_name": display_name,
            "allow": {flag: allow[flag] for flag in PERMISSIONS}}


class UserStore:
    """The invited-person allowlist and their feature flags.

    Construction reads the file once; every mutation validates, then rewrites
    the whole document atomically at 0600. All access is serialized on one
    lock, so the multi-threaded request pool can share one instance. The
    directory must already exist (the server's cache dir); the store never
    creates or chmods anything outside its own file.
    """

    def __init__(self, path: str | os.PathLike | None) -> None:
        self._lock = threading.RLock()
        self._path = None if path is None else os.fspath(path)
        self._users: dict[str, dict[str, Any]] = {}
        self._blocked: frozenset[str] = frozenset()
        if self._path is not None:
            self._users, self._blocked = self._load()

    # -- reading ------------------------------------------------------------

    @property
    def path(self) -> str | None:
        return self._path

    def _load(self) -> tuple[dict[str, dict[str, Any]], frozenset[str]]:
        try:
            with open(self._path, "rb") as handle:
                raw = handle.read(MAX_USERS_BYTES + 1)
        except FileNotFoundError:
            return {}, frozenset()
        except OSError as exc:
            raise UserStoreError(
                f"the user list is unreadable ({exc.strerror or 'storage error'})") from None
        if len(raw) > MAX_USERS_BYTES:
            raise UserStoreError("the user list is larger than its bounded size")
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise UserStoreError("the user list is not valid UTF-8 JSON") from None
        if type(document) is not dict or set(document) not in (
                {"schema", "users"}, {"schema", "users", "blocked"}):
            raise UserStoreError("the user list has an unexpected field set")
        if document["schema"] != USERS_SCHEMA:
            raise UserStoreError("the user list schema is unknown")
        users = document["users"]
        if type(users) is not dict:
            raise UserStoreError("the user list is not an object of users")
        if len(users) > MAX_USERS:
            raise UserStoreError("the user list exceeds its bounded size")
        loaded = {email: _validate_record(email, record)
                  for email, record in users.items()}
        blocked = document.get("blocked", [])
        if type(blocked) is not list or len(blocked) > MAX_USERS:
            raise UserStoreError("the blocked list is not a bounded list")
        for email in blocked:
            if not isinstance(email, str) or _EMAIL_RE.fullmatch(email) is None \
                    or email != email.strip().lower():
                raise UserStoreError("a blocked email is not in its normalized form")
            if email in loaded:
                raise UserStoreError("an address is both invited and blocked")
        return loaded, frozenset(blocked)

    def list_users(self) -> list[dict[str, Any]]:
        """Every invited person, in email order, as plain JSON-able dicts."""
        with self._lock:
            return [_row(email, record)
                    for email, record in sorted(self._users.items())]

    def get(self, email: str) -> dict[str, Any] | None:
        """One invited person, or ``None`` — never an exception for absence
        (the caller decides what absence means)."""
        normalized = normalise_email(email)
        with self._lock:
            record = self._users.get(normalized)
            return None if record is None else _row(normalized, record)

    def is_invited(self, email: str) -> bool:
        """True exactly for a normalized, invited address (no exception: an
        unusable address is simply not invited)."""
        try:
            normalized = normalise_email(email)
        except UserStoreError:
            return False
        with self._lock:
            return normalized in self._users

    def is_blocked(self, email: str) -> bool:
        """True for an address removed by the admin (and not re-invited)."""
        try:
            normalized = normalise_email(email)
        except UserStoreError:
            return False
        with self._lock:
            return normalized in self._blocked

    def permissions(self, email: str) -> dict[str, bool]:
        """The feature flags of an invited person; all-denied when the address
        is unknown, so a caller cannot grant a feature by forgetting a check."""
        try:
            normalized = normalise_email(email)
        except UserStoreError:
            return dict(_UNKNOWN_PERMISSIONS)
        with self._lock:
            record = self._users.get(normalized)
            if record is None:
                return dict(_UNKNOWN_PERMISSIONS)
            return {flag: bool(record["allow"][flag]) for flag in PERMISSIONS}

    def email_for_sub(self, sub: str) -> str | None:
        """The invited email that owns this provider account id, if any."""
        wanted = _normalise_sub(sub)
        with self._lock:
            for email, record in self._users.items():
                if record["sub"] == wanted:
                    return email
        return None

    # -- writing ------------------------------------------------------------

    def add(self, email: str) -> dict[str, Any]:
        """Invite an address (idempotent: an existing person keeps their
        permissions and their bound account id). Lifts any block."""
        normalized = normalise_email(email)
        with self._lock:
            existing = self._users.get(normalized)
            if existing is not None:
                return _row(normalized, existing)
            if len(self._users) >= MAX_USERS:
                raise UserStoreError("the user list is full")
            record = {"sub": None, "added": _now_stamp(), "display_name": "",
                      "allow": {flag: DEFAULT_ALLOW[flag]
                                for flag in PERMISSIONS}}
            self._publish({**self._users, normalized: record},
                          self._blocked - {normalized})
            return _row(normalized, record)

    def auto_add(self, email: str) -> bool:
        """Invite an address on its first sign-in unless the admin removed it.

        ``True`` when the address is (now) invited. A new account gets the
        same defaults as a hand-made invite (Q8 only).
        """
        normalized = normalise_email(email)
        with self._lock:
            if normalized in self._users:
                return True
            if normalized in self._blocked:
                return False
            self.add(normalized)
            return True

    def set_permissions(self, email: str, **flags: bool) -> dict[str, Any]:
        """Change some flags, leaving the others alone.

        Only :data:`PERMISSIONS` may be named and each value must be an
        exact bool: a typo is a typed refusal instead of a silently ignored
        field.
        """
        normalized = normalise_email(email)
        unknown = sorted(set(flags) - set(PERMISSIONS))
        if unknown:
            raise UserStoreError(
                "unknown feature flag(s): " + ", ".join(unknown))
        for flag, value in flags.items():
            if type(value) is not bool:
                raise UserStoreError(f"feature flag {flag!r} must be a boolean")
        with self._lock:
            record = self._users.get(normalized)
            if record is None:
                raise UserStoreError("that email is not invited")
            allow = {**record["allow"], **flags}
            if allow == record["allow"]:
                return _row(normalized, record)
            updated = {**record, "allow": allow}
            self._publish({**self._users, normalized: updated})
            return _row(normalized, updated)

    def set_display_name(self, email: str, name: str) -> dict[str, Any]:
        """Set the name the UI greets this person by (bounded printable text)."""
        normalized = normalise_email(email)
        cleaned = normalise_display_name(name)
        with self._lock:
            record = self._users.get(normalized)
            if record is None:
                raise UserStoreError("that email is not invited")
            if record["display_name"] == cleaned:
                return _row(normalized, record)
            updated = {**record, "display_name": cleaned}
            self._publish({**self._users, normalized: updated})
            return _row(normalized, updated)

    def bind_sub(self, email: str, sub: str) -> bool:
        """Record the provider account id on first sign-in; later sign-ins must
        present the same one.

        Returns ``True`` when the pair is (now) consistent and ``False`` when
        this account id disagrees with the one already stored — the caller
        refuses the sign-in rather than letting a recycled email address reach
        the previous person's history.
        """
        normalized = normalise_email(email)
        wanted = _normalise_sub(sub)
        with self._lock:
            record = self._users.get(normalized)
            if record is None:
                raise UserStoreError("that email is not invited")
            stored = record["sub"]
            if stored is None:
                self._publish({**self._users,
                               normalized: {**record, "sub": wanted}})
                return True
            return stored == wanted

    def remove(self, email: str, *, delete_data: bool = False) -> bool:
        """Un-invite an address. ``True`` when it was on the list.

        ``delete_data`` states what the CALLER is about to do with the person's
        jobs and reference index entries — this module owns neither, so it
        only validates the flag and never touches them here; the server applies
        the same flag to :meth:`JobController.forget_owner` and the reference
        owner index. It is required to be an exact bool so an admin route can
        never pass a truthy string by accident.
        """
        if type(delete_data) is not bool:
            raise UserStoreError("delete_data must be boolean")
        normalized = normalise_email(email)
        with self._lock:
            if normalized not in self._users:
                return False
            remaining = {key: value for key, value in self._users.items()
                         if key != normalized}
            blocked = self._blocked
            if len(blocked) < MAX_USERS:
                blocked = blocked | {normalized}
            self._publish(remaining, blocked)
            return True

    # -- persistence --------------------------------------------------------

    def _document(self, users: Mapping[str, Mapping[str, Any]],
                  blocked: frozenset[str]) -> dict[str, Any]:
        document = {"schema": USERS_SCHEMA,
                    "users": {email: {key: (dict(value) if key == "allow" else value)
                                      for key, value in record.items()}
                              for email, record in users.items()}}
        # Written only when non-empty, so a list nobody was ever removed from
        # keeps the original two-key shape.
        if blocked:
            document["blocked"] = sorted(blocked)
        return document

    def _publish(self, users: dict[str, dict[str, Any]],
                 blocked: frozenset[str] | None = None) -> None:
        """Write the whole document atomically, then adopt it.

        The in-memory copy is replaced ONLY after the file took the new
        document: a save that fails must not leave the running server believing
        something the next start-up would not read back.
        """
        if self._path is None:
            raise UserStoreError("this server has no cache directory to store users in")
        if blocked is None:
            blocked = self._blocked
        payload = json.dumps(self._document(users, blocked), sort_keys=True,
                             ensure_ascii=False).encode("utf-8")
        if len(payload) > MAX_USERS_BYTES:
            raise UserStoreError("the user list exceeds its bounded size")
        directory = os.path.dirname(self._path) or "."
        descriptor, temp_path = tempfile.mkstemp(prefix=".users-", suffix=".tmp",
                                                dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self._path)
            _fsync_directory(directory)
        except OSError as exc:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise UserStoreError(
                f"the user list could not be written ({exc.strerror or 'storage error'})") from None
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
        self._users = users
        self._blocked = frozenset(blocked)


def _fsync_directory(directory: str) -> None:
    """Make the rename itself durable; best-effort (a directory that cannot be
    opened for fsync — some filesystems — still has the rename)."""
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NONBLOCK)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


__all__ = [
    "DEFAULT_ALLOW", "MAX_USERS", "PERMISSIONS",
    "USERS_FILENAME", "USERS_SCHEMA", "UserStore", "UserStoreError",
    "normalise_display_name", "normalise_email", "users_path",
]
