"""The local-only Admin page's wire contract and its five routes (C4).

One page, one server-side document. Every write answers with the WHOLE
refreshed admin document, so the page re-renders what the store now holds
instead of patching its own guess: a permission shown on screen is a
permission the file holds.

Reach: these routes exist on the LOCAL listener only. ``server.py`` refuses
them with the ordinary ``404 not_found`` when the request arrived on a remote
listener, exactly like a route that was never published — a remote user must
not be able to tell an admin route exists.

No path, no address and no header value is ever accepted or echoed: an admin
request names an account by EMAIL, a boolean or two, and nothing else. Every
document is a closed set of keys; an unknown or missing key is a typed
refusal rather than a silently dropped field.
"""
from __future__ import annotations

from typing import Any, Mapping

from qwen_engine import users as _users
from qwen_engine.users import UserStore, UserStoreError
from qwen_engine.web.api import (
    ApiError, ApiResponse, parse_request_json,
)

#: The admin document, and one request schema per write. Closed tags, never
#: bumped for an additive change the page also parses.
ADMIN_SCHEMA = "qwen_engine.web_admin.v1"
ADMIN_ADD_SCHEMA = "qwen_engine.web_admin_add.v1"
ADMIN_REMOVE_SCHEMA = "qwen_engine.web_admin_remove.v1"
ADMIN_PERMISSIONS_SCHEMA = "qwen_engine.web_admin_permissions.v1"
ADMIN_DISPLAY_NAME_SCHEMA = "qwen_engine.web_admin_display_name.v1"

#: The remote-user history cap (contract C3). The admin page prints it beside
#: each account's usage; enforcement belongs to the job store, not to this
#: module — one number, quoted, never re-derived.
HISTORY_CAP = 5

#: The four permission flags, in the order the table shows them. Mirrors
#: :data:`qwen_engine.users.PERMISSIONS`; the store validates the same set.
PERMISSION_KEYS = ("q8", "2k", "edit", "official_enhancer")

#: Closed key sets, so a typo is a refusal.
_ADMIN_DOCUMENT_KEYS = ("schema", "cap", "users")
_ADMIN_USER_KEYS = ("email", "display_name", "invited", "first_sign_in",
                    "signed_in", "history", "active", "allow")

_EMAIL_KEY = "email"
_DELETE_DATA_KEY = "delete_data"
_ALLOW_KEY = "allow"
_DISPLAY_NAME_KEY = "display_name"


# ---------------------------------------------------------------------------
# envelopes (pure)
# ---------------------------------------------------------------------------


def _envelope(raw: bytes, schema: str, what: str, *,
              required: tuple[str, ...],
              optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """One closed request object: the schema tag, every required key, and
    nothing outside the allowed set."""
    data = parse_request_json(raw)
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       f"the {what} must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    allowed = {"schema", *required, *optional}
    missing = sorted({"schema", *required} - set(data))
    extra = sorted(set(data) - allowed)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected key(s): " + ", ".join(extra))
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != schema:
        raise ApiError(400, "invalid_envelope", f"schema must be {schema!r}")
    return {key: value for key, value in data.items() if key != "schema"}


def build_add_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/admin/users/add``: an email and an optional display
    name. Permissions are deliberately NOT accepted — a new account starts
    with q8 only, and the tick boxes change it afterwards."""
    fields = _envelope(raw, ADMIN_ADD_SCHEMA, "admin add-user request",
                       required=(_EMAIL_KEY,),
                       optional=(_DISPLAY_NAME_KEY,))
    email = _checked_email(fields[_EMAIL_KEY])
    name = ""
    if _DISPLAY_NAME_KEY in fields:
        name = _checked_display_name(fields[_DISPLAY_NAME_KEY])
    return {"email": email, "display_name": name}


def build_remove_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/admin/users/remove``: which account, and whether its
    images and references go with it."""
    fields = _envelope(raw, ADMIN_REMOVE_SCHEMA, "admin remove-user request",
                       required=(_EMAIL_KEY, _DELETE_DATA_KEY))
    if type(fields[_DELETE_DATA_KEY]) is not bool:
        raise ApiError(400, "invalid_envelope",
                       "delete_data must be true or false",
                       (("delete_data", "must be a boolean"),))
    return {"email": _checked_email(fields[_EMAIL_KEY]),
            "delete_data": fields[_DELETE_DATA_KEY]}


def build_permissions_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/admin/users/permissions``: named flags for one account.

    The ``allow`` block must itself be closed over the four flags: a partial
    block would leave the page and the store disagreeing about the rest.
    """
    fields = _envelope(raw, ADMIN_PERMISSIONS_SCHEMA,
                       "admin permission request",
                       required=(_EMAIL_KEY, _ALLOW_KEY))
    allow = fields[_ALLOW_KEY]
    if not isinstance(allow, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "allow must be a JSON object",
                       (("allow", "must be a JSON object"),))
    if any(not isinstance(key, str) for key in allow):
        raise ApiError(400, "invalid_envelope", "permission names must be strings")
    if set(allow) != set(PERMISSION_KEYS):
        raise ApiError(400, "invalid_envelope",
                       "allow must carry exactly " + ", ".join(PERMISSION_KEYS),
                       (("allow", "carries exactly the four flags"),))
    flags = {}
    for name in PERMISSION_KEYS:
        if type(allow[name]) is not bool:
            raise ApiError(400, "invalid_envelope",
                           f"permission {name} must be true or false",
                           (("allow", f"{name} must be a boolean"),))
        flags[name] = allow[name]
    return {"email": _checked_email(fields[_EMAIL_KEY]), "allow": flags}


def build_display_name_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/admin/users/display-name``: one line of display text."""
    fields = _envelope(raw, ADMIN_DISPLAY_NAME_SCHEMA,
                       "admin display-name request",
                       required=(_EMAIL_KEY, _DISPLAY_NAME_KEY))
    return {"email": _checked_email(fields[_EMAIL_KEY]),
            "display_name": _checked_display_name(fields[_DISPLAY_NAME_KEY])}


def _checked_email(value: Any) -> str:
    try:
        return _users.normalise_email(value)
    except UserStoreError as exc:
        raise ApiError(422, "invalid_user", str(exc),
                       (("email", str(exc)),)) from None


def _checked_display_name(value: Any) -> str:
    try:
        return _users.normalise_display_name(value)
    except UserStoreError as exc:
        raise ApiError(422, "invalid_user", str(exc),
                       ((_DISPLAY_NAME_KEY, str(exc)),)) from None


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def validate_admin_document(document: Any) -> dict[str, Any]:
    """The whole ``GET /api/v1/admin/users`` document, checked closed.

    The server validates its own document before sending it, so the page can
    trust a document or reject it whole — never render half a table.
    """
    if not isinstance(document, Mapping) or set(document) != set(_ADMIN_DOCUMENT_KEYS):
        raise ApiError(500, "admin_unavailable",
                       "the admin user list is malformed")
    if document["schema"] != ADMIN_SCHEMA:
        raise ApiError(500, "admin_unavailable",
                       "the admin user list is malformed")
    if type(document["cap"]) is not int or document["cap"] < 1:
        raise ApiError(500, "admin_unavailable",
                       "the admin user list is malformed")
    rows = document["users"]
    if not isinstance(rows, list) or len(rows) > _users.MAX_USERS:
        raise ApiError(500, "admin_unavailable",
                       "the admin user list is malformed")
    seen: set[str] = set()
    out = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(_ADMIN_USER_KEYS):
            raise ApiError(500, "admin_unavailable",
                           "the admin user list is malformed")
        if not isinstance(row["email"], str) or row["email"] in seen:
            raise ApiError(500, "admin_unavailable",
                           "the admin user list is malformed")
        seen.add(row["email"])
        for key in ("display_name", "invited"):
            if not isinstance(row[key], str):
                raise ApiError(500, "admin_unavailable",
                               "the admin user list is malformed")
        if row["first_sign_in"] is not None and not isinstance(row["first_sign_in"], str):
            raise ApiError(500, "admin_unavailable",
                           "the admin user list is malformed")
        for key in ("signed_in",):
            if type(row[key]) is not bool:
                raise ApiError(500, "admin_unavailable",
                               "the admin user list is malformed")
        for key in ("history", "active"):
            if type(row[key]) is not int or row[key] < 0:
                raise ApiError(500, "admin_unavailable",
                               "the admin user list is malformed")
        allow = row["allow"]
        if not isinstance(allow, Mapping) or set(allow) != set(PERMISSION_KEYS) \
                or any(type(allow[name]) is not bool for name in PERMISSION_KEYS):
            raise ApiError(500, "admin_unavailable",
                           "the admin user list is malformed")
        out.append({"email": row["email"],
                    "display_name": row["display_name"],
                    "invited": row["invited"],
                    "first_sign_in": row["first_sign_in"],
                    "signed_in": row["signed_in"],
                    "history": row["history"],
                    "active": row["active"],
                    "allow": {name: allow[name] for name in PERMISSION_KEYS}})
    return {"schema": document["schema"], "cap": document["cap"],
            "users": out}


def _usage_by_owner(generation: Any) -> dict[str, tuple[int, int]]:
    """``owner -> (finished jobs in history, queued or running jobs)``.

    The job store is authority: records are read once per request and counted,
    never cached here (a cached count would drift from the History tab).
    ``JobSnapshot`` gains its ``owner`` from the server-core work; until then
    every record reads as the local owner, which is exactly true today.
    """
    counts: dict[str, tuple[int, int]] = {}
    controller = getattr(generation, "controller", None)
    snapshots = getattr(controller, "snapshots", None)
    if not callable(snapshots):
        return counts
    for record in snapshots():
        owner = getattr(record, "owner", "local")
        if not isinstance(owner, str):
            continue
        terminal = getattr(record, "terminal", None)
        if not isinstance(terminal, bool):
            state = getattr(getattr(record, "state", None), "value", None)
            terminal = state in ("cancelled", "succeeded", "failed")
        history, active = counts.get(owner, (0, 0))
        counts[owner] = ((history + 1, active) if terminal
                         else (history, active + 1))
    return counts


def admin_document(store: UserStore, generation: Any = None) -> dict[str, Any]:
    """The whole admin document: every invited account and what it has used."""
    usage = _usage_by_owner(generation)
    rows = []
    for entry in store.list_users():
        sub = entry["sub"]
        owners = [owner for owner in (sub, entry["email"]) if owner]
        history = active = 0
        for owner in owners:
            used = usage.get(owner)
            if used:
                history, active = used
                break
        rows.append({
            "email": entry["email"],
            "display_name": entry["display_name"],
            "invited": entry["added"],
            # C2 records when the account was invited, not when it first signed
            # in, so no sign-in time is invented here: the field carries the
            # store's own ``first_sign_in`` when a later store revision records
            # one, and null otherwise. ``signed_in`` is the honest answer to
            # "has this account ever used the page", because the sub is
            # recorded exactly on the first sign-in.
            "first_sign_in": entry.get("first_sign_in")
            if isinstance(entry.get("first_sign_in"), str) else None,
            "signed_in": bool(sub),
            "history": history,
            "active": active,
            "allow": {name: bool(entry["allow"][name]) for name in PERMISSION_KEYS},
        })
    return validate_admin_document({"schema": ADMIN_SCHEMA, "cap": HISTORY_CAP,
                                    "users": rows})


# ---------------------------------------------------------------------------
# the service (one per listener that serves the page)
# ---------------------------------------------------------------------------


class AdminService:
    """The local listener's admin state: the user store plus the job store.

    ``generation`` is the same object the job routes use, or ``None`` on a
    planning-only server (every account then reads as zero jobs, which is the
    truth there).
    """

    def __init__(self, store: UserStore, generation: Any = None) -> None:
        if not callable(getattr(store, "list_users", None)):
            raise ValueError("an admin service needs a user store")
        self.store = store
        self.generation = generation

    # -- reads -----------------------------------------------------------

    def document(self) -> dict[str, Any]:
        try:
            return admin_document(self.store, self.generation)
        except UserStoreError as exc:
            raise ApiError(503, "admin_unavailable",
                           f"the user list could not be read: {exc}") from None

    def list_response(self) -> ApiResponse:
        return ApiResponse(200, self.document())

    # -- writes ----------------------------------------------------------

    def post_response(self, kind: str, body: bytes) -> ApiResponse:
        """Route one admin write by the resolved route kind."""
        route = _POST_ROUTES.get(kind)
        if route is None:  # pragma: no cover - resolve_route owns the set
            raise ApiError(404, "not_found", "no such route")
        return route(self, body)

    def add_response(self, body: bytes) -> ApiResponse:
        fields = build_add_request(body)
        try:
            self.store.add(fields["email"])
            if fields["display_name"]:
                self.store.set_display_name(fields["email"],
                                            fields["display_name"])
        except UserStoreError as exc:
            raise _write_error(exc) from None
        return ApiResponse(200, self.document())

    def remove_response(self, body: bytes) -> ApiResponse:
        """Uninvite one account, optionally deleting everything it owns.

        The data goes FIRST: a store entry removed before the jobs would leave
        the page with no way to name the owner whose files remain. The job
        store is asked for that owner's FINISHED jobs only — a running job is
        never killed from the admin page — and the reference store is asked
        through its own hook when the server core provides one.
        """
        fields = build_remove_request(body)
        email = fields["email"]
        try:
            entry = self.store.get(email)
        except UserStoreError as exc:
            raise _write_error(exc) from None
        if entry is None:
            raise ApiError(404, "user_unknown", "that account is not invited")
        if fields["delete_data"]:
            owners = tuple(owner for owner in (entry["sub"], email) if owner)
            self._forget_owner_jobs(owners)
            self._forget_owner_references(owners)
        try:
            self.store.remove(email, delete_data=fields["delete_data"])
        except UserStoreError as exc:
            raise _write_error(exc) from None
        return self.list_response()

    def permissions_response(self, body: bytes) -> ApiResponse:
        fields = build_permissions_request(body)
        try:
            self.store.set_permissions(fields["email"], **fields["allow"])
        except UserStoreError as exc:
            raise _write_error(exc) from None
        return ApiResponse(200, self.document())

    def display_name_response(self, body: bytes) -> ApiResponse:
        fields = build_display_name_request(body)
        try:
            self.store.set_display_name(fields["email"], fields["display_name"])
        except UserStoreError as exc:
            raise _write_error(exc) from None
        return ApiResponse(200, self.document())

    # -- data removal ----------------------------------------------------

    def _forget_owner_jobs(self, owners: tuple[str, ...]) -> tuple[int, int]:
        """Forget every FINISHED job owned by one of ``owners``, with files.

        Returns ``(forgotten, deleted_files)``. A live job is skipped: the
        admin page removes accounts, it does not kill work in progress.
        """
        controller = getattr(self.generation, "controller", None)
        snapshots = getattr(controller, "snapshots", None)
        forget = getattr(controller, "forget", None)
        if not callable(snapshots) or not callable(forget):
            return 0, 0
        forgotten = deleted = 0
        for record in snapshots():
            owner = getattr(record, "owner", "local")
            if owner not in owners:
                continue
            terminal = getattr(record, "terminal", None)
            if not isinstance(terminal, bool):
                state = getattr(getattr(record, "state", None), "value", None)
                terminal = state in ("cancelled", "succeeded", "failed")
            if not terminal:
                continue
            try:
                deleted += forget(record.job_id, delete_files=True)
            except (ValueError, OSError):
                # A job that vanished between the snapshot and the call is
                # already gone: nothing left to delete, nothing to report.
                continue
            forgotten += 1
        return forgotten, deleted

    def _forget_owner_references(self, owners: tuple[str, ...]) -> None:
        """Ask the reference store to drop one owner's uploads, if it can.

        The store is owner-aware only once the server-core work lands, so this
        is a capability hook rather than an assumption: a store without
        ``forget_owner`` simply keeps its files.
        """
        from qwen_engine import reference_store

        hook = getattr(reference_store, "forget_owner", None)
        if not callable(hook):
            return
        root = getattr(getattr(self.generation, "runtime", None),
                       "references_dir", None)
        if not root:
            return
        for owner in owners:
            try:
                hook(root, owner)
            except (OSError, ValueError):
                continue


def _write_error(exc: UserStoreError) -> ApiError:
    """One typed answer for every store refusal, never an internal dump."""
    message = str(exc)
    if "not invited" in message or "already bound" in message:
        return ApiError(404, "user_unknown", message)
    if "written" in message or "unreadable" in message:
        return ApiError(503, "admin_unavailable", message)
    return ApiError(422, "invalid_user", message)


#: The route kind (as ``server.resolve_route`` names it) to its handler. One
#: table, so a new admin action is one line here and cannot be half-wired.
_POST_ROUTES = {
    "admin_add": AdminService.add_response,
    "admin_remove": AdminService.remove_response,
    "admin_permissions": AdminService.permissions_response,
    "admin_display_name": AdminService.display_name_response,
}
