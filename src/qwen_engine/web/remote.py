"""Remote-port policy: proxy identity, per-user gates and the remote wire block.

The Railway container terminates TLS and Google sign-in, then forwards every
request over Tailscale with two pieces of proven identity in headers (the
shared proxy token, and the signed-in account). This module is the pure half of
that contract — no sockets, no filesystem reads at request time:

* :func:`load_proxy_token` reads the operator's 0600 secret ONCE at startup;
* :func:`tokens_match` compares it in constant time;
* :func:`parse_identity` turns the identity headers into one
  :class:`RemoteIdentity` (normalized, bounded, single-valued);
* :func:`check_feature_gates` enforces C3's per-user feature permissions
  server-side, whatever the page chose to hide;
* :func:`user_block` builds the additive status/history block the page reads.

Everything here is refused by :class:`~qwen_engine.web.api.ApiError`, so the
transport never has to translate a second error type.
"""
from __future__ import annotations

import hmac
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qwen_engine.owners import LOCAL_OWNER, MAX_OWNER_CHARS
from qwen_engine.users import PERMISSIONS, UserStoreError, normalise_email
from qwen_engine.web.api import ApiError

#: The Railway proxy's shared secret, and the two identity headers it sets.
#: All three are LOWERCASE because the handler's header map is lowercase.
TOKEN_HEADER = "x-lightseed-proxy-token"
EMAIL_HEADER = "x-lightseed-user-email"
SUB_HEADER = "x-lightseed-user-sub"

#: The secret must be worth comparing: a file shorter than this is a
#: misconfiguration, and refusing at startup beats running with a weak token.
MIN_TOKEN_BYTES = 16
MAX_TOKEN_BYTES = 512
#: Bounds for the identity headers (Google ``sub`` is 21 digits; the slack
#: keeps a future provider's opaque id legal without allowing an unbounded
#: value into a record or a log line).
#: The account id becomes the OWNER id of the person's data, so it is bounded
#: by the owner grammar too.
MAX_SUB_CHARS = MAX_OWNER_CHARS
MAX_SUB_BYTES = 1024
_SUB_RE = re.compile(r"\S{1,%d}\Z" % MAX_SUB_CHARS)

#: C3: a remote user keeps at most this many FINISHED jobs. The sixth success
#: replaces the oldest one, files included.
REMOTE_HISTORY_CAP = 5
#: C3: one live job (queued or running) per remote user; a second is 409.
REMOTE_QUEUE_LIMIT = 1

#: Routes that exist on the local port and MUST NOT exist on the remote one:
#: the Settings rewriter controls, the model downloader and the cross-user
#: "delete all history". They answer 404 there, exactly like any other unknown
#: target, so a remote client cannot even probe for them.
REMOTE_FORBIDDEN_KINDS = frozenset({
    "settings_get", "settings_set", "settings_test",
    "models", "models_download", "models_cancel",
    "forget_all",
})

#: Feature id -> the C3 rule it enforces, in the order the refusals are
#: checked. Every id is also a permission flag in the user store.
# Singular subjects, so every refusal reads "<label> needs the '<flag>'
# permission" without the caller assembling a sentence or a verb table.
FEATURE_LABELS = {
    "q8": "8-bit (Q8) generation",
    "2k": "generation above 1024x1024",
    "edit": "reference-image editing",
}


class RemoteConfigError(ValueError):
    """The remote listener's own configuration (token file or origin) is unusable."""


@dataclass(frozen=True)
class RemoteIdentity:
    """Who the proxy says is calling: the signed-in account, normalized."""

    email: str
    sub: str
    #: Filled from the user store by the caller (the store is not this
    #: module's dependency); empty when the person has not set a name.
    display_name: str = ""


# ---------------------------------------------------------------------------
# The shared secret
# ---------------------------------------------------------------------------

def load_proxy_token(path: str | os.PathLike | None) -> bytes:
    """Read the proxy secret from an operator-supplied 0600 regular file.

    Refused at startup (never at request time) when the file is missing, is
    not a regular file, is reachable through a symlink, is group/other
    readable, or its content is not a plausible secret. An empty or padded
    secret must not silently become a working (or a trivially guessable)
    credential.
    """
    if path is None:
        raise RemoteConfigError("--remote-token-file is required for the remote port")
    try:
        raw_path = os.fspath(path)
    except TypeError:
        raise RemoteConfigError("--remote-token-file must be a filesystem path") from None
    if not isinstance(raw_path, str) or not raw_path.strip() or "\0" in raw_path:
        raise RemoteConfigError("--remote-token-file must be a non-blank path")
    try:
        info = os.lstat(raw_path)
    except OSError as exc:
        raise RemoteConfigError(
            f"the remote proxy token file is unreadable ({exc.strerror or 'storage error'})") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RemoteConfigError("the remote proxy token file must be a regular file")
    if info.st_mode & 0o077:
        raise RemoteConfigError("the remote proxy token file must not be readable by group or others")
    if info.st_size > MAX_TOKEN_BYTES:
        raise RemoteConfigError("the remote proxy token file is larger than any legal token")
    try:
        with open(raw_path, "rb") as handle:
            secret = handle.read(MAX_TOKEN_BYTES + 1)
    except OSError as exc:
        raise RemoteConfigError(
            f"the remote proxy token file is unreadable ({exc.strerror or 'storage error'})") from None
    # A file written by an editor conventionally ends with one newline; that
    # terminator is not part of the secret. Nothing else is stripped.
    if secret.endswith(b"\n"):
        secret = secret[:-1]
    if secret.endswith(b"\r"):
        secret = secret[:-1]
    if len(secret) < MIN_TOKEN_BYTES:
        raise RemoteConfigError(
            f"the remote proxy token must be at least {MIN_TOKEN_BYTES} bytes")
    return secret


def tokens_match(provided: str | None, expected: bytes) -> bool:
    """Constant-time comparison of one request header against the secret.

    ``provided`` is the raw header text as received; it is compared as bytes
    in its UTF-8 form so a value carrying non-ASCII bytes is simply wrong
    rather than an encoding error, and the comparison time does not depend on
    how much of the secret matched.
    """
    if type(expected) is not bytes or not expected:
        return False
    if type(provided) is not str:
        return False
    try:
        candidate = provided.encode("utf-8", "surrogatepass")
    except (UnicodeEncodeError, AttributeError):
        return False
    return hmac.compare_digest(candidate, expected)


def check_proxy_token(headers: Mapping[str, list[str]], expected: bytes) -> None:
    """Refuse (401, no detail) unless exactly one correct token header is present.

    The refusal is deliberately bare: a caller that does not hold the secret
    learns nothing about why it was refused — not "missing" versus "wrong".
    """
    values = headers.get(TOKEN_HEADER, [])
    if len(values) != 1 or not tokens_match(values[0], expected):
        raise ApiError(401, "unauthorized", "this listener requires proxy authentication")


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _single(headers: Mapping[str, list[str]], name: str) -> str:
    values = headers.get(name, [])
    if len(values) != 1:
        raise ApiError(401, "unauthorized",
                       "the proxy identity headers are missing or duplicated")
    value = values[0]
    if len(value.encode("utf-8", "replace")) > MAX_SUB_BYTES:
        raise ApiError(401, "unauthorized", "a proxy identity header is too long")
    return value


def parse_identity(headers: Mapping[str, list[str]]) -> RemoteIdentity:
    """The proxy's proven identity, or 401.

    The email is normalized through the SAME function the user store uses, so
    the allowlist lookup cannot disagree with the stored key; the account id is
    bounded non-blank text. Both headers must appear exactly once: a duplicated
    identity header is an attack, not a client quirk.
    """
    raw_email = _single(headers, EMAIL_HEADER)
    raw_sub = _single(headers, SUB_HEADER)
    try:
        email = normalise_email(raw_email)
    except UserStoreError:
        raise ApiError(401, "unauthorized", "the proxy identity headers are malformed") from None
    # ``local`` is the owner id of the person at the machine: an account id
    # spelled that way would be handed the owner's whole history.
    if _SUB_RE.fullmatch(raw_sub) is None or raw_sub == LOCAL_OWNER:
        raise ApiError(401, "unauthorized", "the proxy identity headers are malformed")
    return RemoteIdentity(email=email, sub=raw_sub)


def verify_user(store: Any, identity: RemoteIdentity, *,
                auto_invite: bool = False) -> RemoteIdentity:
    """Allowlist check plus account binding; 403 when not invited.

    With ``auto_invite`` an address that is not on the list yet is added on
    the spot (Q8 only), unless the admin removed it before: then Google
    sign-in is the only gate, and removal still keeps a person out.

    The store is duck-typed (``is_invited``/``bind_sub``/``get``) so this
    module keeps no import-time dependency on the server's wiring. A bound
    account id that disagrees with the sign-in is treated as not invited: a
    recycled address must never reach the previous person's data.
    """
    if not store.is_invited(identity.email) and not (
            auto_invite and _auto_add(store, identity.email)):
        raise ApiError(403, "not_invited",
                       "this account is not invited to use this server")
    try:
        bound = store.bind_sub(identity.email, identity.sub)
    except UserStoreError:
        raise ApiError(403, "not_invited",
                       "this account is not invited to use this server") from None
    if not bound:
        raise ApiError(403, "not_invited",
                       "this account is not invited to use this server")
    invited = store.get(identity.email)
    name = "" if invited is None else str(invited.get("display_name") or "")
    return RemoteIdentity(email=identity.email, sub=identity.sub, display_name=name)


def _auto_add(store: Any, email: str) -> bool:
    try:
        return bool(store.auto_add(email))
    except UserStoreError:  # full list, unwritable file: stay closed
        return False


def permissions(store: Any, identity: RemoteIdentity) -> dict[str, bool]:
    """The four C3 flags of one invited person (all denied when unknown)."""
    return {flag: bool(value) for flag, value in store.permissions(identity.email).items()
            if flag in PERMISSIONS}


# ---------------------------------------------------------------------------
# Feature gates (C3)
# ---------------------------------------------------------------------------

#: 1024x1024*1.3, compared as an exact integer ratio (``pixels*10 > 1024*1024*13``)
#: so no float rounding can move the boundary by a pixel.
TWO_K_SCALED_PIXELS = 1024 * 1024 * 13
TWO_K_SCALE = 10


def check_reference_ownership(root: str | None, reference_ids,
                              owner: str) -> None:
    """Refuse (404) a rewrite that names somebody else's reference ids.

    The rewrite route loads stored images itself, so the ownership rule the
    reference and job routes enforce has to be applied here too — otherwise a
    remote user could have another account's picture described (and thus
    revealed) through the rewriter. An id that is unknown and an id that
    belongs to another account are the SAME refusal, by ordinal.
    """
    from qwen_engine import reference_store

    if root is None:
        raise ApiError(422, "reference_unavailable",
                       "the server reference store is not configured")
    for ordinal, reference_id in enumerate(reference_ids, start=1):
        try:
            if owner not in reference_store.owners_for(root, reference_id):
                raise ApiError(404, "reference_unknown",
                               f"reference {ordinal} is not available in the "
                               "server reference store")
        except reference_store.ReferenceStoreError:
            raise ApiError(404, "reference_unknown",
                           f"reference {ordinal} is not available in the "
                           "server reference store") from None


def rewrite_backend_for(allow: Mapping[str, bool]) -> str:
    """C3: which rewriter answers a remote user.

    Everyone without ``official_enhancer`` is served by the OpenAI-compatible
    endpoint — ALWAYS, whatever the owner's Settings say — so a remote draft
    never takes the V100 from the person at the machine. A person who HAS the
    permission gets the saved choice, exactly like the local owner.
    """
    return None if allow.get("official_enhancer") else "local-llm"


def check_rewrite_permission(task: str, allow: Mapping[str, bool]) -> None:
    """C3: an edit rewrite is editing, so it needs the ``edit`` permission."""
    if task == "edit" and not allow.get("edit"):
        raise ApiError(403, "feature_not_allowed", denial("edit"))


def denial(feature: str) -> str:
    """One refusal sentence naming the permission to ask the host for."""
    return (f"{FEATURE_LABELS[feature]} needs the '{feature}' permission; "
            "ask the host")


def check_feature_gates(config: Any, allow: Mapping[str, bool]) -> None:
    """Enforce the C3 table against a validated ``GenerationConfig``.

    Refusals are typed and name the feature (the page shows the permission the
    person has to ask the host for). Turbo imposes no rule of its own; the
    adapter's existing recipe gate still applies afterwards. Sizes are the
    CONFIG's own width/height (already snapped to the 32-pixel grid), never a
    browser-supplied pixel count.
    """
    width = getattr(config, "width", 0)
    height = getattr(config, "height", 0)
    if (type(width) is int and type(height) is int
            and width > 0 and height > 0
            and width * height * TWO_K_SCALE > TWO_K_SCALED_PIXELS
            and not allow.get("2k")):
        raise ApiError(403, "feature_not_allowed", denial("2k"))
    references = getattr(config, "references", ())
    if references and not allow.get("edit"):
        raise ApiError(403, "feature_not_allowed", denial("edit"))
    quant = getattr(config, "quant", None)
    quant_name = getattr(quant, "value", quant)
    if quant_name == "q8" and not allow.get("q8"):
        raise ApiError(403, "feature_not_allowed", denial("q8"))


# ---------------------------------------------------------------------------
# Wire blocks
# ---------------------------------------------------------------------------

def history_block(count: int, cap: int = REMOTE_HISTORY_CAP) -> dict[str, Any]:
    """One person's history occupancy: finished jobs, the cap, and the warning.

    Used both by the status ``user`` block and by the submit/job payloads so
    the page can warn ("the next success replaces your oldest") before the
    person starts a job rather than after it replaced something. A negative
    count reads as none and a cap below one as one, so a bad caller value
    cannot make the page believe more room exists than the reaper grants.
    """
    kept = max(0, int(count))
    limit = max(1, int(cap))
    return {"count": kept, "cap": limit, "will_replace_oldest": kept >= limit}


def user_block(identity: RemoteIdentity, allow: Mapping[str, bool], *,
               history_count: int, history_cap: int = REMOTE_HISTORY_CAP) -> dict[str, Any]:
    """The remote status ``user`` block (D3's parser depends on this shape).

    ``allow`` is always all four flags — a missing key would make the page
    guess — and ``history`` is the count of the person's finished jobs with the
    cap and whether the next success would replace the oldest one.
    """
    return {
        "email": identity.email,
        "display_name": identity.display_name,
        "allow": {flag: bool(allow.get(flag)) for flag in PERMISSIONS},
        "history": history_block(history_count, history_cap),
    }


__all__ = [
    "EMAIL_HEADER", "FEATURE_LABELS", "MAX_SUB_CHARS", "MIN_TOKEN_BYTES",
    "REMOTE_FORBIDDEN_KINDS", "REMOTE_HISTORY_CAP", "REMOTE_QUEUE_LIMIT",
    "RemoteConfigError", "RemoteIdentity", "SUB_HEADER", "TOKEN_HEADER",
    "TWO_K_SCALE", "TWO_K_SCALED_PIXELS", "check_feature_gates", "check_proxy_token",
    "history_block", "load_proxy_token", "parse_identity", "permissions",
    "check_reference_ownership", "check_rewrite_permission", "rewrite_backend_for",
    "tokens_match", "user_block", "verify_user",
]
