"""One grammar for data ownership, shared by the job queue and the reference store.

An owner is an identity key, never free text: :data:`LOCAL_OWNER` for the person
at the machine, the provider account id (``sub``) for a remote user. It ends up
in per-user filters, in stored records and in refusal messages, so the rules are
one place and deliberately narrow:

* exact ``str`` only — a ``str`` subclass could change what equality means long
  after a record was written;
* non-empty and bounded (:data:`MAX_OWNER_CHARS`);
* printable, without whitespace — a value carrying a control character or
  padding is refused rather than trimmed.

Nothing here raises a domain-specific error: each caller maps :class:`OwnerError`
onto its own typed refusal (``JobError``, ``ReferenceStoreError``), so the
message a caller sees still belongs to its own contract.
"""
from __future__ import annotations

from typing import Any

#: The owner id of everything done by the person at the machine.
LOCAL_OWNER = "local"
MAX_OWNER_CHARS = 200


class OwnerError(ValueError):
    """One owner id is not a bounded printable identity key."""


def is_owner(value: Any) -> bool:
    """True exactly for a well-formed owner id (see the module docstring)."""
    if type(value) is not str or not value or len(value) > MAX_OWNER_CHARS:
        return False
    return all(ch.isprintable() and not ch.isspace() for ch in value)


def check_owner(value: Any) -> str:
    """Return ``value`` when it is a valid owner id, else raise :class:`OwnerError`."""
    if not is_owner(value):
        raise OwnerError("owner must be a bounded non-empty printable string without whitespace")
    return value


__all__ = ["LOCAL_OWNER", "MAX_OWNER_CHARS", "OwnerError", "check_owner", "is_owner"]
