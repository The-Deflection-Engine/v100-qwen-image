"""Pure W2 web contract: status / plan / editable-config-export (no sockets).

Every function here is deterministic and ML-free. The STATUS response is
served from metadata CACHED at startup — it performs no filesystem I/O at
request time. PLAN and EXPORT run the SHARED config validator, which may
STAT configured runtime-path metadata (directories only, never weight
bytes) while validating the request against the manifest. No path here
ever READS model weights or computes hashes. The only filesystem knowledge
enters through :class:`WebRuntime`, which the server validates and
canonicalises once at startup. Browser JSON can never select a runtime path — unknown
path fields are refused, not stripped.

Wire rules that are load-bearing (reviewed W2 contract):

* Public seeds are **decimal strings or null** in plan/status — never numeric
  JSON (JS ``Number`` would round past 2^53). The config export is the one
  deliberate exception: raw bytes with the exact Python uint64 number,
  downloaded as a blob and never re-serialized by the browser.
* Request seed wire: decimal ``[0-9]{1,20}`` string or null; floats, bools,
  numbers and over-long digit strings are refused typed — never parsed
  through a float and never silently rounded.
* Responses carry only schema-versioned, bounded, safe content: no runtime
  paths, no manifest filenames, no hashes, no tracebacks, no raw internal
  exception text.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from qwen_engine import _jsonsafe
from qwen_engine import model_setup as _setup
from qwen_engine import reference_store as _reference_store
from qwen_engine import rewrite as _rewrite
from qwen_engine import rewrite_client as _rewrite_client
from qwen_engine.config import REF_MAX, SEED_MAX, ConfigError, GenerationConfig
from qwen_engine.core import plan as core_plan
from qwen_engine.owners import LOCAL_OWNER

#: Wire schema tags (versioned like every other qwen_engine document).
WEB_STATUS_SCHEMA = "qwen_engine.web_status.v1"
WEB_PLAN_SCHEMA = "qwen_engine.web_plan.v1"
WEB_PLAN_REQUEST_SCHEMA = "qwen_engine.web_plan_request.v1"
WEB_ERROR_SCHEMA = "qwen_engine.web_error.v1"
WEB_REFERENCE_SCHEMA = "qwen_engine.web_reference.v1"
#: Prompt-rewriting wire tags (plan §4.1). Defined HERE, next to the other
#: wire tags, because this module is the transport's contract boundary; the
#: service imports them from here rather than keeping a second copy.
REWRITE_REQUEST_SCHEMA = "qwen_engine.web_rewrite_request.v1"
WEB_REWRITE_SCHEMA = "qwen_engine.web_rewrite.v1"

#: Route body cap for /api/v1/rewrite. DERIVED, never chosen independently:
#: the longest legal draft (``PROMPT_MAX_CHARS``) at 4 bytes/char UTF-8 plus
#: a ≤1 KiB envelope. The browser applies the same formula locally, so this
#: is a backstop; ``MAX_BODY_BYTES`` (64 KiB) is larger than this and is NOT
#: the binding cap for this route.
MAX_REWRITE_REQUEST_BYTES = _rewrite.PROMPT_MAX_CHARS * 4 + 1024

#: Route cap for raw reference uploads — the store's own byte cap is the
#: single source of truth; JSON routes keep the small 64 KiB MAX_BODY_BYTES.
MAX_REFERENCE_UPLOAD_BYTES = _reference_store.MAX_REFERENCE_BYTES

#: Sniffed-store-format -> response Content-Type (design §2.4; identical
#: magic table, one source of truth: the store).
_REFERENCE_CONTENT_TYPES = {"png": "image/png", "jpeg": "image/jpeg",
                            "webp": "image/webp"}

#: The browser-editable GenerationConfig fields — exactly these, no more.
#: Runtime paths (output/cache/log/benchmarks/models/manifest) are
#: server-controlled and must never arrive from browser JSON.
EDITABLE_FIELDS = frozenset({
    "prompt", "negative_prompt", "width", "height", "steps", "recipe",
    "true_cfg_scale", "seed", "quant", "offload", "attention_backend",
    "vae_tiled_decode", "output_format", "transparent_background",
    "references",
})

#: Server-controlled runtime path fields (never editable from the browser).
RUNTIME_PATH_FIELDS = ("output_dir", "cache_dir", "log_dir",
                       "benchmarks_dir", "models_dir", "manifest_path",
                       "q8_manifest_path", "references_dir")

#: GenerationConfig fields that are deliberately NOT on the browser wire YET.
#: Each entry is an announced deferral with its owning milestone, never a
#: silent carve-out: the partition guard below FAILS for any field that is
#: neither editable, a runtime path, nor listed here — so a new config field
#: still cannot slip past this contract unnoticed.
#:
#: EMPTY today: ``recipe`` was the last deferral and became editable with the
#: Turbo web surface (docs/turbo-plan.md §6). The set and its guard stay in
#: place as the place a future field must announce itself in.
DEFERRED_WIRE_FIELDS = frozenset()

#: Seed wire grammar: ASCII decimal digits only, at most 20 characters
#: (the full uint64 range stays an exact string; range is validated later
#: against ``config.SEED_MAX``). Leading zeros are allowed; whitespace,
#: signs, exponents and non-ASCII digits are not.
_SEED_WIRE_RE = re.compile(r"[0-9]{1,20}\Z")

#: Bound on any single public error message (safe, single-line, truncated).
MAX_MESSAGE_CHARS = 200

#: The rewrite status block is exactly these four keys (plan §4.1), plus the
#: additive keys below: the UI parser is strict, so the server never widens
#: its own contract by surprise — and a block that carries a key the validator
#: does not know is still a wire change and is refused.
_REWRITE_BLOCK_KEYS = frozenset({"state", "code", "backend", "reason"})

#: OPTIONAL additions. A server that predates the two-backend rewriter sends
#: none of them, and the new UI must parse such a block unchanged:
#: ``backends`` — which rewriters exist here and what each can do;
#: ``settings`` — the route the Settings page writes its choice to, and whether
#: a saved document is on disk. A ROUTE, never a filesystem path;
#: ``active_by_task`` — which rewriter answers EACH draft kind, so a page can
#: say "text drafts use the official enhancer, edit drafts the endpoint". The
#: single ``backend`` stays for readers that predate per-task choices.
_REWRITE_BLOCK_KEYS_OPTIONAL = frozenset({"backends", "settings",
                                          "active_by_task"})

#: The CLOSED state/code pairs, mirroring ``REWRITE_STATES`` in app.js: the
#: code is the key, the state is the value it must pair with. Rewriting is
#: optional and never a gate, so these are deliberately NOT in ``_gates()``.
_REWRITE_PAIRS: Mapping[str, str] = {
    "rewrite_available": "available",
    "rewrite_disabled": "unavailable",
}

#: Shapes that must never reach the browser: the status document is served to
#: a page, and a reason is operator-facing prose, not a place to leak where the
#: backend lives. Refusing is cheap; a leaked loopback address is not.
_LOCATION_MARKERS = ("://", "/", "\\")
_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
#: A command-line option (``-m``, ``--no-webui``, ``--port=8080``). Only the
#: enhancer's spawn error can carry one, and nothing that reaches the browser
#: may quote the operator's command line.
_COMMAND_FLAG_RE = re.compile(r"(?:\A|(?<=\s))--?[A-Za-z][\w-]*")


def strip_command(text: str) -> str:
    """One line of prose with addresses, paths and options removed.

    Shared by the two places that turn a backend or spawn failure into a
    sentence for the page (:func:`rewrite_test_result` here, ``_public_reason``
    in the rewrite service), so neither can drift from the other.
    """
    text = _IPV4_RE.sub("the host", " ".join(str(text or "").split()))
    for marker in _LOCATION_MARKERS:
        text = text.replace(marker, " ")
    text = _COMMAND_FLAG_RE.sub(" ", text)
    return " ".join(text.split())


#: A backend id as it travels on the wire: a label, never an address or path.
_BACKEND_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")

#: The two rewriter backends, in preference order. The official enhancer is
#: the standard rewriter; the local LLM service is the alternative that never
#: touches the GPU. ``off`` is NOT a backend — it is reachable from Settings
#: only, and never builds a client.
REWRITE_BACKENDS: tuple[str, ...] = ("pe-llamacpp", "local-llm")

#: Where the rewriter choice lives, as the browser sees it: a ROUTE, never a
#: filesystem path. ONE definition, because the status block, the Settings
#: payload and the router must name the same door.
REWRITE_SETTINGS_HREF = "/api/v1/settings/rewrite"
REWRITE_SETTINGS_TEST_HREF = REWRITE_SETTINGS_HREF + "/test"

#: Everything the Settings page may choose, including switching off.
REWRITE_SETTINGS_BACKENDS: tuple[str, ...] = REWRITE_BACKENDS + ("off",)

#: The friendly names. ONE definition, because the status block, the Settings
#: payload and the suggestion box must not disagree about what a rewriter is
#: called. A label is display text: never a path, an address or a model id.
REWRITE_BACKEND_LABELS: Mapping[str, str] = {
    "pe-llamacpp": "Official enhancer (Q8, uses GPU)",
    "local-llm": "OpenAI-compatible endpoint",
    "off": "Off",
}

#: Which rewriter takes the V100. The Settings page says so plainly next to the
#: radio, because choosing it means an image job may wait.
REWRITE_BACKEND_USES_GPU: Mapping[str, bool] = {"pe-llamacpp": True,
                                                "local-llm": False}

#: The two draft kinds the Settings page offers a rewriter for, in the order
#: the page lists them, with the words the user sees. ONE definition, because
#: the radio groups, the server's own fallback messages and the suggestion box
#: must call "Edit prompt enhance" the same thing.
REWRITE_TASK_LABELS: Mapping[str, str] = {
    "t2i": "Prompt enhance (text-to-image)",
    "edit": "Edit prompt enhance",
}

#: The closed shape of one ``backends`` row. ``tasks`` is what that backend can
#: rewrite (the enhancer has no edit weights), ``uses_gpu`` says plainly that
#: choosing it takes the V100 — and so waits for, or blocks, image jobs. In the
#: Settings form a row also carries whether it is usable here and why not.
_BACKEND_ROW_KEYS = frozenset({"id", "label", "tasks", "uses_gpu"})
_BACKEND_ROW_KEYS_SETTINGS = _BACKEND_ROW_KEYS | {"available", "reason"}


def _check_availability(available: Any, reason: Any, what: str) -> None:
    """The `available`/`reason` pair every greyed-out option carries, pure.

    ONE rule for the Settings page's backend rows and its per-task choice rows:
    an option that works explains nothing, an option that does not must say why,
    and that sentence is rendered in a browser — so it may never contain a path,
    a URL or an address. Refusing here is what keeps the rule checkable instead
    of a convention nobody follows.
    """
    if type(available) is not bool or not isinstance(reason, str):
        raise ValueError(f"a {what} needs a boolean available and a string reason")
    if available:
        if reason != "":
            raise ValueError(f"an available {what} carries no reason")
        return
    if not reason.strip():
        raise ValueError(f"an unavailable {what} needs a reason")
    for marker in _LOCATION_MARKERS:
        if marker in reason:
            raise ValueError("a rewrite availability reason must not disclose a "
                             "location (URL or path)")
    if _IPV4_RE.search(reason):
        raise ValueError("a rewrite availability reason must not disclose an address")


def validate_rewrite_backend_row(row: Any, *, settings_form: bool = False) -> dict[str, Any]:
    """One ``backends`` row, pure. Raises :class:`ValueError`, bounded.

    The label must be EXACTLY the one this module publishes: a server that
    invented a friendlier name would be changing what the browser displays
    without changing the browser.
    """
    want = _BACKEND_ROW_KEYS_SETTINGS if settings_form else _BACKEND_ROW_KEYS
    if not isinstance(row, Mapping) or set(row) != want:
        raise ValueError(
            "a rewrite backend row must carry exactly "
            f"{sorted(want)}; got {sorted(map(str, row)) if isinstance(row, Mapping) else type(row).__name__}")
    backend = row["id"]
    if backend not in REWRITE_BACKENDS:
        raise ValueError(f"unknown rewrite backend id {backend!r}")
    if row["label"] != REWRITE_BACKEND_LABELS[backend]:
        raise ValueError(
            f"the label for {backend!r} must be {REWRITE_BACKEND_LABELS[backend]!r}")
    tasks = row["tasks"]
    # A status block is validated TWICE: once by the service that built it (a
    # plain document) and once by ``WebRuntime`` after ``freeze_json`` turned
    # every list into a tuple. Both shapes are legal input; what comes back is
    # always the plain JSON form, because ``json.dumps`` cannot serialize a
    # mappingproxy and a status payload must never 500 on its own shape.
    if (not isinstance(tasks, (list, tuple))
            or len(set(tasks)) != len(tasks)
            or any(task not in _rewrite.TASKS for task in tasks)):
        raise ValueError(
            f"a rewrite backend row needs a unique subset of "
            f"{list(_rewrite.TASKS)} as tasks")
    # A row published in the status block always answers something; a row on the
    # settings page may answer NOTHING, which is how "started with this
    # checkpoint, but not one that can read your picture" reaches the page
    # instead of the option quietly vanishing from it.
    if not tasks and not settings_form:
        raise ValueError("a rewrite backend row must answer at least one draft kind")
    if type(row["uses_gpu"]) is not bool:
        raise ValueError("a rewrite backend row needs a boolean uses_gpu")
    if settings_form:
        _check_availability(row["available"], row["reason"],
                            "settings rewrite backend row")
    out = {"id": backend, "label": row["label"], "tasks": list(tasks),
           "uses_gpu": row["uses_gpu"]}
    if settings_form:
        out["available"] = row["available"]
        out["reason"] = row["reason"]
    return out


def validate_rewrite_backend_rows(rows: Any,
                                  *, settings_form: bool = False) -> list[dict[str, Any]]:
    """A non-empty list of unique backend rows, in the server's order."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("the rewrite backends list must be a non-empty list")
    checked = [validate_rewrite_backend_row(row, settings_form=settings_form)
               for row in rows]
    ids = [row["id"] for row in checked]
    if len(set(ids)) != len(ids):
        raise ValueError(f"the rewrite backends list repeats a backend: {ids}")
    return checked


def validate_rewrite_block(block: Any) -> dict[str, Any]:
    """The ONE definition of a legal rewrite status block, pure.

    Shared by :class:`WebRuntime` (which refuses a bad block at construction,
    never filtering it silently) and by the rewrite service (which builds its
    block through here), so the server cannot emit a document its own UI
    parser would reject. Raises :class:`ValueError` with a bounded message.

    The rules are the UI's rules: the four keys (plus the optional additive
    ``backends`` list and ``settings`` pointer), a CLOSED state/code pair, a
    backend IFF available, a reason IFF unavailable, no location in the reason,
    and a backend id that is one of the rows the block itself advertises.
    """
    if not isinstance(block, Mapping):
        raise ValueError("web rewrite block must be a JSON mapping")
    keys = set(block)
    if not _REWRITE_BLOCK_KEYS <= keys <= (_REWRITE_BLOCK_KEYS
                                           | _REWRITE_BLOCK_KEYS_OPTIONAL):
        raise ValueError(
            "web rewrite block must carry "
            f"{sorted(_REWRITE_BLOCK_KEYS)} (plus optionally "
            f"{sorted(_REWRITE_BLOCK_KEYS_OPTIONAL)}); "
            f"got {sorted(map(str, keys))}")
    state, code = block["state"], block["code"]
    backend, reason = block["backend"], block["reason"]
    if _REWRITE_PAIRS.get(code) != state:
        raise ValueError(
            f"web rewrite block state {state!r} does not pair with code {code!r}")
    if not isinstance(reason, str):
        raise ValueError("web rewrite block reason must be a string")
    if state == "available":
        if not isinstance(backend, str) or not backend or len(backend) > 64:
            raise ValueError(
                "an available rewrite block needs a non-empty backend (<= 64 chars)")
        # The backend is echoed to the browser too: it is a LABEL ("local-llm"),
        # never an address or path.
        if not _BACKEND_LABEL_RE.match(backend):
            raise ValueError(
                "a rewrite backend must be a label (letters, digits, . _ -), "
                "not an address or path")
        if reason != "":
            raise ValueError("an available rewrite block carries no reason")
    else:
        if backend is not None:
            raise ValueError("an unavailable rewrite block has backend null")
        if not reason.strip():
            raise ValueError("an unavailable rewrite block needs a reason")
    for marker in _LOCATION_MARKERS:
        if marker in reason:
            raise ValueError(
                "a rewrite reason must not disclose a location (URL or path)")
    if _IPV4_RE.search(reason):
        raise ValueError("a rewrite reason must not disclose a location (address)")
    if "backends" in block:
        rows = validate_rewrite_backend_rows(block["backends"])
        # Normalized, so what ``WebRuntime`` stores is JSON-serializable even
        # though it arrived frozen (a tuple of mappingproxies).
        block = {**block, "backends": rows}
        if state == "available" and backend not in [row["id"] for row in rows]:
            # The block would advertise one running rewriter and a different
            # list of what may be chosen: refuse the mismatch.
            raise ValueError(
                f"an available rewrite block must list its own backend "
                f"{backend!r} among its backends")
    if "settings" in block:
        block = {**block, "settings": validate_rewrite_settings_pointer(
            block["settings"])}
    if "active_by_task" in block:
        block = {**block, "active_by_task": validate_rewrite_active_by_task(
            block["active_by_task"],
            rows=block.get("backends"), state=state)}
    return dict(block)


def validate_rewrite_settings_pointer(pointer: Any) -> dict[str, Any]:
    """The ``settings`` key of a status block: WHERE the choice is changed.

    The amendment asks status to report "the settings location". On a wire the
    browser reads, a location is a ROUTE — the on-disk file lives in the
    server's cache dir and is named on the operator's console, never here.
    """
    if (not isinstance(pointer, Mapping) or set(pointer) != {"href", "saved"}
            or pointer["href"] != REWRITE_SETTINGS_HREF
            or not isinstance(pointer["saved"], bool)):
        raise ValueError(
            "a rewrite settings pointer must be exactly "
            f'{{"href": {REWRITE_SETTINGS_HREF!r}, "saved": <bool>}}')
    return {"href": REWRITE_SETTINGS_HREF, "saved": pointer["saved"]}


def validate_rewrite_active_by_task(active: Any, *, rows: Any = None,
                                    state: str = "available") -> dict[str, Any]:
    """The ``active_by_task`` key: which rewriter answers EACH draft kind.

    One entry per task in :data:`qwen_engine.rewrite.TASKS`, each either a
    backend id or null (nothing answers that kind right now). A name must be a
    rewriter the SAME block advertises and that can handle that task — a block
    claiming the enhancer answers an edit draft while its own row says the
    enhancer does text only is a contradiction, not a status.

    Accepts the frozen form (a mapping) and always returns a plain dict, so a
    status payload cannot 500 on its own shape.
    """
    if not isinstance(active, Mapping) or set(active) != set(_rewrite.TASKS):
        raise ValueError(
            "rewrite active_by_task must carry exactly "
            f"{list(_rewrite.TASKS)}; got "
            f"{sorted(map(str, active)) if isinstance(active, Mapping) else type(active).__name__}")
    by_row = {row["id"]: row for row in rows} if rows else {}
    checked: dict[str, Any] = {}
    for task in _rewrite.TASKS:
        backend = active[task]
        if backend is not None:
            if not isinstance(backend, str) or backend not in REWRITE_BACKENDS:
                raise ValueError(
                    f"rewrite active_by_task[{task!r}] must be one of "
                    f"{list(REWRITE_BACKENDS)} or null")
            if by_row:
                row = by_row.get(backend)
                if row is None:
                    raise ValueError(
                        f"rewrite active_by_task names {backend!r}, which the "
                        "block does not list among its backends")
                if task not in row["tasks"]:
                    raise ValueError(
                        f"rewrite active_by_task says {backend!r} answers "
                        f"{task!r}, but that rewriter handles {row['tasks']}")
        checked[task] = backend
    if state == "available" and not any(checked.values()):
        raise ValueError(
            "an available rewrite block must have at least one task answered")
    if state != "available" and any(checked.values()):
        raise ValueError(
            "an unavailable rewrite block answers no task")
    return checked


def _safe(text: Any, limit: int = MAX_MESSAGE_CHARS) -> str:
    """One-line, length-bounded public message: control characters (and any
    newline smuggled through a repr) collapse to a space; over-length text
    gets an explicit ellipsis instead of an internal dump."""
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return flat[:limit - 1] + "\u2026"


class ApiError(Exception):
    """Typed, transport-ready failure: HTTP status + stable code + bounded
    safe message (+ optional per-field list). Never carries raw internals."""

    def __init__(self, status: int, code: str, message: str,
                 field_errors: tuple[tuple[str, str], ...] = (),
                 headers: tuple[tuple[str, str], ...] = ()) -> None:
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.message = _safe(message)
        self.field_errors = tuple((f, _safe(m)) for f, m in field_errors)
        self.headers = headers  # e.g. Allow on 405; transport-relevant only

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": WEB_ERROR_SCHEMA,
            "code": self.code,
            "message": self.message,
        }
        if self.field_errors:
            out["fields"] = [{"field": f, "message": m}
                             for f, m in self.field_errors]
        return out


@dataclass(frozen=True)
class WebRuntime:
    """Server-controlled runtime context, canonicalised at startup.

    ``manifest_summary`` is the metadata-only summary computed once when the
    server started (see :func:`summarize_manifest_bytes` / the server's
    bounded one-shot file read). It is never re-read per request and never
    triggers artifact verification — that would stream-hash gigabytes.
    """

    output_dir: str | None = None
    cache_dir: str | None = None
    log_dir: str | None = None
    benchmarks_dir: str | None = None
    models_dir: str | None = None
    manifest_path: str | None = None
    q8_manifest_path: str | None = None
    #: Server-owned content-addressed reference store (uploads land here;
    #: the browser exchanges ids only, never paths).
    references_dir: str | None = None
    manifest_summary: Mapping[str, Any] = field(default_factory=dict)
    generation_enabled: bool = False
    #: The rewriter capability block, computed ONCE at startup from the
    #: service configuration (never per request, never a filesystem read).
    #: ``None`` means "rewriting is not available here" and is rendered as
    #: the canonical disabled block — the same thing an unconfigured server
    #: reports. Additive to the status document: the schema tag does not
    #: change (plan §4.1).
    rewrite_block: Mapping[str, Any] | None = None
    #: The first-run model setup service (:mod:`qwen_engine.model_setup`), or
    #: ``None`` on a server that cannot offer downloads. Not JSON: it is read by
    #: the ``/api/v1/models`` routes and by :func:`_blockers`, and never enters a
    #: document, so it stays out of ``freeze_json``.
    setup_service: Any = None

    def __post_init__(self) -> None:
        if type(self.generation_enabled) is not bool:
            raise ValueError("generation_enabled must be a bool")
        errors: list[str] = []
        summary = _jsonsafe.freeze_json(self.manifest_summary, "manifest_summary", errors)
        if errors or not isinstance(summary, Mapping):
            raise ValueError("web manifest summary must be a strict JSON mapping")
        object.__setattr__(self, "manifest_summary", summary)
        if self.rewrite_block is not None:
            block = _jsonsafe.freeze_json(self.rewrite_block, "rewrite_block", errors)
            if errors or not isinstance(block, Mapping):
                raise ValueError("web rewrite block must be a strict JSON mapping")
            # ONE validator, so the server can never emit a block its own UI
            # parser would reject (and never leak a backend location).
            object.__setattr__(self, "rewrite_block", validate_rewrite_block(block))
        # A service, or nothing: a half-wired object must not half-work.
        if self.setup_service is not None and not callable(
                getattr(self.setup_service, "required_missing", None)):
            object.__setattr__(self, "setup_service", None)

    def path_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in RUNTIME_PATH_FIELDS}


# ---------------------------------------------------------------------------
# Strict JSON parsing (shared typed boundary for every POST body)
# ---------------------------------------------------------------------------


class _DuplicateKey(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate JSON key {key!r}")
        self.key = key


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(key)
        out[key] = value
    return out


def _finite_float(raw: str) -> float:
    value = float(raw)  # float("1e999999...") yields inf, never raises
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number literal {raw!r}")
    return value


def _iter_strings(value: Any):
    """Iterative (never recursive) walk so no parsed shape can blow the
    stack while we look for unpaired surrogates."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            for ch in item:
                if "\ud800" <= ch <= "\udfff":
                    return True
        elif isinstance(item, Mapping):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def parse_request_json(raw: bytes) -> Any:
    """Bytes -> strict JSON value with every hostile shape typed away:
    invalid UTF-8, duplicate keys (recursively), NaN/Inf literals, float
    overflow to infinity, non-finite results, unpaired surrogates, and
    structures beyond the shared depth bound. Never a raw traceback."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "invalid_utf8",
                       "request body is not valid UTF-8") from None
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_pairs,
                          parse_constant=_jsonsafe.reject_constant,
                          parse_float=_finite_float)
    except _DuplicateKey as exc:
        raise ApiError(400, "duplicate_json_key", _safe(exc)) from None
    except RecursionError:
        raise ApiError(400, "json_too_deep",
                       "request JSON nests deeper than the parser accepts") from None
    except ValueError as exc:
        # JSONDecodeError, the int-string digit limit, our float guard and
        # reject_constant all surface as typed parse failures here.
        raise ApiError(400, "invalid_json",
                       _safe(f"request body is not valid JSON: {exc}")) from None
    violations = _jsonsafe.json_violations(data, at="request")
    if violations:
        raise ApiError(400, "invalid_json_structure", _safe(violations[0]))
    if _iter_strings(data):
        raise ApiError(400, "invalid_text",
                       "request contains unpaired Unicode surrogates")
    return data


# ---------------------------------------------------------------------------
# Envelope + config validation (server runtime paths injected here only)
# ---------------------------------------------------------------------------


def build_config(runtime: WebRuntime, data: Any) -> GenerationConfig:
    """Validate the exact request envelope and build the merged
    ``GenerationConfig`` (editable browser fields + server runtime paths)."""
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "request must be a JSON object with exactly the keys "
                       "'schema' and 'config'")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    missing = sorted({"schema", "config"} - set(data))
    extra = sorted(set(data) - {"schema", "config"})
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing envelope key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected envelope key(s): " + ", ".join(extra))
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != WEB_PLAN_REQUEST_SCHEMA:
        raise ApiError(400, "invalid_envelope",
                       f"schema must be {WEB_PLAN_REQUEST_SCHEMA!r}")
    raw_config = data["config"]
    if not isinstance(raw_config, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "'config' must be a JSON object")
    if any(not isinstance(key, str) for key in raw_config):
        raise ApiError(400, "unknown_config_field", "config keys must be strings")
    unknown = sorted(set(raw_config) - EDITABLE_FIELDS)
    if unknown:
        raise ApiError(
            400, "unknown_config_field",
            "config fields are not browser-editable (unknown or runtime "
            "path fields are refused, never stripped): " + ", ".join(unknown),
        )
    merged: dict[str, Any] = dict(runtime.path_fields())
    for key, value in raw_config.items():
        merged[key] = _seed_wire_to_int(value) if key == "seed" else value
    try:
        return GenerationConfig.from_dict(merged)
    except ConfigError as exc:
        # Errors on browser-editable fields are reported verbatim (they
        # describe only browser data). Runtime-path fields NEVER surface:
        # startup validation is not a guarantee (TOCTOU — e.g. a configured
        # dir replaced by a file afterwards), and ConfigError messages for
        # path fields can contain the raw absolute path. A runtime-path
        # mismatch is therefore ONE generic code with no path, no stat
        # detail and no raw message.
        public: list[tuple[str, str]] = []
        runtime_hit = False
        for err in exc.errors:
            if err.field in RUNTIME_PATH_FIELDS:
                runtime_hit = True
            else:
                public.append((err.field, err.message))
        code = "invalid_config"
        if runtime_hit:
            code = "server_runtime_configuration"
            public.append(("server_runtime_configuration",
                           "one or more server runtime paths configured at "
                           "startup no longer validate; a server operator "
                           "must fix or restart qwen-web"))
        # Final leak guard: no configured runtime path value may appear in
        # any message, whatever upstream produced it.
        path_values = [v for v in runtime.path_fields().values() if v]
        if any(v in message for _, message in public for v in path_values):
            public = [("server_runtime_configuration",
                       "one or more server runtime paths configured at "
                       "startup no longer validate; a server operator "
                       "must fix or restart qwen-web")]
            code = "server_runtime_configuration"
        raise ApiError(422, code, "configuration is invalid",
                       tuple(public)) from None


def _seed_wire_to_int(value: Any) -> int | None:
    """Decimal-string/null seed wire -> Python int, typed. Numbers, bools,
    over-long digit strings and any non-ASCII digits are refused — never
    coerced through a float and never rounded."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str):
        raise ApiError(400, "invalid_seed",
                       "seed must be a decimal string (at most 20 ASCII "
                       "digits) or null — numeric JSON is refused because "
                       "JavaScript numbers would round past 2^53")
    if not value:
        raise ApiError(400, "invalid_seed",
                       "seed string must contain at least one ASCII digit")
    if _SEED_WIRE_RE.fullmatch(value):
        return int(value)
    raise ApiError(400, "invalid_seed",
                    "seed must be at most 20 ASCII decimal digits "
                    "(0 to 18446744073709551615); the text was refused "
                    "exactly as entered")


def _public_seed(config: GenerationConfig) -> str | None:
    """Public seed wire: decimal STRING or null — never numeric JSON."""
    if config.seed is None:
        return None
    return str(int(config.seed))


def editable_config_dict(config: GenerationConfig) -> dict[str, Any]:
    """The editable subset of a validated config with the public seed wire
    (decimal string / null). This is what plan echoes back.

    Deferred fields are dropped for the same reason runtime paths are: the
    wire is only ever what the browser can send back. Echoing a field that
    ``build_config`` refuses would break export -> import round-trips with a
    config the server itself produced.
    """
    out = config.to_dict()
    for name in (*RUNTIME_PATH_FIELDS, *DEFERRED_WIRE_FIELDS):
        out.pop(name, None)
    out["seed"] = _public_seed(config)
    out["references"] = list(config.references)
    return out


# ---------------------------------------------------------------------------
# Manifest metadata (never verification — no streaming hashes for the UI)
# ---------------------------------------------------------------------------

#: Stable reason codes for the manifest gate.
MANIFEST_UNCONFIGURED = "manifest_unconfigured"
MANIFEST_FILE_MISSING = "manifest_file_missing"
MANIFEST_INVALID = "manifest_invalid"
MANIFEST_UNREADABLE = "manifest_unreadable"


def summarize_manifest_bytes(raw: bytes | None, unreadable: str | None = None) -> dict[str, Any]:
    """Metadata-only summary of the (already bounded) manifest file bytes.

    Parses with the existing strict :func:`manifest_from_dict` — no
    ``resolve_manifest`` / ``manifest_status`` anywhere near the UI path,
    because verification would stream-hash every weight. Reports identity
    *counts and revisions* only: never filenames, sha256 hashes or paths.
    ``unreadable`` is the typed read-failure code from the server's one-shot
    bounded read (missing / not a regular file / too large / ...).
    """
    base: dict[str, Any] = {"configured": True, "valid": False,
                            "verification": "not_performed"}
    if unreadable is not None:
        # CLOSED reason codes only: whatever the caller's string was, an
        # arbitrary value (which could embed a path) never reaches the wire.
        base["reason"] = unreadable if unreadable in (
            MANIFEST_FILE_MISSING, MANIFEST_INVALID, MANIFEST_UNREADABLE,
            MANIFEST_UNCONFIGURED) else MANIFEST_UNREADABLE
        return base
    if raw is None:
        return {"configured": False, "valid": False, "reason": MANIFEST_UNCONFIGURED,
                "verification": "not_performed"}
    from qwen_engine.manifest import ManifestFormatError, manifest_from_dict
    try:
        manifest = manifest_from_dict(_jsonsafe.strict_json_loads(raw.decode("utf-8")))
    except (ManifestFormatError, UnicodeDecodeError, ValueError, RecursionError):
        base["reason"] = MANIFEST_INVALID
        return base
    return {
        "configured": True,
        "valid": True,
        "schema_version": manifest.schema_version,
        "artifact_count": len(manifest.artifacts),
        "revisions": sorted({a.revision for a in manifest.artifacts}),
        "verification": "not_performed",
    }


def _manifest_core(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Metadata-only manifest block shared by status and plan: never
    filenames, hashes or paths, and always verification:"not_performed"."""
    state: dict[str, Any] = {
        "configured": summary.get("configured") is True,
        "valid": False,
        "verification": "not_performed",
    }
    schema, count = summary.get("schema_version"), summary.get("artifact_count")
    state["valid"] = (state["configured"] and summary.get("valid") is True
                      and type(schema) is int and schema == 1
                      and type(count) is int and count > 0)
    if state["valid"]:
        state.update(schema_version=schema, artifact_count=count)
    else:
        reason = summary.get("reason")
        reasons = (MANIFEST_UNCONFIGURED, MANIFEST_FILE_MISSING,
                   MANIFEST_INVALID, MANIFEST_UNREADABLE)
        state["reason"] = (reason if isinstance(reason, str) and reason in reasons
                           else MANIFEST_INVALID if state["configured"]
                           else MANIFEST_UNCONFIGURED)
    return state


def manifest_plan_state(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The plan's explicit metadata-only manifest state: NOT verified, fit
    not evaluated, never 'ready' — an invalid/unconfigured manifest is a
    blocker, but planning itself still proceeds."""
    state = _manifest_core(summary)
    state["fit"] = "not_evaluated"
    state["ready"] = False
    return state


def _manifest_blocker(summary: Mapping[str, Any]) -> str | None:
    state = _manifest_core(summary)
    if not state["configured"]:
        return "manifest_unconfigured"
    if not state["valid"]:
        reason = state["reason"]
        return {"manifest_file_missing": MANIFEST_FILE_MISSING,
                MANIFEST_INVALID: MANIFEST_INVALID,
                MANIFEST_UNREADABLE: MANIFEST_UNREADABLE}.get(reason, MANIFEST_INVALID)
    return None


#: Capabilities/evidence, separate from static submission blockers.
def _gates() -> dict[str, dict[str, str]]:
    return {
        "adapter": {
            "state": "available",
            "code": "adapter_available",
            "reason": "the FP16 resident Qwen adapter is implemented; actual "
                      "load and resource checks run inside each job",
        },
        "hardware": {
            "state": "unknown",
            "code": "hardware_not_probed",
            "reason": "status/plan do not probe hardware; each actual job "
                      "validates its device, so submission availability is "
                      "not hardware verification",
        },
        "assets": {
            "state": "not_verified",
            "code": "assets_unverified",
            "reason": "artifact verification is not performed by the status "
                      "or plan path (it would stream-hash every weight); "
                      "verification belongs to an actual run",
        },
        "calibration": {
            "state": "unknown",
            "code": "calibration_unknown",
            "reason": "calibration is not VERIFIED for this server: this "
                      "status/plan path does not inspect any benchmark "
                      "records; memory fitness is unknown here, never "
                      "claimed established or absent",
        },
    }


def _blockers(runtime: WebRuntime) -> list[str]:
    """Static submission blockers, never a claim of measured memory fitness.

    Hardware/assets are checked by the actual engine job, just as in the CLI.
    Exact calibration preflight remains a separate, unimplemented integration.
    """
    codes = [] if runtime.generation_enabled else ["generation_disabled"]
    if any(getattr(runtime, name) is None for name in
           ("output_dir", "cache_dir", "models_dir", "manifest_path")):
        codes.append("runtime_unconfigured")
    blocker = _manifest_blocker(runtime.manifest_summary)
    if blocker:
        codes.append(blocker)
    service = getattr(runtime, "setup_service", None)
    if service is not None and service.required_missing():
        # Generate cannot succeed without the base assets, so the page says so
        # with one button ("Get the basics") instead of a job that would fail in
        # the loader. The blocker list is additive: no schema tag moves.
        codes.append(MODELS_MISSING)
    return codes


def startup_blockers(runtime: WebRuntime) -> list[str]:
    """The blockers that must refuse a generation worker, and the one that must
    not: a machine whose weights are simply not here yet is the machine that
    needs the page.

    ``models_missing`` is excluded because first-run setup IS the way those
    bytes arrive: refusing to build the worker would leave a server that cannot
    start at all, which is worse than a server that starts, shows the Models
    section, and refuses a job until the download has finished. Every other
    blocker (generation off, an unconfigured runtime, an unreadable manifest)
    still means there is nothing to run jobs against.

    Both ``web.server.main`` and :class:`WebGeneration` ask THIS question, so
    the startup check and the worker's own guard can never drift apart.
    """
    return [code for code in _blockers(runtime) if code != MODELS_MISSING]


def models_missing(runtime: WebRuntime) -> bool:
    """True while a required model of this server's is not fully on disk."""
    return MODELS_MISSING in _blockers(runtime)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResponse:
    """Transport-ready result: status + JSON payload, or raw bytes for the
    attachment export (never re-serialized by the transport)."""

    status: int
    payload: Mapping[str, Any] | None = None
    body: bytes | None = None
    headers: tuple[tuple[str, str], ...] = ()


def status_response(runtime: WebRuntime, *, server_version: str | None = None,
                    admin: bool | None = None,
                    user: Mapping[str, Any] | None = None) -> ApiResponse:
    """The status document. Additive and closed: no schema bump.

    Every addition is OPTIONAL and listener-supplied, so this stays the pure
    runtime document: ``admin`` is True only where the owner-only routes exist
    (the local port); ``server_version`` is the checkout's git short sha read
    once at startup (C5, ``"unknown"`` when there is none); ``user`` is the
    remote listener's identity block
    (:func:`qwen_engine.web.remote.user_block`), already built by the transport
    that proved the identity. A caller that passes none of them gets exactly
    the key set this document had before, which is what keeps the schema tag
    un-bumped and every existing status test meaningful.
    """
    summary = dict(runtime.manifest_summary)
    payload = {
        "schema": WEB_STATUS_SCHEMA,
        "planning_available": True,
        "generation_available": not _blockers(runtime),
        "gates": _gates(),
        "manifest": _manifest_core(summary),
        "generation_blockers": _blockers(runtime),
        # Additive and optional (plan §4.1): the schema tag is NOT bumped, and
        # nothing here participates in the generation gate.
        "rewrite": rewrite_status_block(runtime),
    }
    # Each addition is present only when the LISTENER supplies it, so this
    # function stays the pure runtime document: a caller that owns the value
    # (the local listener's ``admin: true``, the remote listener's identity
    # block, the startup read of C5's version) decides, and one that passes
    # nothing gets exactly the key set this document had before.
    if server_version is not None:
        payload["server_version"] = server_version
    if admin is not None:
        payload["admin"] = admin
    if user is not None:
        payload["user"] = user
    return ApiResponse(status=200, payload=payload)


def _plan_core_document(runtime: WebRuntime, config: GenerationConfig) -> dict[str, Any]:
    """core.plan runs on the MERGED config; the raw document contains runtime
    paths and a numeric seed, so it is only ever an internal input to the
    WEB PROJECTION below — never returned verbatim."""
    return core_plan(config, manifest=dict(runtime.manifest_summary))


def plan_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    try:
        config = build_config(runtime, parse_request_json(raw_body))
    except ApiError as exc:
        return ApiResponse(status=exc.status, payload=exc.to_payload())
    core_doc = _plan_core_document(runtime, config)
    payload = {
        "schema": WEB_PLAN_SCHEMA,
        "planning_available": True,
        "generation_available": not _blockers(runtime),
        "config": editable_config_dict(config),
        "cfg_active": core_doc["cfg_active"],
        "latent_tokens": core_doc["latent_tokens"],
        "hardware": {"status": core_doc["hardware"].get("status", "unknown"),
                     "detail": "this qwen-web server has not probed hardware; "
                               "unknown here, never claimed present, absent "
                               "or verified"},
        "calibration": {"status": core_doc["calibration"].get("status", "unknown"),
                        "detail": "calibration is not verified for this "
                                  "server; the status/plan path does not "
                                  "inspect benchmark records"},
        "manifest": manifest_plan_state(runtime.manifest_summary),
        "generation_blockers": _blockers(runtime),
    }
    return ApiResponse(status=200, payload=payload)


def export_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    """Raw-attachment export of the browser-editable GenerationConfig
    fields. Runtime paths are OMITTED and the seed is the EXACT uint64
    Python number in JSON — these bytes are downloaded as a blob and must
    never be JSON-parsed/reserialized by the browser."""
    try:
        config = build_config(runtime, parse_request_json(raw_body))
    except ApiError as exc:
        return ApiResponse(status=exc.status, payload=exc.to_payload())
    out = config.to_dict()
    for name in (*RUNTIME_PATH_FIELDS, *DEFERRED_WIRE_FIELDS):
        out.pop(name, None)
    body = (json.dumps(out, sort_keys=True) + "\n").encode("utf-8")
    return ApiResponse(
        status=200, body=body,
        headers=(("Content-Disposition",
                  'attachment; filename="qwen-editable-config.json"'),),
    )


def import_response(runtime: WebRuntime, raw_body: bytes) -> ApiResponse:
    """Validate an editable export without rounding its numeric uint64 seed.

    Files contain raw JSON config; Python parses integers exactly before the
    ordinary browser envelope validator rejects runtime/unknown fields.
    """
    try:
        data = parse_request_json(raw_body)
        if not isinstance(data, Mapping):
            raise ApiError(400, "invalid_config_file", "configuration file must contain a JSON object")
        config = dict(data)
        if type(config.get("seed")) is int:
            config["seed"] = str(config["seed"])
        envelope = {"schema": WEB_PLAN_REQUEST_SCHEMA, "config": config}
        # Reuse plan projection so the browser has one authoritative config
        # parser, with paths omitted and seeds always decimal strings.
        return plan_response(runtime, json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
    except ApiError as exc:
        return ApiResponse(exc.status, exc.to_payload())


def upload_reference_response(runtime: WebRuntime, raw: bytes, *,
                              owner: str = LOCAL_OWNER) -> ApiResponse:
    """Store one raw uploaded reference image (design §2.4).

    Validation is the store's (magic sniff, Pillow decode, size/dimension
    caps, store fullness); every refusal is typed and prompt-free: 415 for
    non-images, 409 when the store is at its cap, 503 when the server has no
    usable store — never a path, never a library message. Idempotent: the
    same bytes return the same id.

    ``owner`` (C3) records WHICH account may reference these bytes. The store
    stays content-addressed and shared, so two accounts uploading identical
    bytes share one blob; the edge is what makes each of them able to use it
    and nobody else.
    """
    if runtime.references_dir is None:
        raise ApiError(503, "reference_store_unconfigured",
                       "the server reference store directory is not configured")
    if not raw:
        raise ApiError(415, "unsupported_media_type",
                       "an empty body is not an image")
    try:
        # The server-owned store directory: created 0700 at startup by the
        # server, re-created here if an operator removed it mid-run (uploads
        # are the only writers; content-addressed, so no clobbering).
        os.makedirs(runtime.references_dir, mode=0o700, exist_ok=True)
    except OSError:
        raise ApiError(503, "reference_store_unavailable",
                       "the reference store directory is not usable") from None
    try:
        reference_id = _reference_store.put_bytes(runtime.references_dir, raw)
        sidecar = _reference_store.read_sidecar(runtime.references_dir, reference_id)
        _reference_store.add_owner(runtime.references_dir, owner, reference_id)
    except _reference_store.ReferenceStoreError as exc:
        text = str(exc)
        if text.startswith(("cannot publish", "reference root is not")):
            raise ApiError(503, "reference_store_unavailable",
                           "the reference store could not accept the upload") from None
        if "store is full" in text:
            raise ApiError(409, "reference_store_full",
                           "the reference store is at its cap; an operator must "
                           "prune it") from None
        raise ApiError(415, "unsupported_media_type",
                       "the body must be a valid PNG, JPEG or WebP image "
                       "within the size and dimension caps") from None
    return ApiResponse(201, {"schema": WEB_REFERENCE_SCHEMA, "id": reference_id,
                             "width": sidecar["width"], "height": sidecar["height"],
                             "has_alpha": bool(sidecar["has_alpha"]),
                             "bytes": len(raw)})


def reference_bytes(runtime: WebRuntime, reference_id: str, *,
                    thumb: bool = False,
                    owner: str | None = None) -> tuple[bytes, str]:
    """Hash-verified stored bytes (original or PNG thumbnail) + sniffed
    Content-Type. Unknown/malformed ids and unreadable blobs are ONE 404 —
    the client learns nothing beyond absence.

    ``owner`` (C3) narrows the answer to ids THAT account uploaded: somebody
    else's id is refused with the same fixed 404 an unknown id gets, so an id
    cannot be confirmed by guessing it. ``None`` is the owner-side path.
    """
    if runtime.references_dir is None:
        raise ApiError(404, "reference_unknown", "no such reference")
    if owner is not None:
        try:
            if owner not in _reference_store.owners_for(runtime.references_dir, reference_id):
                raise ApiError(404, "reference_unknown", "no such reference")
        except _reference_store.ReferenceStoreError:
            raise ApiError(404, "reference_unknown", "no such reference") from None
    try:
        data = (_reference_store.read_thumb(runtime.references_dir, reference_id)
                if thumb else
                _reference_store.read_blob(runtime.references_dir, reference_id))
    except _reference_store.ReferenceStoreError:
        raise ApiError(404, "reference_unknown", "no such reference") from None
    if thumb:
        return data, "image/png"
    try:
        return data, _REFERENCE_CONTENT_TYPES[_reference_store.sniff_format(data)]
    except _reference_store.ReferenceStoreError:
        raise ApiError(404, "reference_unknown", "no such reference") from None


def build_rewrite_request(raw: bytes) -> dict[str, Any]:
    """``/api/v1/rewrite`` body -> the exact four-key request mapping.

    The wire envelope carries EXACTLY
    ``schema``/``task``/``prompt``/``draft_revision``/``image_ids``, so a typo
    is a typed refusal rather than a silently ignored field. Envelope shape is
    checked here; field SEMANTICS live in :func:`validate_rewrite_fields`,
    which the service calls too — one implementation, so the transport and the
    executor cannot disagree about what a legal request is.

    The prompt is validated for shape and size only; it is never logged, never
    echoed into an error message, and never copied anywhere but the single
    ``original_prompt`` field of a successful response.
    """
    data = parse_request_json(raw)
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the rewrite request must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    allowed = ("schema", "task", "prompt", "draft_revision", "image_ids")
    missing = sorted(set(allowed) - set(data))
    extra = sorted(set(data) - set(allowed))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected key(s): " + ", ".join(extra))
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != REWRITE_REQUEST_SCHEMA:
        raise ApiError(400, "invalid_envelope",
                       f"schema must be {REWRITE_REQUEST_SCHEMA!r}")
    return validate_rewrite_fields({k: v for k, v in data.items() if k != "schema"})


#: At most this many reference images in one edit rewrite. DERIVED from the
#: edit-job cap (``config.REF_MAX``) rather than chosen here: a rewrite that
#: accepted more references than the job it describes could not be generated.
MAX_REWRITE_IMAGES = REF_MAX


def _check_rewrite_image_ids(value: Any) -> list[str]:
    """Validate ``image_ids`` as an ORDERED list of stored reference ids.

    Order is the whole point: the edit system prompt addresses references as
    ``<imageN>``, so the Nth id must stay the Nth image. Duplicates are
    allowed, matching edit-job validation (``config`` caps the count and
    checks the id grammar; it does not forbid repeating one).

    Identity, not existence: whether an id resolves to a readable blob is the
    service's question (it owns the store directory), and it answers 404.
    """
    if not isinstance(value, list):
        raise ApiError(400, "invalid_envelope",
                       "image_ids must be a list of reference ids")
    for ordinal, entry in enumerate(value, start=1):
        if not _reference_store.is_reference_id(entry):
            raise ApiError(
                400, "invalid_envelope",
                f"image_ids[{ordinal - 1}] must be a 64-character lowercase "
                "hex reference id")
    if len(value) > MAX_REWRITE_IMAGES:
        raise ApiError(422, "too_many_references",
                       f"at most {MAX_REWRITE_IMAGES} reference images are "
                       f"allowed, got {len(value)}")
    return list(value)


def validate_rewrite_fields(request: Mapping[str, Any]) -> dict[str, Any]:
    """Semantic validation of the four request fields, pure.

    Shared by the transport (after the envelope check) and by
    ``RewriteService.submit`` (which is also called directly by tests), so
    there is exactly ONE definition of a legal rewrite request.
    """
    if not isinstance(request, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the rewrite request must be a JSON object")
    allowed = {"task", "prompt", "draft_revision", "image_ids"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ApiError(400, "invalid_envelope",
                       "unknown rewrite field(s): "
                       + ", ".join(map(str, unknown)))
    missing = sorted(allowed - set(request))
    if missing:
        raise ApiError(400, "invalid_envelope",
                       "missing rewrite field(s): " + ", ".join(missing))
    try:
        task = _rewrite.check_task(request["task"])
    except _rewrite.RewriteError as exc:
        raise ApiError(422, "task_unsupported", str(exc)) from None
    image_ids = _check_rewrite_image_ids(request["image_ids"])
    if task == "t2i" and image_ids:
        # Never silently downgrade: a t2i draft has no <imageN> to point at, so
        # sending the pictures would change what the rewrite means.
        raise ApiError(422, "task_unsupported",
                       "a t2i rewrite takes no reference images; use task 'edit'")
    if task == "edit" and not image_ids:
        # The edit system prompt addresses references as <imageN>; with none,
        # the model would invent them, and validation must not pretend a
        # text-only edit rewrite is the same request.
        raise ApiError(422, "reference_required",
                       "an edit rewrite needs at least one reference image id")
    revision = request["draft_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ApiError(400, "invalid_envelope",
                       "draft_revision must be a non-negative integer")
    prompt = request["prompt"]
    if not isinstance(prompt, str):
        raise ApiError(400, "invalid_envelope", "prompt must be a string")
    if len(prompt) > _rewrite.PROMPT_MAX_CHARS:
        # Length is the only prompt-derived fact ever put in a message.
        raise ApiError(422, "prompt_too_long",
                       f"the prompt is {len(prompt)} characters; the limit is "
                       f"{_rewrite.PROMPT_MAX_CHARS}")
    if not prompt.strip():
        raise ApiError(400, "invalid_envelope", "prompt must be non-empty")
    return {"task": task, "prompt": prompt, "draft_revision": revision,
            "image_ids": image_ids}


#: The canonical "no rewriter here" block. One source of truth: a server built
#: without a rewrite service reports this exact shape, so the UI's strict
#: parser has one thing to recognise. A live service OVERWRITES the reason with
#: its own (it knows whether the user switched it off or the flags are missing)
#: and reports whether a saved document is on disk.
REWRITE_DISABLED_BLOCK: Mapping[str, Any] = {
    "state": "unavailable",
    "code": "rewrite_disabled",
    "backend": None,
    "reason": "this server was started without a prompt rewriter",
    "settings": {"href": REWRITE_SETTINGS_HREF, "saved": False},
}


def rewrite_status_block(runtime: WebRuntime) -> dict[str, Any]:
    """The additive, OPTIONAL ``rewrite`` status block (plan §4.1).

    Four keys, closed value sets, never a host, port or path: this document
    is served to the browser, and the reason string is operator-facing prose
    that must not disclose where the backend lives. ``GATE_STATES`` does not
    grow — rewriting is an optional capability and must never block (or
    appear to block) generation.

    ``WebRuntime`` already refuses any block that is not legal, so this is a
    copy, not a filter; :func:`validate_rewrite_block` is that single rule.
    """
    block = runtime.rewrite_block
    if block is None:
        return dict(REWRITE_DISABLED_BLOCK)
    return dict(block)


# ---------------------------------------------------------------------------
# the Settings contract: which rewriter this server uses (server-side, shared)
# ---------------------------------------------------------------------------

#: Schema tags for the Settings round trip and the connection probe.
REWRITE_SETTINGS_SCHEMA = "qwen_engine.web_rewrite_settings.v1"
REWRITE_TEST_REQUEST_SCHEMA = "qwen_engine.web_rewrite_test_request.v1"
REWRITE_TEST_SCHEMA = "qwen_engine.web_rewrite_test.v1"

#: The closed key set of the saved/POSTed choice. Six keys and NONE of them is
#: a path. The llama-server binary, the GGUFs, the system-prompt files and the
#: enhancer timeout stay launch flags: a browser that may name a file can name
#: any file on this machine, so the rule is absolute rather than scoped to a few
#: field names. The choice itself is PER TASK (`t2i`, `edit`) plus one `enabled`
#: switch; the old single ``backend`` key is not accepted here — a file that
#: still carries it is migrated before it reaches this validator.
_REWRITE_SETTINGS_KEYS = ("t2i", "edit", "enabled", "endpoint", "model",
                          "timeout_s")

#: The probe's own request keys (it always tests the LOCAL service: the
#: enhancer cannot be probed without loading 9.5 GiB onto the GPU).
_REWRITE_TEST_KEYS = ("endpoint", "model", "timeout_s")

#: A key shaped like a filesystem location. Refused with its own sentence, so
#: a mistaken client learns the rule instead of a bare "unknown key".
_PATH_SHAPED_KEY_RE = re.compile(
    r"[/\\]|(^|_)(path|dir|folder|file|binary|bin|exe|gguf|model_path)$", re.I)


def validate_rewrite_settings(payload: Any, *,
                              allow_unset_choice: bool = False,
                              require_endpoint: bool = True) -> dict[str, Any]:
    """The ONE definition of a legal rewriter choice, pure.

    Shared by ``POST /api/v1/settings/rewrite``, the saved document on disk and
    the settings state validator, so a file written by an older version,
    hand-edited, or produced by the route cannot disagree with the route about
    what is legal.

    ``require_endpoint`` is off ONLY for the saved record, whose endpoint and
    model are overrides: a record saying "the endpoint for edits, address from
    the launch flags" is half-empty by design, and demanding an address there
    made a server refuse to publish its own Settings page. The rule that an
    enabled endpoint choice needs an address and a model name applies to the
    EFFECTIVE document — what the page posts and what the server acts on.

    ``t2i`` and ``edit`` are CLOSED per task (``pe-llamacpp`` / ``local-llm``)
    and ``enabled`` is the single Off switch — ``off`` is deliberately NOT a
    choice value any more, because "both tasks off" and "the switch is off" are
    one fact that must be stated once. ``endpoint``, when present, must pass
    :func:`qwen_engine.rewrite_client.parse_endpoint` (loopback, explicit port,
    no credentials), since a prompt draft is user text; ``timeout_s`` is in
    ``(0, 3600]``. The three endpoint fields describe the SHARED service and may
    be saved while neither task uses it; they are required only when a task
    chooses it. ``allow_unset_choice`` is for a saved document (and the ``saved``
    view inside the settings state), where a null choice means "not chosen: use
    the default"; it is never accepted from the browser.
    """
    if not isinstance(payload, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the rewrite settings must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ApiError(400, "invalid_envelope", "rewrite setting keys must be strings")
    unknown = sorted(set(payload) - set(_REWRITE_SETTINGS_KEYS))
    if unknown:
        message = "unknown rewrite setting(s): " + ", ".join(map(str, unknown))
        if "backend" in unknown:
            message += (" — the rewriter is now chosen per draft kind (t2i, "
                        "edit); a file that still carries a single backend is "
                        "migrated when the server reads it")
        if any(_PATH_SHAPED_KEY_RE.search(key) for key in unknown):
            message += (" — this server takes no filesystem path from the "
                        "browser")
        raise ApiError(400, "invalid_envelope", message)
    # ``allow_unset_choice`` means a choice key may be ABSENT, not merely null:
    # a saved document carries what the user actually picked, and the state
    # validator runs on output this module built — so the rules must be
    # idempotent, or the server refuses to send what it just made.
    required = set(_REWRITE_SETTINGS_KEYS)
    if allow_unset_choice:
        required -= {"t2i", "edit", "enabled"}
    missing = sorted(required - set(payload))
    if missing:
        raise ApiError(400, "invalid_envelope",
                       "missing rewrite setting(s): " + ", ".join(missing))
    choices: dict[str, Any] = {}
    for task in _rewrite.TASKS:
        choice = payload.get(task)
        if choice is None:
            if not allow_unset_choice:
                raise ApiError(400, "invalid_envelope",
                               f"{task} must be one of {list(REWRITE_BACKENDS)}")
        elif choice not in REWRITE_BACKENDS:
            raise ApiError(422, "backend_unknown",
                           f"the rewriter for a {task} draft must be one of "
                           f"{list(REWRITE_BACKENDS)}, got {choice!r}")
        choices[task] = choice
    enabled = payload.get("enabled")
    if enabled is None:
        if not allow_unset_choice:
            raise ApiError(400, "invalid_envelope",
                           "enabled must be true or false")
    elif not isinstance(enabled, bool):
        raise ApiError(400, "invalid_envelope", "enabled must be true or false",
                       (("enabled", "choose a rewriter or switch rewriting off"),))
    values: dict[str, Any] = {}
    for name in ("endpoint", "model"):
        value = payload[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            values[name] = None
            continue
        if not isinstance(value, str):
            raise ApiError(400, "invalid_envelope",
                           f"{name} must be a string or null",
                           ((name, f"{name} must be text, or empty for none"),))
        values[name] = value.strip()
    timeout_s = payload["timeout_s"]
    if timeout_s is None:
        values["timeout_s"] = None
    else:
        if (isinstance(timeout_s, bool)
                or not isinstance(timeout_s, (int, float))
                or not 0 < float(timeout_s) <= 3600):
            raise ApiError(400, "invalid_envelope",
                           "timeout_s must be a number of seconds in "
                           "(0, 3600], or null",
                           (("timeout_s", "seconds must be in (0, 3600]"),))
        values["timeout_s"] = float(timeout_s)
    if values["endpoint"] is not None:
        try:
            _rewrite_client.parse_endpoint(values["endpoint"])
        except _rewrite_client.RewriteConfigError as exc:
            # The user typed this address, so the bounded reason may quote it;
            # it is never copied into the STATUS block, whose validator refuses
            # any location.
            raise ApiError(422, "endpoint_unsupported", _safe(exc),
                           (("endpoint",
                             "use http on a loopback address with an explicit "
                             "port, for example "
                             "http://127.0.0.1:8079/v1/chat/completions"),)) from None
    if (require_endpoint and enabled
            and any(choices[task] == "local-llm" for task in _rewrite.TASKS)
            and not (values["endpoint"] and values["model"])):
        # The endpoint is shared, so ONE sentence covers either task choosing
        # it: without an address and a model name there is nothing to call.
        raise ApiError(422, "backend_incomplete",
                       "the OpenAI-compatible endpoint needs both an endpoint "
                       "URL and a model name",
                       (("endpoint", "required for the OpenAI-compatible "
                                     "endpoint"),
                        ("model", "required for the OpenAI-compatible "
                                  "endpoint")))
    return {"t2i": choices["t2i"], "edit": choices["edit"],
            "enabled": enabled, "endpoint": values["endpoint"],
            "model": values["model"], "timeout_s": values["timeout_s"]}


def _rewrite_envelope(raw: bytes, allowed: tuple[str, ...],
                      schema: str, what: str) -> dict[str, Any]:
    """The shared POST envelope: a JSON object, a schema tag, and nothing
    outside the allowed keys — a typo is a refusal, not a silently dropped
    field."""
    data = parse_request_json(raw)
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope", f"the {what} must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    missing = sorted(set(allowed) - set(data))
    extra = sorted(set(data) - set(allowed))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected key(s): " + ", ".join(extra))
            if any(_PATH_SHAPED_KEY_RE.search(key) for key in extra):
                # Say WHICH rule was hit: a client that guessed a key named
                # ``model_path`` deserves the reason, not a bare "unexpected".
                parts.append("this server takes no filesystem path from the "
                             "browser")
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != schema:
        raise ApiError(400, "invalid_envelope", f"schema must be {schema!r}")
    return {key: value for key, value in data.items() if key != "schema"}


def build_rewrite_settings_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/settings/rewrite`` body -> the validated choice."""
    fields = _rewrite_envelope(raw, ("schema",) + _REWRITE_SETTINGS_KEYS,
                               REWRITE_SETTINGS_SCHEMA, "rewrite settings request")
    return validate_rewrite_settings(fields)


def build_rewrite_test_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/settings/rewrite/test`` body -> the endpoint to probe.

    The probe is always the SHARED OpenAI-compatible service: the same field
    rules apply and no choice keys are accepted, so a Test button press cannot
    ask the server to load an enhancer checkpoint onto the GPU. It does need an
    address and a model name, which is the one thing a probe is for.
    """
    fields = _rewrite_envelope(raw, ("schema",) + _REWRITE_TEST_KEYS,
                               REWRITE_TEST_REQUEST_SCHEMA,
                               "rewrite connection test request")
    checked = validate_rewrite_settings({"enabled": True,
                                         "endpoint": fields.get("endpoint"),
                                         "model": fields.get("model"),
                                         "timeout_s": fields.get("timeout_s")},
                                        allow_unset_choice=True)
    if not checked["endpoint"] or not checked["model"]:
        raise ApiError(422, "backend_incomplete",
                       "the connection test needs both an endpoint URL and a "
                       "model name",
                       (("endpoint", "required for the connection test"),
                        ("model", "required for the connection test")))
    return {key: checked[key] for key in _REWRITE_TEST_KEYS}


# ---------------------------------------------------------------------------
# First-run model setup: the catalogue the Settings page shows, and the ONE
# request shape that may ask for a download.
#
# The catalogue is built from files the SERVER owns (the committed manifests
# plus the pins in :mod:`qwen_engine.model_setup`). Neither document may carry a
# repository, a revision, a URL or a filesystem path: the page's only power is
# to name items by ``id``.
# ---------------------------------------------------------------------------

#: Routes, published once so the page never types one by hand.
MODELS_HREF = "/api/v1/models"
MODELS_DOWNLOAD_HREF = MODELS_HREF + "/download"
MODELS_CANCEL_HREF = MODELS_DOWNLOAD_HREF + "/cancel"

#: A generation blocker, and the app.js list it must appear in.
MODELS_MISSING = "models_missing"

#: A request key that names a SOURCE rather than a choice. The catalogue lives in
#: this checkout, so a browser that sends a repo, revision, URL, hash or path is
#: asking for something this server will never fetch; the refusal names the rule
#: it hit instead of answering with a bare "unexpected key".
_SOURCE_SHAPED_KEY_RE = re.compile(
    r"[/\\]|(^|_)(path|dir|folder|file|binary|bin|exe|gguf|model_path|repo|"
    r"repository|revision|ref|commit|url|uri|link|sha|sha256|hash|digest|"
    r"filename|source)$", re.I)

_MODELS_KEYS = ("schema", "items", "bytes_needed", "free_bytes",
                "required_missing", "can_download", "server_command", "job",
                "notices")
_MODEL_ITEM_KEYS = ("id", "label", "required", "reason", "files", "dest",
                    "installed", "missing_files", "bytes_needed", "extra")
_MODEL_FILE_KEYS = ("name", "bytes")
_MODEL_JOB_KEYS = ("schema", "state", "ids", "files", "bytes_done",
                   "bytes_total", "error")
_MODEL_JOB_FILE_KEYS = ("item", "name", "done", "total")
_MODEL_ERROR_KEYS = ("code", "message")

#: A destination is a repo-relative directory and nothing else: no leading
#: slash, no parent step, no drive. It is display text, and keeping it that
#: small is what makes "the browser never learns a path" checkable.
_DEST_RE = re.compile(r"[A-Za-z0-9._-]{1,64}(?:/[A-Za-z0-9._-]{1,64}){0,4}")
_MODEL_NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}(?:/[A-Za-z0-9._-]{1,128}){0,4}")

#: The message a non-loopback client is given. It is the whole remedy: the
#: command runs on the machine the models live on.
SETUP_SERVER_COMMAND = ("python tools/fetch_models.py --manifest "
                        "models/manifest.json --models-dir models/assets --fetch")


def _bounded_text(value: Any, what: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"{what} must be a string of 1-{limit} characters")
    return value


def _bounded_count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{what} must be a non-negative integer")
    return value


def _plain_relative(value: str, what: str) -> str:
    """A relative name with no parent step. The regexes above already refuse a
    leading slash, a drive and any separator but ``/``; this closes the last door
    an attacker has left, which is ``..`` as a path segment."""
    if value in (".", "..") or any(part == ".." for part in value.split("/")):
        raise ValueError(f"{what} must not step outside the directory it is in")
    return value


def _validate_model_file(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != set(_MODEL_FILE_KEYS):
        raise ValueError("a model file must carry exactly "
                         f"{list(_MODEL_FILE_KEYS)}")
    name = _bounded_text(row["name"], "a model file name", 256)
    if _MODEL_NAME_RE.fullmatch(name) is None:
        raise ValueError("a model file name must be a plain relative name")
    return {"name": _plain_relative(name, "a model file name"),
            "bytes": _bounded_count(row["bytes"], "a model file size")}


def validate_model_item(row: Any) -> dict[str, Any]:
    """One catalogue row, pure. Raises :class:`ValueError`, which the web layer
    turns into a console warning and a Settings notice: a broken catalogue must
    never reach the page as a half-rendered list."""
    if not isinstance(row, Mapping) or set(row) != set(_MODEL_ITEM_KEYS):
        raise ValueError(f"a model catalogue row must carry exactly "
                         f"{list(_MODEL_ITEM_KEYS)}")
    item_id = _bounded_text(row["id"], "a model id", 64)
    if item_id not in _setup.ITEM_IDS:
        raise ValueError(f"unknown model id {item_id!r}")
    files = row["files"]
    if (not isinstance(files, (list, tuple)) or not files
            or len(files) > _setup.MAX_FILES_PER_ITEM):
        raise ValueError(f"model {item_id!r} needs between one and "
                         f"{_setup.MAX_FILES_PER_ITEM} files")
    dest = _bounded_text(row["dest"], "a model destination", 320)
    if _DEST_RE.fullmatch(dest) is None:
        raise ValueError("a model destination must be a repo-relative "
                         "directory, never an absolute path")
    dest = _plain_relative(dest, "a model destination")
    if row["installed"] not in _setup.INSTALL_STATES:
        raise ValueError(f"model {item_id!r} has an unknown install state "
                         f"{row['installed']!r}")
    missing = row["missing_files"]
    if (not isinstance(missing, (list, tuple))
            or any(not isinstance(name, str) or not name or len(name) > 256
                   or ".." in name.split("/") for name in missing)):
        raise ValueError(f"model {item_id!r} lists missing files badly")
    if row["extra"] not in ("", _setup.NEEDS_BINARY):
        raise ValueError(f"model {item_id!r} carries an unknown extra "
                         "requirement")
    if not isinstance(row["required"], bool):
        raise ValueError(f"model {item_id!r} needs a boolean required flag")
    return {
        "id": item_id,
        "label": _bounded_text(row["label"], "a model label", 120),
        "required": row["required"],
        "reason": _bounded_text(row["reason"], "a model reason", 400),
        "files": [_validate_model_file(one) for one in files],
        "dest": dest,
        "installed": row["installed"],
        "missing_files": [str(name) for name in missing],
        "bytes_needed": _bounded_count(row["bytes_needed"],
                                       "a model byte count"),
        "extra": row["extra"],
    }


def validate_models_job(job: Any) -> dict[str, Any]:
    """The one download job, pure; ``None`` means idle."""
    if job is None:
        return {"schema": _setup.JOB_SCHEMA, "state": _setup.STATE_IDLE,
                "ids": [], "files": [], "bytes_done": 0, "bytes_total": 0,
                "error": None}
    if not isinstance(job, Mapping) or set(job) != set(_MODEL_JOB_KEYS):
        raise ValueError(f"a download job must carry exactly "
                         f"{list(_MODEL_JOB_KEYS)}")
    if job["schema"] != _setup.JOB_SCHEMA:
        raise ValueError(f"a download job must be tagged "
                         f"{_setup.JOB_SCHEMA!r}")
    if job["state"] not in _setup.JOB_STATES:
        raise ValueError(f"unknown download state {job['state']!r}")
    ids = job["ids"]
    if (not isinstance(ids, (list, tuple))
            or any(item not in _setup.ITEM_IDS for item in ids)):
        raise ValueError("a download job lists an unknown model id")
    rows = job["files"]
    if not isinstance(rows, (list, tuple)) or len(rows) > _setup.MAX_FILES_PER_ITEM:
        raise ValueError("a download job lists too many files")
    files = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(_MODEL_JOB_FILE_KEYS):
            raise ValueError("a download progress row must carry exactly "
                             f"{list(_MODEL_JOB_FILE_KEYS)}")
        name = _bounded_text(row["name"], "a download file name", 256)
        if _MODEL_NAME_RE.fullmatch(name) is None:
            raise ValueError("a download file name must be a plain relative name")
        done = _bounded_count(row["done"], "a download byte count")
        total = _bounded_count(row["total"], "a download byte total")
        if done > total:
            raise ValueError("a download row counted more bytes than the file has")
        files.append({"item": _bounded_text(row["item"], "a model id", 64),
                      "name": _plain_relative(name, "a download file name"),
                      "done": done, "total": total})
    error = job["error"]
    if error is None:
        checked_error = None
    elif isinstance(error, Mapping) and set(error) == set(_MODEL_ERROR_KEYS):
        checked_error = {
            "code": _bounded_text(error["code"], "a download error code", 64),
            "message": _bounded_text(error["message"],
                                     "a download error message", 400)}
    else:
        raise ValueError("a download error must carry exactly "
                         f"{list(_MODEL_ERROR_KEYS)} or be null")
    bytes_done = _bounded_count(job["bytes_done"], "a download byte count")
    bytes_total = _bounded_count(job["bytes_total"], "a download byte total")
    if bytes_done > bytes_total:
        raise ValueError("the download counted more bytes than it was asked for")
    return {"schema": _setup.JOB_SCHEMA, "state": job["state"],
            "ids": [str(item) for item in ids], "files": files,
            "bytes_done": bytes_done, "bytes_total": bytes_total,
            "error": checked_error}


def validate_models_state(document: Any) -> dict[str, Any]:
    """The whole ``GET /api/v1/models`` document, pure.

    The server validates its own catalogue before it sends it, the same way the
    rewriter choice is validated before it is saved: a page must never receive a
    document it cannot parse, and the key set is closed so a stale page and a
    new server fail loudly rather than silently.
    """
    if not isinstance(document, Mapping) or set(document) != set(_MODELS_KEYS):
        raise ApiError(500, "models_unavailable",
                       "the model catalogue is malformed",
                       (("models", "the server could not build a model list"),))
    if document["schema"] != _setup.MODELS_SCHEMA:
        raise ApiError(500, "models_unavailable",
                       "the model catalogue is malformed",
                       (("models", "the server could not build a model list"),))
    items = document["items"]
    if not isinstance(items, (list, tuple)) or not items:
        raise ApiError(500, "models_unavailable",
                       "the model catalogue is empty",
                       (("models", "no model could be read from this "
                                   "server's pinned list"),))
    try:
        rows = [validate_model_item(row) for row in items]
        job = validate_models_job(document["job"])
        needed = _bounded_count(document["bytes_needed"],
                                "the catalogue byte total")
        free = document["free_bytes"]
        free = None if free is None else _bounded_count(free, "the free space")
        command = _bounded_text(document["server_command"],
                                "the setup command", 320)
        if command != SETUP_SERVER_COMMAND:
            # One definition, pinned like the schema tag: the page renders this
            # string as the remedy, so it may only ever be the real command.
            raise ValueError("the setup command is not the one this build ships")
        notices = document["notices"]
        if not isinstance(notices, (list, tuple)):
            raise ValueError("the setup notices must be a list of sentences")
        notices = [_bounded_text(one, "a setup notice", 400) for one in notices]
        for one in notices:
            if "://" in one or "//" in one or "/home/" in one or "/tmp/" in one:
                raise ValueError("a setup notice must not name a location")
    except ValueError as exc:
        raise ApiError(500, "models_unavailable",
                       "the model catalogue is malformed",
                       (("models", str(exc)[:400]),)) from None
    # The same rule the service applies: a required id that is not in the rows
    # at all (an unreadable base manifest) is missing too, so a server that
    # cannot list the base model still reports the truth instead of being
    # refused by its own consistency check.
    required_missing = any(
        row["required"] and row["installed"] != _setup.INSTALLED for row in rows
    ) or any(item_id not in {row["id"] for row in rows}
             for item_id in _setup.REQUIRED_IDS)
    if bool(document["required_missing"]) != required_missing:
        raise ApiError(500, "models_unavailable",
                       "the model catalogue is inconsistent",
                       (("models", "the required-model flag did not match the "
                                   "rows"),))
    return {"schema": _setup.MODELS_SCHEMA, "items": rows,
            "bytes_needed": needed, "free_bytes": free,
            "required_missing": required_missing,
            "can_download": bool(document["can_download"]),
            "server_command": command, "job": job, "notices": notices}


def build_models_download_request(raw: bytes) -> dict[str, Any]:
    """``POST /api/v1/models/download`` body -> ``{"ids": [...]}``.

    Ids only. There is no key that could carry a path, a repo or a revision, and
    an empty id list is legal: the required items are always added by the
    service, which is what makes "Get the basics" a one-button action.
    """
    data = parse_request_json(raw)
    if not isinstance(data, Mapping):
        raise ApiError(400, "invalid_envelope",
                       "the model download request must be a JSON object")
    if any(not isinstance(key, str) for key in data):
        raise ApiError(400, "invalid_envelope", "request keys must be strings")
    missing = sorted({"schema", "ids"} - set(data))
    extra = sorted(set(data) - {"schema", "ids"})
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing key(s): " + ", ".join(missing))
        if extra:
            parts.append("unexpected key(s): " + ", ".join(extra))
            if any(_SOURCE_SHAPED_KEY_RE.search(key) for key in extra):
                parts.append("this server takes no repository, revision, URL, "
                             "hash or path from the browser: a download is "
                             "requested by model id only")
        raise ApiError(400, "invalid_envelope", "; ".join(parts))
    if data["schema"] != _setup.DOWNLOAD_REQUEST_SCHEMA:
        raise ApiError(400, "invalid_envelope",
                       f"schema must be {_setup.DOWNLOAD_REQUEST_SCHEMA!r}")
    ids = data["ids"]
    if (not isinstance(ids, (list, tuple))
            or any(not isinstance(item, str) or not item or len(item) > 64
                   for item in ids)):
        raise ApiError(422, "invalid_request",
                       "the download ids must be a list of model id strings",
                       (("ids", f"expected some of: {', '.join(_setup.ITEM_IDS)}"),))
    return {"ids": list(dict.fromkeys(ids))}
#: The Settings page's per-task groups: one row per draft kind, each listing
#: every rewriter with `available`/`reason` already resolved.
_TASK_ROW_KEYS = frozenset({"task", "label", "choices"})
_TASK_CHOICE_KEYS = frozenset({"id", "label", "available", "uses_gpu", "reason"})


def validate_rewrite_task_rows(rows: Any) -> list[dict[str, Any]]:
    """The ``tasks`` list of the Settings payload, pure. Raises ValueError.

    The page must not have to know WHICH launch flag gates an option: the server
    answers, per draft kind, whether each rewriter can be chosen here and why not.
    Labels and order are pinned to this module's, so "Edit prompt enhance" is one
    string everywhere and a stale group heading cannot survive a save.

    Like the other row validators this accepts the frozen form and returns plain
    JSON, because ``json.dumps`` cannot serialize a mappingproxy.
    """
    if (not isinstance(rows, (list, tuple))
            or len(rows) != len(_rewrite.TASKS)):
        raise ValueError("the rewrite settings tasks must hold exactly one row "
                         f"per draft kind ({list(_rewrite.TASKS)})")
    checked = []
    for want, row in zip(_rewrite.TASKS, rows):
        if not isinstance(row, Mapping) or set(row) != _TASK_ROW_KEYS:
            raise ValueError(
                f"a rewrite settings task row must carry exactly "
                f"{sorted(_TASK_ROW_KEYS)}; got "
                f"{sorted(map(str, row)) if isinstance(row, Mapping) else type(row).__name__}")
        if row["task"] != want:
            raise ValueError("the rewrite settings tasks must be in the order "
                             f"{list(_rewrite.TASKS)}")
        if row["label"] != REWRITE_TASK_LABELS[want]:
            raise ValueError(f"the label for task {want!r} must be "
                             f"{REWRITE_TASK_LABELS[want]!r}")
        choices = row["choices"]
        if (not isinstance(choices, (list, tuple))
                or [choice["id"] for choice in choices
                    if isinstance(choice, Mapping)] != list(REWRITE_BACKENDS)):
            raise ValueError(
                "a rewrite settings task row needs one choice per rewriter, in "
                f"the order {list(REWRITE_BACKENDS)}")
        out_choices = []
        for choice in choices:
            if set(choice) != _TASK_CHOICE_KEYS:
                raise ValueError(
                    f"a rewrite task choice must carry exactly "
                    f"{sorted(_TASK_CHOICE_KEYS)}")
            backend = choice["id"]
            if choice["label"] != REWRITE_BACKEND_LABELS[backend]:
                raise ValueError(f"the label for {backend!r} must be "
                                 f"{REWRITE_BACKEND_LABELS[backend]!r}")
            if choice["uses_gpu"] is not REWRITE_BACKEND_USES_GPU[backend]:
                raise ValueError(f"{backend!r} uses_gpu must be "
                                 f"{REWRITE_BACKEND_USES_GPU[backend]}")
            _check_availability(choice["available"], choice["reason"],
                                f"{want} rewrite choice")
            out_choices.append({"id": backend, "label": choice["label"],
                                "available": choice["available"],
                                "uses_gpu": choice["uses_gpu"],
                                "reason": choice["reason"]})
        checked.append({"task": want, "label": row["label"],
                        "choices": out_choices})
    return checked


def validate_rewrite_settings_state(payload: Any) -> dict[str, Any]:
    """The ``GET /api/v1/settings/rewrite`` payload, checked before it is sent.

    This is the ONE document allowed to carry the endpoint the user typed (they
    configured it). It is never allowed to carry a filesystem path: no binary,
    no GGUF, no prompt file, no log file — and the closed key set is what makes
    that checkable rather than hoped for.
    """
    allowed = {"schema", "enabled", "choices", "saved", "local", "backends",
               "tasks", "off", "persistent", "note"}
    if not isinstance(payload, Mapping):
        raise ValueError("the rewrite settings state must be a JSON mapping")
    if set(payload) - allowed:
        raise ValueError("unexpected rewrite settings state key(s): "
                         + ", ".join(sorted(map(str, set(payload) - allowed))))
    if payload["schema"] != REWRITE_SETTINGS_SCHEMA:
        raise ValueError("rewrite settings state schema must be "
                         f"{REWRITE_SETTINGS_SCHEMA!r}")
    if not isinstance(payload["enabled"], bool):
        raise ValueError("the rewrite settings enabled flag must be a boolean")
    choices = payload["choices"]
    if (not isinstance(choices, Mapping)
            or set(choices) != set(_rewrite.TASKS)):
        raise ValueError("the rewrite settings choices must carry exactly "
                         f"{list(_rewrite.TASKS)}")
    for task in _rewrite.TASKS:
        if choices[task] not in REWRITE_BACKENDS:
            raise ValueError(
                f"the effective rewriter choice for {task!r} must be one of "
                f"{list(REWRITE_BACKENDS)} — the page needs a checked radio, "
                "never a null")
    # The saved record is what the last save chose, overrides and all: its
    # endpoint may legitimately be empty because the launch flags supply one.
    # The address rule belongs to the effective document on the next line.
    saved = validate_rewrite_settings(payload["saved"], allow_unset_choice=True,
                                      require_endpoint=False)
    local = validate_rewrite_settings(payload["local"], allow_unset_choice=True)
    rows = validate_rewrite_backend_rows(payload["backends"], settings_form=True)
    if [row["id"] for row in rows] != list(REWRITE_BACKENDS):
        raise ValueError("the settings backends must be "
                         f"{list(REWRITE_BACKENDS)}, in that order")
    task_rows = validate_rewrite_task_rows(payload["tasks"])
    off = payload["off"]
    if (not isinstance(off, Mapping) or set(off) != {"id", "label", "available", "reason"}
            or off["id"] != "off" or off["label"] != REWRITE_BACKEND_LABELS["off"]
            or off["available"] is not True or off["reason"] != ""):
        raise ValueError("the settings off-row must be off/Off/available/no reason")
    if type(payload["persistent"]) is not bool:
        raise ValueError("the rewrite settings persistent flag must be a boolean")
    note = payload.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise ValueError("the rewrite settings note must be a non-empty string")
    checked = {"schema": REWRITE_SETTINGS_SCHEMA,
               "enabled": payload["enabled"],
               "choices": {task: choices[task] for task in _rewrite.TASKS},
               "saved": saved,
               # What the SHARED endpoint would be reached as RIGHT NOW: the
               # saved override where there is one, else the launch flags. The
               # page prefills from this, so it never shows a field the server
               # is not actually using.
               "local": {key: local[key] for key in ("endpoint", "model",
                                                     "timeout_s")},
               "backends": rows, "tasks": task_rows, "off": dict(off),
               "persistent": payload["persistent"]}
    if note is not None:
        checked["note"] = note
    return checked


def rewrite_test_result(ok: bool, detail: str, elapsed_ms: int) -> dict[str, Any]:
    """The probe's answer, with any location or command line removed from the
    plain-words line.

    A failed connection may quote an address; a spawn failure may quote the
    command that failed. The page only needs to know that it failed and why in
    words, so addresses, path markers and option flags are stripped here
    rather than trusted to whatever the exception happened to carry.
    """
    text = strip_command(detail)
    return {"schema": REWRITE_TEST_SCHEMA, "ok": bool(ok),
            "detail": _safe(text) or ("ok" if ok else "failed"),
            "elapsed_ms": max(0, int(elapsed_ms))}


def internal_error_response() -> ApiResponse:
    """Catch-all typed failure: no traceback, no exception text, no paths."""
    exc = ApiError(500, "internal_error",
                   "the server failed to handle the request; check the "
                   "server console for a bounded diagnostic")
    return ApiResponse(status=500, payload=exc.to_payload())


__all__ = [
    "EDITABLE_FIELDS", "RUNTIME_PATH_FIELDS", "ApiError", "ApiResponse",
    "WEB_ERROR_SCHEMA", "WEB_PLAN_REQUEST_SCHEMA", "WEB_PLAN_SCHEMA",
    "WEB_STATUS_SCHEMA", "build_config", "editable_config_dict",
    "export_response", "internal_error_response", "manifest_plan_state",
    "parse_request_json", "plan_response", "status_response",
    "summarize_manifest_bytes",
    "MAX_REFERENCE_UPLOAD_BYTES", "WEB_REFERENCE_SCHEMA",
    "reference_bytes", "upload_reference_response",
    # Prompt rewriting (plan §4.1): the wire contract, not the transport.
    "REWRITE_REQUEST_SCHEMA", "WEB_REWRITE_SCHEMA",
    "MAX_REWRITE_REQUEST_BYTES", "REWRITE_DISABLED_BLOCK",
    "build_rewrite_request", "validate_rewrite_fields", "validate_rewrite_block",
    "rewrite_status_block",
    # Two backends plus the server-side Settings choice (schema tags, closed
    # rows, and the one validator each document goes through).
    "REWRITE_BACKENDS", "REWRITE_SETTINGS_BACKENDS", "REWRITE_BACKEND_LABELS",
    "REWRITE_BACKEND_USES_GPU", "REWRITE_TASK_LABELS", "REWRITE_SETTINGS_HREF",
    "REWRITE_SETTINGS_SCHEMA", "REWRITE_TEST_REQUEST_SCHEMA",
    "REWRITE_TEST_SCHEMA", "validate_rewrite_backend_row",
    "validate_rewrite_backend_rows", "validate_rewrite_task_rows",
    "validate_rewrite_active_by_task", "validate_rewrite_settings",
    "validate_rewrite_settings_state", "build_rewrite_settings_request",
    "build_rewrite_test_request", "rewrite_test_result", "strip_command",
    "MODELS_HREF", "MODELS_DOWNLOAD_HREF", "MODELS_CANCEL_HREF",
    "MODELS_MISSING", "SETUP_SERVER_COMMAND", "build_models_download_request",
    "validate_model_item", "validate_models_job", "validate_models_state",
]


# Guard against accidental field drift between config.py and this contract.
def _check_editable_subset() -> None:
    names = {f.name for f in fields(GenerationConfig)}
    assert EDITABLE_FIELDS <= names, "EDITABLE_FIELDS references unknown config fields"
    assert not (EDITABLE_FIELDS & set(RUNTIME_PATH_FIELDS)), \
        "a field cannot be both browser-editable and server-controlled"
    assert names - EDITABLE_FIELDS - set(RUNTIME_PATH_FIELDS) == set(DEFERRED_WIRE_FIELDS), \
        ("every GenerationConfig field must be browser-editable, a runtime "
         "path, or an announced deferral — unclassified: "
         f"{sorted(names - EDITABLE_FIELDS - set(RUNTIME_PATH_FIELDS) - set(DEFERRED_WIRE_FIELDS))}")
    assert DEFERRED_WIRE_FIELDS <= names, \
        f"DEFERRED_WIRE_FIELDS names unknown config fields: {sorted(DEFERRED_WIRE_FIELDS - names)}"


_check_editable_subset()
