"""Default local loader + Qwen-Image-2.1 diffusers adapter (M4 groundwork).

Boundary rules:

* Importing this module stays **ML-free**. ``torch``, ``diffusers`` and
  ``transformers`` are imported only inside :meth:`Qwen21DiffusersAdapter.load`,
  i.e. only on an actual run.
* Every weight comes from the verified local reference bundle
  (``manifest`` + ``models_dir`` -> :func:`resolve_local_bundle`), never from a
  repo id, an ambient Hugging Face cache or the network. The CLI has already
  resolved the same bundle once; the adapter re-resolves so it never trusts a
  caller-supplied path list, and pays the documented re-hash cost.
* Only the reviewed resident profile is implemented: unquantized FP16 on one
  V100, native SDPA, transformer + VAE resident, no text encoder in-process
  (the Qwen3-VL encoder runs in the subprocess worker), no hooks/offload.
  Q8 uses the separately verified GGUF path. Streaming remains unimplemented
  and is refused rather than silently ignored.
* The adapter **loads and attaches**; it grants no memory fit, no calibration
  and no image-quality acceptance. The measured memory policy (M5.3) remains a
  separate gate, and the CPU encoder's own conservative host-memory preflight
  still refuses until complete trusted budget evidence exists. The single
  opt-in exception is the constructor's ``trust_unbounded_host`` — the
  operator's own attestation that this process tree has no cgroup memory
  limit; it is off by default and the gate still applies a fresh
  ``MemAvailable`` read, so it can never supply a number.
* Nothing here writes outside the configured cache/output directories, changes
  a driver, touches ROCm, or flushes a global allocator on behalf of other
  processes.
"""
from __future__ import annotations

import gc
import inspect
import threading
from functools import partial
from pathlib import Path
from typing import Any

from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine.adapters.supplied_qwen21 import (load_reference_images,
                                                  SuppliedQwen21Adapter)
from qwen_engine.config import (
    AttentionBackend,
    ConfigError,
    FieldError,
    GenerationConfig,
    OffloadMode,
    Quant,
    Recipe,
    TURBO_STEPS,
)
from qwen_engine.encoder import EmbeddingCache
from qwen_engine.encoder_client import ProductionEncoder
from qwen_engine.errors import EngineError, HardwareAbsentError
from qwen_engine.hardware_probe import is_v100_memory
from qwen_engine.local_bundle import (LocalReferenceBundle, resolve_local_bundle,
                                      TURBO_REFERENCE_PROFILE)
from qwen_engine.manifest import load_manifest
from qwen_engine.runtime_identity import observe_generation_runtime
from qwen_engine.performance import stage, attach_cuda, current

#: The only encoder dtype this loader asks the CPU worker for. The checkpoint is
#: BF16, so this expands no weights; fp16/fp32 are alternatives only after a
#: measured decision (C5).
ENCODER_DTYPE = "bf16"

#: Explicit, non-defaulted host headroom passed to the encoder worker's
#: conservative preflight. It is a *prerequisite* input, not a measured peak.
ENCODER_HEADROOM_BYTES = 4 * 1024**3

#: Bounded encoder worker lifetime; a full 17.5 GB CPU load is slow.
ENCODER_TIMEOUT_S = 3600.0

#: The eight sigmas the Turbo transformer was evaluated with (turbo-plan
#: §3.1). ``FlowMatchEulerDiscreteScheduler.__call__`` IGNORES
#: ``num_inference_steps`` when the pipeline config carries these, which is
#: why the recipe owns the step count. Copy-pasted deliberately — the value
#: is the reviewed artifact constant; tests re-derive it from the bundle
#: file's sha256 so a drift can never pass silently.
TURBO_SAMPLE_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508,
                       0.845148, 0.704534, 0.414568)

#: The Turbo ``scheduler_config.json`` content (its diffusers ``_``-prefixed
#: metadata keys excluded), as shipped in the bundle. Against the base file
#: the diff is exactly two fields: ``shift_terminal`` 0.02 -> null and
#: ``use_dynamic_shifting`` true -> false. Guarded constant: a test ties it
#: content-for-content to the sha256-pinned bundle file listed in
#: models/turbo/manifest.json, and every turbo load re-checks the on-disk
#: file (:func:`_assert_turbo_scheduler`).
TURBO_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def turbo_bundle_paths(manifest_path: str | Path) -> tuple[Path, Path]:
    """Return ``(turbo manifest path, turbo asset root)`` for a run whose BASE
    manifest lives at ``manifest_path``.

    Layout convention (models/turbo/manifest.json beside models/assets/):
    both sit under ``<base manifest dir>/turbo/``. Pure path derivation — no
    filesystem access; resolution/fail-closed verification happens via
    ``resolve_local_bundle`` with :data:`TURBO_REFERENCE_PROFILE`.
    """
    root = Path(manifest_path).parent / "turbo"
    return root / "manifest.json", root / "assets"


def _assert_turbo_scheduler(scheduler_dir: str | Path) -> None:
    """The Turbo weights are only evaluated on the shipped Turbo schedule.

    The bundle's ``scheduler_config.json`` (minus diffusers' ``_``-prefixed
    metadata keys) must equal :data:`TURBO_SCHEDULER_CONFIG` exactly — the
    shipped file's two Turbo properties, ``shift_terminal`` null and
    ``use_dynamic_shifting`` false, versus the base file's 0.02/true. The
    FILE is checked, never the loaded scheduler object: ``from_pretrained``
    blends class defaults into ``scheduler.config``, so an inherited base
    config could otherwise masquerade as a Turbo one (§3.2).
    """
    path = Path(scheduler_dir) / "scheduler_config.json"
    try:
        document = strict_json_loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EngineError(
            f"the turbo scheduler config is unreadable: "
            f"{type(exc).__name__}: {exc}") from None
    if not isinstance(document, dict):
        raise EngineError("the turbo scheduler config is not a JSON object")
    observed = {key: value for key, value in document.items()
                if not key.startswith("_")}
    if observed != TURBO_SCHEDULER_CONFIG:
        common = set(observed) & set(TURBO_SCHEDULER_CONFIG)
        changed = sorted(key for key in common
                         if observed[key] != TURBO_SCHEDULER_CONFIG[key])
        missing = sorted(set(TURBO_SCHEDULER_CONFIG) - set(observed))
        unexpected = sorted(set(observed) - set(TURBO_SCHEDULER_CONFIG))
        detail = []
        if changed:
            detail.append("changed: " + ", ".join(changed))
        if missing:
            detail.append("missing: " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected: " + ", ".join(unexpected))
        raise EngineError("the turbo scheduler config disagrees with the "
                          "reviewed Turbo schedule (" + "; ".join(detail) + ")")


#: The reviewed execution device is the first CUDA logical device. The
#: hardware gate below still verifies its identity before anything is loaded.
LOGICAL_DEVICE_INDEX = 0

#: The pinned pipeline's ``output_resolution`` default. The bridge never
#: overrides it, so reference preprocessing must target exactly this area —
#: both at load time (precomputed latents) and inside ``__call__``.
REFERENCE_OUTPUT_RESOLUTION = 1024

#: Exact pinned ``QwenImage21Pipeline._encode_vae_image`` parameter list
#: (bound). The load-phase encode and the per-call override refuse on ANY
#: drift instead of guessing.
_ENCODE_VAE_IMAGE_PARAMETERS = ("image", "generator")

#: Latent-token count above which the transformer's SwiGLU MLP is evaluated in
#: token chunks (1024² is exactly 4096, so 1024² and below stay unchunked).
#: At 2048² the joint sequence is 16,384 tokens and one SwiGLU intermediate is
#: ~386 MiB; two of them are live at once, which is what pushed the measured
#: 2K attempt over the 16 GiB card. Chunking is exact — the MLP is token-wise —
#: so it changes peak memory and kernel count, never the result.
MLP_CHUNK_ABOVE_TOKENS = 4096

#: Token chunk used above that threshold. 2048 tokens keeps each intermediate
#: near 48 MiB and measured 2K denoise inside the card with headroom.
MLP_CHUNK_TOKENS = 2048

#: Tiled-decode geometry. The library default (256 px tiles, 192 px stride)
#: leaves a visible grid of dashed seams on a 2048² image; 1024 px tiles with
#: a 768 px stride leave one boundary per axis and measured clean at 1:1.
TILED_DECODE_TILE = 1024
TILED_DECODE_STRIDE = 768

#: Ordered reference-image cap accepted by this loader (plan §4: ten, subject
#: to measured V100 limits). Refused before any manifest read or ML import;
#: the config layer validates its own identical bound independently.
REFERENCE_MAX = 10


def _chunked_mlp_forward(inner_forward, torch, chunk: int, hidden_states):
    """Evaluate a token-wise MLP in fixed token chunks (exact, not an approximation)."""
    tokens = hidden_states.shape[1]
    if tokens <= chunk:
        return inner_forward(hidden_states)
    parts = [inner_forward(hidden_states[:, start:start + chunk])
             for start in range(0, tokens, chunk)]
    return torch.cat(parts, dim=1)


def _apply_mlp_chunking(torch, transformer, chunk: int) -> int:
    """Bound the SwiGLU intermediates for high token counts.

    Only the per-block image MLP is touched: it is token-wise, so slicing the
    sequence dimension is mathematically identical to one pass. Text tokens do
    not pass through it. Returns the number of wrapped blocks for the caller's
    evidence; a model without the expected blocks refuses rather than silently
    running unchunked.
    """
    blocks = getattr(transformer, "transformer_blocks", None)
    if not blocks:
        raise EngineError("transformer layout has no blocks to chunk")
    wrapped = already = 0
    for block in blocks:
        mlp = getattr(block, "img_mlp", None)
        if mlp is None:
            continue
        if getattr(mlp, "_token_chunk", None) is not None:
            already += 1
            continue
        mlp._token_chunk = chunk
        mlp.forward = partial(_chunked_mlp_forward, mlp.forward, torch, chunk)
        wrapped += 1
    if wrapped + already != len(blocks):
        raise EngineError(
            f"MLP chunking wrapped {wrapped} of {len(blocks)} transformer blocks")
    return wrapped


def _reference_token_total(resized_dimensions) -> int:
    """Packed denoise-sequence tokens contributed by reference images.

    Each reference is resized to the pinned 1024²-area rule (multiples of 32),
    the VAE compresses 16x spatially, and ``_pack_latents`` groups 2x2 latent
    patches: exactly one token per 32x32 pixels. Pure arithmetic so the CPU
    tests cover it; the GPU path feeds it the pinned
    ``calculate_dimensions`` results.
    """
    return sum((width // 32) * (height // 32)
               for width, height in resized_dimensions)


def _libraries():
    """Import the heavy stack *and* validate the pinned runtime first.

    Package policy runs before every ML import, exactly like the supplied
    component bridge; a drifted environment refuses here rather than loading
    mismatched classes.
    """
    from qwen_engine.encoder_identity import (
        observe_installed_packages,
        validate_installed_packages,
    )

    validate_installed_packages(observe_installed_packages())
    import torch
    from diffusers import (
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from transformers import Qwen3VLProcessor

    return (
        torch,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
        AutoencoderKLQwenImage21,
        FlowMatchEulerDiscreteScheduler,
        Qwen3VLProcessor,
    )


def _require_v100(torch) -> Any:
    """Refuse anything but the reviewed Volta profile before a weight is read.

    Device *enumeration* is not execution proof, but a wrong CUDA build, a
    non-Volta device or an absent sm_70 kernel set cannot run this stack, and
    must surface as the hardware exit code (5) rather than a load traceback.
    """
    if getattr(torch.version, "cuda", None) != "12.6" or not torch.cuda.is_available():
        raise HardwareAbsentError(
            "reviewed CUDA 12.6 PyTorch build and a visible V100 are required; "
            "install the pinned cu126 wheel and check the device"
        )
    if not 0 <= LOGICAL_DEVICE_INDEX < torch.cuda.device_count():
        raise HardwareAbsentError("the reviewed logical CUDA device is not visible")
    properties = torch.cuda.get_device_properties(LOGICAL_DEVICE_INDEX)
    if (
        (properties.major, properties.minor) != (7, 0)
        or "V100" not in properties.name
        or not is_v100_memory(properties.total_memory)
    ):
        raise HardwareAbsentError(
            "a Tesla V100 (16GB or 32GB) is required; "
            f"found {properties.name!r} (compute {properties.major}.{properties.minor})"
        )
    if "sm_70" not in torch.cuda.get_arch_list():
        raise HardwareAbsentError(
            "the installed torch build has no sm_70 kernels; the pinned cu126 "
            "wheel is required for Volta"
        )
    return properties


def _fp32_decode(config: GenerationConfig) -> bool:
    """Decode precision policy: fp32 for tiled decode, transparent output,
    and reference edits.

    Tiled decode and RGBA (af37930) are the two measured fp16 failures; the
    third is the V100 Q8 2-reference edit (batch4), which sampled cleanly
    and died at ``vae_decoding`` with exactly af37930's non-finite
    attestation — an edit through the joint sequence can land latents the
    fp16 VAE cannot carry even when the output is opaque. Cost is nil by
    construction: the VAE is the only resident component at decode time
    (the transformer is long released) and fp32 1K decode is measured fine
    everywhere fp16 was.
    """
    return (config.vae_tiled_decode or config.transparent_background
            or bool(getattr(config, "references", ())))


def _validate_policy(config: GenerationConfig) -> None:
    """Review-scope refusals, before any manifest read or ML import."""
    if not isinstance(config, GenerationConfig):
        raise ConfigError(FieldError("config", "a GenerationConfig is required"))
    errors: list[FieldError] = []
    if config.steps == 1:
        errors.append(FieldError(
            "steps", "the pinned base schedule produces non-finite values with one step; "
                     "use at least two steps"))
    if config.quant not in (Quant.FP16, Quant.Q8):
        errors.append(FieldError(
            "quant", "the resident loader supports fp16 or reviewed Q8_0 only"))
    if config.quant is Quant.Q8 and config.q8_manifest_path is None:
        errors.append(FieldError("q8_manifest_path", "Q8 requires an explicit --q8-manifest sidecar"))
    if config.quant is Quant.Q8:
        from qwen_engine.adapters.q8_loader import validate_q8_environment
        validate_q8_environment()
    if config.recipe is Recipe.TURBO:
        # Mirror of the config gate (§5.2), so a hand-built or replayed
        # config still dies before any weight read: the sigmas make steps
        # inert, upstream evaluated only CFG 1 with no negative prompt, and
        # no Q8 Turbo artifact exists.
        if config.steps != TURBO_STEPS:
            errors.append(FieldError(
                "steps", f"recipe 'turbo' owns the shipped {TURBO_STEPS}-step "
                         "sigma schedule; steps must match it exactly"))
        if config.cfg_active:
            errors.append(FieldError(
                "true_cfg_scale", "recipe 'turbo' is measured only at CFG 1.0 "
                                  "with no negative prompt"))
        if config.quant is Quant.Q8:
            errors.append(FieldError(
                "quant", "recipe 'turbo' has no reviewed Q8 artifact yet"))
    if config.offload is not OffloadMode.RESIDENT:
        errors.append(FieldError(
            "offload", "the resident loader supports offload=resident only; "
                       "streamed/offload modes are M5"))
    if config.attention_backend not in (AttentionBackend.AUTO, AttentionBackend.SDPA):
        errors.append(FieldError(
            "attention_backend", "this loader supports auto/sdpa only; "
                                  "other backends are benchmark work"))
    for name in ("manifest_path", "models_dir"):
        if getattr(config, name) is None:
            errors.append(FieldError(name, "an actual run requires this path"))
    if config.output_dir is None:
        errors.append(FieldError("output_dir", "an actual run requires an output directory"))
    if config.cache_dir is None:
        errors.append(FieldError(
            "cache_dir", "the encoder cache/scratch requires an explicit cache_dir; "
                         "add --cache-dir cache (created automatically) "
                         "or set cache_dir in the JSON config"))
    references = tuple(getattr(config, "references", ()))
    if references:
        from qwen_engine import reference_store
        if len(references) > REFERENCE_MAX:
            errors.append(FieldError(
                "references",
                f"at most {REFERENCE_MAX} reference images are supported, "
                f"got {len(references)}"))
        if any(not reference_store.is_reference_id(value) for value in references):
            errors.append(FieldError(
                "references", "every entry must be a 64-character lowercase hex "
                              "content id of a stored reference"))
        if getattr(config, "references_dir", None) is None:
            errors.append(FieldError(
                "references_dir",
                "reference ids require the store directory (references_dir)"))
    # Measured fp32-decode envelope (FINDINGS-edit-grid: a Q8 1088x1088
    # reference edit sampled 40/40 and died at vae_decoding; clean with
    # --tiled-decode). The web API already refuses these jobs with
    # tiled_decode_required; the CLI must fail the same way, as a typed
    # config error before any model load — never by silently tiling on.
    # Plain T2I stays out of scope: its fp16 untiled limits are measured
    # separately.
    if (_fp32_decode(config) and not config.vae_tiled_decode
            and config.latent_tokens > 4096):
        errors.append(FieldError(
            "vae_tiled_decode",
            "an fp32-decode run (references or transparent_background) OOMs "
            f"untiled above 4096 latent tokens ({config.latent_tokens} here); "
            "enable --tiled-decode or lower the output size"))
    if errors:
        raise ConfigError(errors)


class Qwen21DiffusersAdapter:
    """One-shot local loader over the reviewed supplied-component bridge.

    ``load`` resolves and verifies the local bundle, checks the pinned runtime
    and the V100 profile, builds the pinned pipeline classes locally (processor,
    transformer, VAE, scheduler; no text encoder), places them as resident FP16
    components, then delegates the five phases to
    :class:`~qwen_engine.adapters.supplied_qwen21.SuppliedQwen21Adapter`.
    """

    def __init__(self, *, trust_unbounded_host: bool = False) -> None:
        if not isinstance(trust_unbounded_host, bool):
            raise ConfigError(
                "trust_unbounded_host must be an explicit boolean operator attestation")
        self._event: threading.Event | None = None
        self._inner: SuppliedQwen21Adapter | None = None
        self._pipeline: Any = None
        self._bundle: LocalReferenceBundle | None = None
        self._torch: Any = None
        self._vae_cls: Any = None
        self._vae_dir: Path | None = None
        self._tiled_decode = False
        self._fp32_decode = False
        self._mlp_chunk_tokens = 0
        self._joint_tokens = None
        self._reference_latents_hook = None
        self._reference_vae_latents: tuple[Any, ...] | None = None
        self._reference_latents_box: list | None = None
        self._mlp_chunked_blocks = 0
        self._trust_unbounded_host = trust_unbounded_host
        self._state = "new"
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        if self._state != "new" or (callback is not None and not callable(callback)):
            raise ConfigError("bind a callable progress callback or None before load")
        self._progress_callback = callback

    def set_cancel_event(self, event: threading.Event | None) -> None:
        if self._state != "new" or (event is not None and not isinstance(event, threading.Event)):
            raise ConfigError("bind a threading.Event or None before load")
        self._event = event

    def load(self, config: GenerationConfig, assets) -> None:
        if self._state != "new" or self._inner is not None:
            raise EngineError("the loader adapter is single-use and already attached")
        _validate_policy(config)
        manifest = load_manifest(config.manifest_path)
        bundle = resolve_local_bundle(manifest, config.models_dir)
        turbo = config.recipe is Recipe.TURBO
        turbo_bundle = None
        if turbo:
            # A separate manifest/root (§3.3), never an implicit "whatever
            # sits next to the base files". The BASE bundle still carries
            # encoder/processor/vae identity and the embedding-cache
            # namespace: the Turbo tree ships none of those (decision: the
            # base encoder is used exactly).
            turbo_manifest_path, turbo_root = turbo_bundle_paths(config.manifest_path)
            turbo_bundle = resolve_local_bundle(
                load_manifest(turbo_manifest_path), turbo_root,
                profile=TURBO_REFERENCE_PROFILE)
        q8_artifact = None
        if config.quant is Quant.Q8:
            from qwen_engine.q8_artifact import resolve_q8_artifact
            q8_artifact = resolve_q8_artifact(config.q8_manifest_path)
        bundle_assets = bundle.to_assets()
        expected_assets = dict(bundle_assets)
        if turbo_bundle is not None:
            # Turbo overrides the dit/scheduler/pipeline roles with its own
            # artifacts; text_encoder/vae/processor stay base.
            expected_assets.update(turbo_bundle.to_assets())
        # The CLI resolver reports paths in the caller's lexical form (e.g. a
        # relative --models-dir); the bundle publishes canonical absolutes.
        # Compare resolved identities, then attach the canonical set.
        supplied_assets = {
            role: tuple(Path(path).resolve() for path in paths)
            for role, paths in dict(assets).items()
        }
        if supplied_assets != expected_assets:
            raise ConfigError(FieldError(
                "assets", "verified asset paths disagree with the pinned local bundle"))

        # CPU-side identity before the CUDA context exists; the post-context
        # observation below must bind to this exact device UUID/index.
        observe_generation_runtime()

        (
            torch,
            pipeline_cls,
            transformer_cls,
            vae_cls,
            scheduler_cls,
            processor_cls,
        ) = _libraries()
        properties = _require_v100(torch)
        attach_cuda(torch)
        if current() is not None:
            current().metadata.update(vae_tile=TILED_DECODE_TILE if config.vae_tiled_decode else None,
                                      vae_stride=TILED_DECODE_STRIDE if config.vae_tiled_decode else None,
                                      vae_precision="fp32" if _fp32_decode(config) else "fp16",
                                      recipe=config.recipe.value)
        device = torch.device("cuda", LOGICAL_DEVICE_INDEX)

        try:
            references = tuple(getattr(config, "references", ()))
            pipeline_kwargs = ({}, {"sample_sigmas": list(TURBO_SAMPLE_SIGMAS)})[turbo]
            if turbo:
                _assert_turbo_scheduler(turbo_bundle.component_dirs["scheduler"])
            scheduler = scheduler_cls.from_pretrained(
                (turbo_bundle if turbo else bundle).component_dirs["scheduler"],
                local_files_only=True)
            processor = processor_cls.from_pretrained(
                bundle.component_dirs["processor"], local_files_only=True)
            if references:
                # Reference residency order (measured on the V100): restoring
                # the VAE inside ``__call__`` while the transformer is
                # resident OOMs (15.05 GiB allocated, +384 MiB requested).
                # So the VAE encodes every reference while it is the ONLY
                # resident component, the tiny latents are kept, the VAE is
                # dropped, and only then does the transformer load. The
                # per-call override (see _install_reference_latents_hook)
                # serves the stored latents during sampling.
                with stage("initial_vae_loading", cuda=True):
                    vae = vae_cls.from_pretrained(
                        bundle.component_dirs["vae"], torch_dtype=torch.float16,
                        local_files_only=True)
                    vae.to(device=device, dtype=torch.float16).eval()
                pipeline = pipeline_cls(scheduler=scheduler, vae=vae,
                                        text_encoder=None, processor=processor,
                                        transformer=None, **pipeline_kwargs)
                self._pipeline = pipeline
                with stage("reference_vae_encode", cuda=True):
                    from PIL import Image
                    self._reference_vae_latents = self._encode_reference_latents(
                        pipeline, load_reference_images(config, Image), torch, device)
                    self._release_vae()
                with stage("transformer_loading", cuda=True):
                    pipeline.transformer = self._load_transformer(
                        transformer_cls, torch, device, bundle, q8_artifact,
                        turbo_bundle, turbo)
            else:
                # Load and place one component at a time: the CPU copy of each is
                # released before the next component's weights are read, so host
                # peak stays near one model rather than the whole stack.
                with stage("transformer_loading", cuda=True):
                    transformer = self._load_transformer(
                        transformer_cls, torch, device, bundle, q8_artifact,
                        turbo_bundle, turbo)
                with stage("initial_vae_loading", cuda=True):
                    vae = vae_cls.from_pretrained(
                        bundle.component_dirs["vae"], torch_dtype=torch.float16,
                        local_files_only=True)
                    vae.to(device=device, dtype=torch.float16).eval()
                pipeline = pipeline_cls(scheduler=scheduler, vae=vae,
                                        text_encoder=None, processor=processor,
                                        transformer=transformer, **pipeline_kwargs)
        except Exception as exc:
            raise EngineError(
                f"local component load refused: {type(exc).__name__}: {exc}") from exc

        runtime = observe_generation_runtime()
        if not runtime.matches_cuda_device(
                uuid=str(properties.uuid), logical_device_index=LOGICAL_DEVICE_INDEX):
            raise EngineError(
                "generation runtime identity is not bound to the reviewed CUDA device")
        if current() is not None:
            artifact_identity = bundle.fingerprints.bundle
            if q8_artifact is not None:
                from qwen_engine.adapters.q8_loader import artifact_set_identity
                artifact_identity = artifact_set_identity(artifact_identity, q8_artifact.identity_sha256)
            current().metadata.update(runtime_identity_sha256=runtime.sha256,
                                      artifact_set_sha256=artifact_identity)
            if turbo:
                current().metadata["turbo_artifact_set_sha256"] = turbo_bundle.fingerprints.bundle
                if tuple(getattr(config, "references", ())):
                    # Base-encoder conditioning combined with Turbo
                    # denoising has never been evaluated; the run is
                    # recorded as unmeasured, never silently assumed
                    # (§3.3).
                    current().metadata["turbo_reference_run_unmeasured"] = True

        cache_root = Path(config.cache_dir)
        scratch = cache_root / "encoder-scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        encoder = ProductionEncoder(
            manifest_path=config.manifest_path,
            models_dir=config.models_dir,
            scratch_dir=scratch,
            headroom_bytes=ENCODER_HEADROOM_BYTES,
            timeout_s=ENCODER_TIMEOUT_S,
            cache=EmbeddingCache(cache_root / "embeds"),
            attested_unbounded_host=self._trust_unbounded_host,
        )

        # Automatic KV-cache policy for reference edits (V100, 1K, 1 ref,
        # 40 steps, measured 2026-10-25): FP16 transformer + KV ON OOMs at
        # the FIRST denoise step; KV OFF fits (denoise 95 s). Q8 + KV ON
        # fits (58 s). Turbo is an FP16 transformer and fits KV-off (19 s).
        # So references keep KV only with Q8. T2I stays KV-ON, unchanged.
        # NOT measured: 2-reference FP16 KV-off residency — deliberately
        # left ALLOWED (the batch5 matrix will measure it); nothing here
        # refuses it. The choice lands in calibration_facts()["use_kv_cache"]
        # (attached adapter) and in the run metadata below.
        use_kv_cache = not references or q8_artifact is not None
        if current() is not None:
            current().metadata["kv_cache_policy"] = (
                "t2i-on" if not references else
                "q8-on" if use_kv_cache else "reference-fp16-off")
        inner = SuppliedQwen21Adapter(
            pipeline=pipeline,
            encoder=encoder,
            manifest=manifest,
            bundle=bundle,
            encoder_dtype=ENCODER_DTYPE,
            execution_dtype="fp16",
            use_kv_cache=use_kv_cache,
        )
        inner.set_cancel_event(self._event)
        inner.set_progress_callback(self._progress_callback)
        inner.load(config, bundle_assets)
        if config.latent_tokens > MLP_CHUNK_ABOVE_TOKENS:
            self._mlp_chunked_blocks = _apply_mlp_chunking(
                torch, pipeline.transformer, MLP_CHUNK_TOKENS)
            self._mlp_chunk_tokens = MLP_CHUNK_TOKENS
        self._pipeline = pipeline
        self._bundle = bundle
        self._torch = torch
        self._vae_cls = vae_cls
        self._vae_dir = bundle.component_dirs["vae"]
        self._tiled_decode = config.vae_tiled_decode
        self._fp32_decode = _fp32_decode(config)
        self._inner = inner
        self._state = "loaded"

    def _attached(self) -> SuppliedQwen21Adapter:
        if self._inner is None:
            raise EngineError("loader adapter is not attached; call load() first")
        return self._inner

    def encode(self, config: GenerationConfig):
        result = self._attached().encode(config)
        facts = self._attached().calibration_facts()
        references = tuple(getattr(config, "references", ()))
        if references:
            # Prompt-free joint-sequence facts (design §4): S_prompt + the
            # packed reference tokens + target tokens. This is the value the
            # MLP chunk decision re-evaluates and the measurement matrix
            # records; it changes nothing for text-to-image runs.
            self._joint_tokens = (facts["prompt_tokens"]
                                  + self._joint_reference_tokens(config)
                                  + config.latent_tokens)
        if current() is not None:
            for key in ("embedding_cache_hit", "prompt_tokens",
                        "negative_prompt_tokens", "references"):
                current().metadata[key] = facts[key]
            if references:
                current().metadata["joint_tokens"] = self._joint_tokens
                # Noise domain separation: the derived sampler seed (never
                # equal to config.seed) belongs in the run metadata next to
                # the joint-token facts; the supplied adapter also seeds the
                # generator with it in denoise().
                for key in ("noise_seed", "noise_seed_derivation"):
                    if facts.get(key) is not None:
                        current().metadata[key] = facts[key]
        return result

    def calibration_facts(self):
        """Delegate the exact conditioning facts to the attached adapter;
        add the measured joint token count once references were encoded."""
        facts = self._attached().calibration_facts()
        if self._joint_tokens is not None:
            facts = {**facts, "joint_tokens": self._joint_tokens}
        return facts

    def denoise(self, config: GenerationConfig, conditioning):
        references = bool(getattr(config, "references", ()))
        with stage("initial_vae_cleanup", cuda=True):
            self._release_vae()
        if references:
            self._apply_joint_chunking()
        with stage("denoising", cuda=True):
            if references:
                self._install_reference_latents_hook()
            try:
                latents = self._attached().denoise(config, conditioning)
                if references and self._reference_latents_box:
                    raise EngineError(
                        "the pipeline consumed "
                        f"{len(self._reference_vae_latents) - len(self._reference_latents_box)}"
                        f" of {len(self._reference_vae_latents)} prepared "
                        "reference VAE latents")
            finally:
                if references:
                    self._remove_reference_latents_hook()
        with stage("transformer_cleanup", cuda=True):
            self._release_denoiser()
        return latents

    def _joint_reference_tokens(self, config: GenerationConfig) -> int:
        """Tokens the references add to the denoise sequence, measured off the
        hash-verified decoded sizes through the pinned resize rule (lazy
        diffusers import: this only runs on an actual reference run)."""
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
        dimensions = [
            tuple(int(value) for value in calculate_dimensions(
                REFERENCE_OUTPUT_RESOLUTION * REFERENCE_OUTPUT_RESOLUTION, image.size[0] / image.size[1])[:2])
            for image in self._attached().reference_images(config)]
        return _reference_token_total(dimensions)

    def _apply_joint_chunking(self) -> None:
        """Reference runs decide MLP chunking on the JOINT sequence (design §4
        correctness gap): the load-time target-only decision cannot know
        S_prompt or the reference tokens. Re-evaluated after encode, before the
        first step; an already-chunked transformer (large target) is kept.
        Chunking is token-wise exact, so the decision changes peak memory and
        kernel count, never the result."""
        if self._joint_tokens is None:
            raise EngineError("reference generation must be encoded before denoise")
        if self._joint_tokens > MLP_CHUNK_ABOVE_TOKENS and not self._mlp_chunk_tokens:
            self._mlp_chunked_blocks = _apply_mlp_chunking(
                self._torch, self._pipeline.transformer, MLP_CHUNK_TOKENS)
            self._mlp_chunk_tokens = MLP_CHUNK_TOKENS

    def _load_transformer(self, transformer_cls, torch, device, bundle,
                          q8_artifact, turbo_bundle, turbo):
        """One transformer, one placement path — shared by both load orders."""
        if q8_artifact is None:
            transformer = transformer_cls.from_pretrained(
                (turbo_bundle if turbo else bundle).component_dirs["dit"],
                torch_dtype=torch.float16, local_files_only=True)
            transformer.to(device=device, dtype=torch.float16).eval()
        else:
            from qwen_engine.adapters.q8_loader import load_q8_transformer
            transformer = load_q8_transformer(transformer_cls, bundle,
                                              q8_artifact, torch, device)
        return transformer

    def _encode_reference_latents(self, pipeline, images, torch, device):
        """Encode every reference through the PINNED pipeline's own
        preprocessing and VAE encode, while the VAE is the only resident
        component.

        Mirrors ``QwenImage21Pipeline.__call__`` step 1 and the per-image
        VAE call of ``prepare_latents`` exactly — RGBA conversion, the
        ``calculate_dimensions`` resize target at
        :data:`REFERENCE_OUTPUT_RESOLUTION` (the pinned ``output_resolution``
        default, which the bridge never overrides),
        ``image_processor.preprocess(...).unsqueeze(2)``, then the pipeline's
        own ``_encode_vae_image`` — by calling those pinned objects, never a
        re-implementation. ``argmax`` sampling ignores the generator, so the
        latents are deterministic and match what an in-call VAE encode would
        have produced; the reference identity itself is bound by the same
        store-id rechecks ``reference_images`` performs at generation time.
        """
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
            calculate_dimensions)
        if tuple(inspect.signature(pipeline._encode_vae_image).parameters) \
                != _ENCODE_VAE_IMAGE_PARAMETERS:
            raise EngineError("pinned pipeline _encode_vae_image signature has "
                              "changed; reference latents cannot be precomputed")
        # The pinned ``__call__`` carries @torch.no_grad(); ``_encode_vae_image``
        # itself does NOT, and a direct call outside that context traces the
        # encoder: an fp16 untiled 1024² reference keeps ~14 GiB of autograd
        # activations on the V100 and the SECOND reference OOMs (a grad-free
        # VAE-alone encode peaks at 2.72 GiB and frees — measured on the
        # card). argmax sampling ignores the generator, so dropping the graph
        # does not change the latents the pinned call would have produced.
        latents = []
        for position, image in enumerate(images, start=1):
            if hasattr(image, "mode") and image.mode != "RGBA":
                image = image.convert("RGBA")
            image_width, image_height = image.size
            input_width, input_height, _ = calculate_dimensions(
                REFERENCE_OUTPUT_RESOLUTION * REFERENCE_OUTPUT_RESOLUTION,
                image_width / image_height)
            vae_image = pipeline.image_processor.preprocess(
                image, width=input_width, height=input_height).unsqueeze(2)
            vae_image = vae_image.to(device=device, dtype=torch.float16)
            with torch.no_grad():
                encoded = pipeline._encode_vae_image(vae_image, None)
            if tuple(encoded.shape[:3]) != (1, pipeline.latent_channels, 1):
                raise EngineError(
                    f"reference {position} produced an unexpected VAE latent "
                    f"shape {tuple(encoded.shape)}")
            # Store the INPUT (pixel) geometry beside the latent: at ``__call__``
            # the override witnesses the pipeline's preprocessed pixel tensor,
            # never a latent — comparing pixel and latent shapes (which never
            # match) would reject every legitimate run.
            latents.append((tuple(vae_image.shape),
                            encoded.detach().to(device="cpu",
                                                dtype=torch.float16)))
        return tuple(latents)

    def _install_reference_latents_hook(self) -> None:
        """Serve the load-phase reference VAE latents during ``__call__``.

        The VAE left the card before the transformer was even loaded
        (:meth:`_encode_reference_latents`); restoring it mid-denoise is the
        measured OOM this replaces. This is a per-call instance override of
        the narrowest method the pinned ``prepare_latents`` invokes per
        image — ``_encode_vae_image`` — returning the stored latents in
        order. Each call's input tensor must still describe the same image
        geometry, requesting more latents than were prepared is a refusal,
        and the caller's ``finally`` (:meth:`_remove_reference_latents_hook`)
        removes the override so nothing survives into memory measurement;
        unconsumed latents are reported by the caller before publication.
        """
        pipeline = self._pipeline
        if pipeline is None or not callable(getattr(pipeline, "_encode_vae_image", None)):
            raise EngineError("reference generation requires the pinned _encode_vae_image")
        if "_encode_vae_image" in vars(pipeline):
            raise EngineError("a _encode_vae_image override is already installed")
        if tuple(inspect.signature(pipeline._encode_vae_image).parameters) \
                != _ENCODE_VAE_IMAGE_PARAMETERS:
            raise EngineError("pinned pipeline _encode_vae_image signature has "
                              "changed; cached reference latents cannot be "
                              "injected blindly")
        if not self._reference_vae_latents:
            raise EngineError("reference generation has no VAE latents prepared at load")
        pending = list(self._reference_vae_latents)
        self._reference_latents_box = pending

        def encode_vae_image(image, generator):
            # The pinned call hands over the preprocessed PIXEL tensor
            # (1, 4, 1, H, W) and the run generator; the stored latents were
            # sampled argmax (generator-independent), so the only witness is
            # geometry: the incoming shape must equal the PIXEL shape the
            # stored latent was computed from at load (a latent's own, smaller
            # shape can never match the pipeline's pixel input).
            if not pending:
                raise EngineError(
                    "the pipeline requested more reference VAE encodes than "
                    "the prepared latents cover")
            input_shape, latent = pending.pop(0)
            if tuple(image.shape) != input_shape:
                raise EngineError(
                    "the pipeline's reference preprocessing disagrees with the "
                    "latents prepared at load")
            return latent.to(device=image.device, dtype=image.dtype)

        pipeline._encode_vae_image = encode_vae_image
        self._reference_latents_hook = pipeline

    def _remove_reference_latents_hook(self) -> None:
        pipeline, self._reference_latents_hook = self._reference_latents_hook, None
        self._reference_latents_box = None
        if pipeline is not None and "_encode_vae_image" in vars(pipeline):
            del pipeline._encode_vae_image

    def decode(self, latents):
        with stage("vae_loading", cuda=True):
            self._restore_vae()
        return self._attached().decode(latents)

    def _release_vae(self) -> None:
        """Take the VAE off the card while the transformer denoises.

        Phase-exclusive residency: denoise needs only the transformer, decode
        needs only the VAE, so the two never have to be resident together.
        Dropping the object (rather than moving it to CPU) also keeps
        ``_execution_device`` unambiguous — a CPU-placed VAE would otherwise be
        the first module the pipeline inspects. The VAE is reloaded from the
        same verified local directory before decode.
        """
        pipeline = self._pipeline
        if pipeline is None or getattr(pipeline, "vae", None) is None:
            return
        pipeline.vae = None
        gc.collect()

    def _restore_vae(self, *, reference: bool = False) -> None:
        pipeline = self._pipeline
        if pipeline is None or self._vae_cls is None or self._vae_dir is None:
            return
        if getattr(pipeline, "vae", None) is not None:
            return
        torch = self._torch
        # Tiled decode runs the VAE in fp32: at 2048² the fp16 tiled path with
        # the library's default 256 px tiles produced non-finite output
        # (measured; the same latents decode finite in fp32), and the
        # transformer is already released, so the wider activations fit
        # (13.4 GiB peak with 1024 px tiles). Untiled decode stays fp16 — it
        # fits the 1024² envelope, and fp16/fp32 differ by at most 1/255 there
        # (2/255 with 2K tiling), so precision is not buying visible quality.
        # Transparent output also decodes in fp32: at 1024² the fp16 decode of
        # RGBA latents was non-finite for all three measured prompts, while the
        # same run with fp32 decode produced a finite image with a real alpha
        # channel.
        dtype = (torch.float32 if self._fp32_decode else torch.float16)
        vae = self._vae_cls.from_pretrained(
            self._vae_dir, torch_dtype=dtype, local_files_only=True)
        vae.to(device=torch.device("cuda", LOGICAL_DEVICE_INDEX),
               dtype=dtype).eval()
        if self._tiled_decode:
            vae.enable_tiling(
                tile_sample_min_height=TILED_DECODE_TILE,
                tile_sample_min_width=TILED_DECODE_TILE,
                tile_sample_stride_height=TILED_DECODE_STRIDE,
                tile_sample_stride_width=TILED_DECODE_STRIDE,
            )
        pipeline.vae = vae

    def _release_denoiser(self) -> None:
        """Drop the transformer's VRAM before the VAE decode (single-use loader).

        The bridge never touches the transformer again after denoise (decode
        uses the VAE, save uses the image), and this adapter is single-use, so
        releasing the reference is safe. The freed blocks stay in this
        process's own allocator cache and are reused by the decode allocation;
        no global allocator flush is performed and nothing outside this process
        is affected.

        ``pipeline.transformer = None`` is not enough on its own: diffusers
        attaches a per-module hook registry that refers back to the module, so
        the transformer survives as an unreachable reference cycle (measured:
        13.9 GiB still allocated after the assignment, 0.64 GiB after
        ``gc.collect()`` breaks the cycle). The collector runs only after the
        reference is dropped, so it cannot collect anything still in use.

        Why it matters (measured on the V100): transformer + VAE resident is
        ~14.8 GiB of 16 GiB, which leaves too little for an untiled 1024²
        decode. Untiled decode at 1024² is also measurably *better* than the
        tiled path: tiled decode leaves faint vertical dashed streaks in flat
        regions (max channel difference 52/255, mean 1.24 against the resident
        decode of the same latents), while the resident decode is clean.
        ``--tiled-decode`` therefore stays available as an explicit
        memory-first knob, not as the default.
        """
        pipeline = self._pipeline
        if pipeline is None or getattr(pipeline, "transformer", None) is None:
            return
        pipeline.transformer = None
        gc.collect()

    def save(self, image, config: GenerationConfig):
        return self._attached().save(image, config)

    def close(self) -> None:
        """Drop owned references; never delete outputs, flush allocators or
        touch another process's devices. Idempotent after partial load."""
        inner, self._inner = self._inner, None
        self._pipeline = None
        self._bundle = None
        self._state = "closed"
        self._progress_callback = None
        self._joint_tokens = None
        self._reference_latents_hook = None
        self._reference_vae_latents = None
        self._reference_latents_box = None
        if inner is not None:
            inner.close()
