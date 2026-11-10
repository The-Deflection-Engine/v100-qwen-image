"""ML-free boundaries and review-scope refusals for the default loader.

No torch/diffusers/transformers import, no model weights, no GPU and no
network occur in this module: the loader's heavy imports live inside
``load`` after its policy and manifest gates.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from qwen_engine import cli
from qwen_engine.config import ConfigError, GenerationConfig
from qwen_engine.errors import EngineError
from qwen_engine.manifest import ManifestError

HEAVY = ("torch", "diffusers", "transformers")


def _config(tmp_path, **overrides):
    values = dict(
        prompt="a red cube",
        output_dir=str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        manifest_path=str(tmp_path / "missing-manifest.json"),
        models_dir=str(tmp_path / "models"),
    )
    values.update(overrides)
    return GenerationConfig(**values)


def test_module_import_stays_ml_free_in_a_fresh_interpreter():
    code = (
        "import sys\n"
        "import qwen_engine.adapters.qwen21_diffusers as mod\n"
        "heavy = sorted(set(sys.modules) & {'torch', 'diffusers', 'transformers'})\n"
        "assert not heavy, heavy\n"
        "assert mod.Qwen21DiffusersAdapter.__module__ == "
        "'qwen_engine.adapters.qwen21_diffusers'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True, text=True, env=dict(os.environ))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_factory_returns_the_default_loader_without_heavy_imports():
    adapter = cli._make_adapter()
    assert type(adapter).__module__ == "qwen_engine.adapters.qwen21_diffusers"
    assert not set(HEAVY) & set(sys.modules)


@pytest.mark.parametrize("field,value", [
    ("quant", "q4"),
    ("offload", "streamed"),
    ("offload", "offload"),
    ("attention_backend", "eager"),
    ("attention_backend", "efficient"),
])
def test_unreviewed_execution_policy_refuses_before_manifest_or_ml(
        tmp_path, field, value):
    adapter = cli._make_adapter()
    with pytest.raises(ConfigError) as caught:
        adapter.load(_config(tmp_path, **{field: value}), {})
    assert field in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


@pytest.mark.parametrize("missing", [
    "output_dir", "cache_dir", "manifest_path", "models_dir",
])
def test_missing_runtime_paths_refuse_before_manifest_or_ml(tmp_path, missing):
    adapter = cli._make_adapter()
    values = {missing: None}
    with pytest.raises(ConfigError) as caught:
        adapter.load(_config(tmp_path, **values), {})
    assert missing in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


def test_missing_manifest_refuses_before_any_ml_import(tmp_path):
    adapter = cli._make_adapter()
    with pytest.raises(ManifestError):
        adapter.load(_config(tmp_path), {})
    assert not set(HEAVY) & set(sys.modules)


def test_single_step_base_schedule_refuses_before_assets_or_model_loading(tmp_path):
    adapter = cli._make_adapter()
    with pytest.raises(ConfigError) as caught:
        adapter.load(_config(tmp_path, steps=1), {})
    assert "steps" in caught.value.fields
    assert "non-finite" in str(caught.value)
    assert not set(HEAVY) & set(sys.modules)


def test_q8_requires_explicit_sidecar_before_assets_or_ml(tmp_path):
    with pytest.raises(ConfigError) as caught:
        cli._make_adapter().load(_config(tmp_path, quant="q8"), {})
    assert "q8_manifest_path" in caught.value.fields
    assert not set(HEAVY) & set(sys.modules)


def test_q8_remote_kernel_mode_refuses_before_ml_import(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFUSERS_GGUF_CUDA_KERNELS", "true")
    with pytest.raises(EngineError, match="CUDA_KERNELS=false"):
        cli._make_adapter().load(_config(tmp_path, quant="q8", q8_manifest_path=str(tmp_path / "q8.json")), {})
    assert not set(HEAVY) & set(sys.modules)


def test_phase_calls_before_load_are_refused():
    adapter = cli._make_adapter()
    with pytest.raises(EngineError):
        adapter.encode(GenerationConfig())
    with pytest.raises(EngineError):
        adapter.denoise(GenerationConfig(), None)
    with pytest.raises(EngineError):
        adapter.decode(None)
    with pytest.raises(EngineError):
        adapter.save(None, GenerationConfig())


def test_cancel_event_binding_and_idempotent_close():
    import threading

    adapter = cli._make_adapter()
    with pytest.raises(ConfigError):
        adapter.set_cancel_event("not-an-event")
    adapter.set_cancel_event(threading.Event())
    adapter.set_cancel_event(None)  # re-binding before load is allowed
    adapter.close()
    adapter.close()  # idempotent after partial/never load


def test_second_load_is_refused_after_close():
    adapter = cli._make_adapter()
    adapter.close()
    with pytest.raises(EngineError):
        adapter.load(GenerationConfig(), {})


def test_config_replacement_keeps_policy_refusal_reachable(tmp_path):
    """replace() on a valid config still yields the reviewed-profile errors."""
    config = _config(tmp_path, quant="q8")
    adapter = cli._make_adapter()
    with pytest.raises(ConfigError):
        adapter.load(replace(config, offload="streamed"), {})


# -- operator host-scope attestation (off by default) -----------------------


def test_loader_defaults_to_no_host_scope_attestation():
    adapter = cli._make_adapter()
    assert adapter._trust_unbounded_host is False


def test_loader_accepts_the_explicit_operator_attestation():
    adapter = cli._make_adapter(trust_unbounded_host=True)
    assert adapter._trust_unbounded_host is True


@pytest.mark.parametrize("value", ["yes", 1, 0, None])
def test_loader_refuses_a_non_boolean_attestation(value):
    with pytest.raises(ConfigError, match="explicit boolean"):
        cli._make_adapter(trust_unbounded_host=value)


def test_cli_exposes_the_attestation_flag_by_default_off():
    parser = cli.build_parser()
    assert parser.parse_args([]).attest_unbounded_host is False
    assert parser.parse_args(["--attest-unbounded-host"]).attest_unbounded_host is True


def test_cli_notice_names_the_attestation_when_used(tmp_path, capsys, monkeypatch):
    """The run transcript must show the attestation before generation; the
    adapter factory is stubbed so no model is touched."""
    import json

    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({"schema": "qwen_engine.manifest.v1",
                                 "artifacts": []}))
    (tmp_path / "models").mkdir()
    captured = {}

    class FakeAdapter:
        def set_cancel_event(self, event):
            pass

    def factory(**kwargs):
        captured.update(kwargs)
        raise EngineError("stop after the notice")

    monkeypatch.setattr(cli, "_make_adapter", factory)
    monkeypatch.setattr(cli, "_verify_assets", lambda config: type(
        "R", (), {"resolved": {}})())
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "prompt": "a red cube", "manifest_path": str(mpath),
        "models_dir": str(tmp_path / "models"),
        "output_dir": str(tmp_path / "out")}))
    assert cli.main(["--config", str(cfg), "--run",
                     "--attest-unbounded-host"]) == 1
    err = capsys.readouterr().err
    assert "attesting that this process tree has no cgroup memory limit" in err
    assert captured == {"trust_unbounded_host": True}


# -- transformer release before the VAE decode ------------------------------


class _RecordingInner:
    def __init__(self, order):
        self.order = order

    def denoise(self, config, conditioning):
        self.order.append("denoise")
        return ["latents"]

    def decode(self, latents):
        self.order.append("decode")
        return "image"


def test_denoise_releases_the_transformer_before_decode():
    """Single-use loader: the transformer is dead weight during decode, and
    releasing it is what lets the untiled (artifact-free) decode fit."""
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    order = []
    adapter._inner = _RecordingInner(order)
    transformer = SimpleNamespace()
    adapter._pipeline = SimpleNamespace(transformer=transformer)

    latents = adapter.denoise(GenerationConfig(), None)
    assert adapter._pipeline.transformer is None
    adapter.decode(latents)
    assert order == ["denoise", "decode"]


def test_release_is_idempotent_without_a_pipeline_or_transformer():
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    adapter._release_denoiser()  # no pipeline at all
    adapter._pipeline = SimpleNamespace(transformer=None)
    adapter._release_denoiser()  # already released
    adapter._pipeline = SimpleNamespace()
    adapter._release_denoiser()  # attribute absent


# -- phase-exclusive residency (VAE off the card during denoise) ------------


class _FakeVae:
    def __init__(self, name):
        self.name = name
        self.tiling = False

    def to(self, **kwargs):
        self.moved = kwargs
        return self

    def eval(self):
        return self

    def enable_tiling(self, **kwargs):
        self.tiling = True
        self.tiling_args = kwargs


class _FakeVaeClass:
    def __init__(self, order):
        self.order = order
        self.calls = []

    def from_pretrained(self, path, **kwargs):
        self.order.append("reload-vae")
        self.calls.append((path, kwargs))
        return _FakeVae(path)


class _VaeInner:
    def __init__(self, order, pipeline):
        self.order = order
        self.pipeline = pipeline

    def denoise(self, config, conditioning):
        self.order.append(("denoise", self.pipeline.vae))
        return ["latents"]

    def decode(self, latents):
        self.order.append(("decode", self.pipeline.vae))
        return "image"


def test_denoise_takes_the_vae_off_the_card_and_decode_restores_it():
    from types import SimpleNamespace

    order = []
    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(transformer=SimpleNamespace(), vae=_FakeVae("resident"))
    adapter._pipeline = pipeline
    adapter._inner = _VaeInner(order, pipeline)
    adapter._vae_cls = _FakeVaeClass(order)
    adapter._vae_dir = "/models/vae"
    adapter._torch = SimpleNamespace(float16="fp16", float32="fp32",
                                     device=lambda *a: ("cuda", a[1]))
    adapter._tiled_decode = True
    adapter._fp32_decode = True

    latents = adapter.denoise(GenerationConfig(), None)
    assert order[0] == ("denoise", None)        # VAE already gone
    assert pipeline.transformer is None         # transformer gone afterwards
    adapter.decode(latents)
    assert order[1] == "reload-vae"
    assert isinstance(order[2][1], _FakeVae)    # decoded with the restored VAE
    assert order[2][1].tiling is True
    assert order[2][1].tiling_args == {
        "tile_sample_min_height": 1024, "tile_sample_min_width": 1024,
        "tile_sample_stride_height": 768, "tile_sample_stride_width": 768}
    path, kwargs = adapter._vae_cls.calls[0]
    assert path == "/models/vae"
    # Tiled decode restores the VAE in fp32: the fp16 tiled path is
    # non-finite at 2048², and the transformer is gone by decode time.
    assert kwargs == {"torch_dtype": "fp32", "local_files_only": True}
    assert order[2][1].moved == {"device": ("cuda", 0), "dtype": "fp32"}


def test_untiled_decode_restores_the_vae_in_fp16():
    from types import SimpleNamespace

    order = []
    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(vae=None)
    adapter._pipeline = pipeline
    adapter._vae_cls = _FakeVaeClass(order)
    adapter._vae_dir = "/models/vae"
    adapter._torch = SimpleNamespace(float16="fp16", float32="fp32",
                                     device=lambda *a: ("cuda", a[1]))
    adapter._tiled_decode = False
    adapter._restore_vae()
    path, kwargs = adapter._vae_cls.calls[0]
    assert path == "/models/vae"
    assert kwargs == {"torch_dtype": "fp16", "local_files_only": True}
    assert pipeline.vae.moved == {"device": ("cuda", 0), "dtype": "fp16"}
    assert pipeline.vae.tiling is False


def test_transparent_output_decodes_untiled_in_fp32(tmp_path):
    # Measured on the V100: fp16 decode of transparent (RGBA) latents was
    # non-finite at 1024², fp32 decode was finite with a real alpha channel.
    from types import SimpleNamespace

    from qwen_engine.adapters.qwen21_diffusers import _fp32_decode

    assert _fp32_decode(GenerationConfig()) is False
    assert _fp32_decode(GenerationConfig(vae_tiled_decode=True)) is True
    assert _fp32_decode(GenerationConfig(transparent_background=True,
                                         output_format="png")) is True
    # Reference edits (batch4 on the V100): the Q8 2-reference run sampled
    # cleanly and died at vae_decoding with af37930's exact non-finite
    # attestation even though the output was opaque — any non-empty
    # references forces fp32 decode; an empty tuple is not a reference run.
    assert _fp32_decode(GenerationConfig(references=())) is False
    assert _fp32_decode(_reference_config(tmp_path)) is True
    order = []
    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(vae=None)
    adapter._pipeline = pipeline
    adapter._vae_cls = _FakeVaeClass(order)
    adapter._vae_dir = "/models/vae"
    adapter._torch = SimpleNamespace(float16="fp16", float32="fp32",
                                     device=lambda *a: ("cuda", a[1]))
    adapter._tiled_decode = False
    adapter._fp32_decode = True
    adapter._restore_vae()
    _, kwargs = adapter._vae_cls.calls[0]
    assert kwargs == {"torch_dtype": "fp32", "local_files_only": True}
    assert pipeline.vae.tiling is False


def test_vae_release_is_idempotent_and_restore_is_skipped_when_present():
    from types import SimpleNamespace

    adapter = cli._make_adapter()
    pipeline = SimpleNamespace(vae=_FakeVae("resident"))
    adapter._pipeline = pipeline
    adapter._vae_cls = None       # no loader available: restore must no-op
    adapter._release_vae()
    assert pipeline.vae is None
    adapter._release_vae()        # already released
    adapter._restore_vae()        # nothing to restore with
    assert pipeline.vae is None


# -- high-token MLP chunking (exact, token-wise) ----------------------------


class _Rows:
    """Minimal ML-free stand-in for a (1, tokens, width) tensor."""

    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def shape(self):
        return (1, len(self.rows), 64)

    def __getitem__(self, item):
        batch, rest = item
        assert batch == slice(None)
        return _Rows(self.rows[rest])


class _RecordingMlp:
    def __init__(self):
        self.calls = []

    def forward(self, hidden):
        self.calls.append(len(hidden.rows))
        return _Rows([row + 1 for row in hidden.rows])


def _chunking_fixture():
    from types import SimpleNamespace
    from qwen_engine.adapters import qwen21_diffusers as qd

    first, second = _RecordingMlp(), _RecordingMlp()
    transformer = SimpleNamespace(transformer_blocks=[
        SimpleNamespace(img_mlp=first), SimpleNamespace(img_mlp=second)])
    fake_torch = SimpleNamespace(
        cat=lambda parts, dim: _Rows(row for part in parts for row in part.rows))
    return qd, transformer, fake_torch, first, second


def test_high_token_denoise_chunks_the_swiglu_mlp_exactly():
    qd, transformer, fake_torch, first, second = _chunking_fixture()
    assert qd._apply_mlp_chunking(fake_torch, transformer, 2048) == 2
    out = first.forward(_Rows(range(5000)))
    assert first.calls == [2048, 2048, 904]
    assert out.rows == [row + 1 for row in range(5000)]
    assert second.calls == []          # wrapped, but not called by this test
    first.forward(_Rows([1, 2]))       # short sequence: one unchunked call
    assert first.calls[-1] == 2


def test_mlp_chunking_is_idempotent_and_refuses_an_unknown_layout():
    qd, transformer, fake_torch, first, _ = _chunking_fixture()
    assert qd._apply_mlp_chunking(fake_torch, transformer, 2048) == 2
    assert qd._apply_mlp_chunking(fake_torch, transformer, 512) == 0
    first.forward(_Rows(range(3000)))
    assert first.calls == [2048, 952]   # original chunk retained, never doubled
    from types import SimpleNamespace
    with pytest.raises(EngineError, match="chunk"):
        qd._apply_mlp_chunking(fake_torch, SimpleNamespace(transformer_blocks=[
            SimpleNamespace()]), 2048)


# -- reference-image orchestration (edit support, part B) ---------------------


def _reference_capable() -> bool:
    try:
        GenerationConfig(references=("a" * 64,), references_dir="/store")
        return True
    except Exception:
        return False


needs_references = pytest.mark.skipif(
    not _reference_capable(), reason="config references patch not applied yet")


def _reference_config(tmp_path, **overrides):
    values = dict(references=("a" * 64,), references_dir=str(tmp_path / "store"))
    values.update(overrides)
    return _config(tmp_path, **values)


def test_reference_token_total_counts_one_token_per_32px_square():
    from qwen_engine.adapters import qwen21_diffusers as qd
    assert qd._reference_token_total([]) == 0
    assert qd._reference_token_total([(1024, 1024)]) == 1024
    assert qd._reference_token_total([(32, 32), (64, 32)]) == 3


@needs_references
def test_reference_policy_refuses_before_any_ml(tmp_path):
    from qwen_engine.adapters import qwen21_diffusers as qd
    assert qd.REFERENCE_MAX == 10
    cases = [
        (dict(references=tuple(f"{i:064x}" for i in range(11))), "references"),
        (dict(references=("A" * 64,), references_dir=str(tmp_path)), "references"),
        (dict(references=("a" * 64,), references_dir=None), "references_dir"),
    ]
    for changes, field in cases:
        with pytest.raises(ConfigError) as excinfo:
            qd._validate_policy(_reference_config(tmp_path, **changes))
        assert field in excinfo.value.fields
    # A within-cap, hex-clean, paired run is not refused by the reference gate.
    try:
        qd._validate_policy(_reference_config(tmp_path))
    except ConfigError as exc:
        assert "references" not in exc.fields and "references_dir" not in exc.fields


@needs_references
def test_fp32_decode_refuses_untiled_runs_above_the_token_ceiling(tmp_path):
    """Measured envelope: an fp32 decode (references or transparent) above
    4096 latent tokens OOMs untiled — the Q8 1088x1088 edit died at
    vae_decoding after 40/40 steps, clean with --tiled-decode. Refuse as a
    typed config error naming the flag, exactly like the web API's 422."""
    from qwen_engine.adapters import qwen21_diffusers as qd
    big = dict(height=1088, width=1088)  # 4624 latent tokens
    cases = [
        _reference_config(tmp_path, **big),
        _config(tmp_path, transparent_background=True, output_format="png", **big),
    ]
    for config in cases:
        with pytest.raises(ConfigError) as excinfo:
            qd._validate_policy(config)
        assert "vae_tiled_decode" in excinfo.value.fields
        assert "--tiled-decode" in str(excinfo.value)
    # Tiling satisfies the gate; 1024² is exactly 4096 tokens and stays free.
    qd._validate_policy(_reference_config(tmp_path, **big, vae_tiled_decode=True))
    qd._validate_policy(_reference_config(tmp_path))
    qd._validate_policy(_config(tmp_path, transparent_background=True, output_format="png",
                                vae_tiled_decode=True))
    # Plain T2I (fp16 untiled decode) is untouched by this gate.
    try:
        qd._validate_policy(_config(tmp_path, **big))
    except ConfigError as exc:
        assert "vae_tiled_decode" not in exc.fields


def test_joint_chunking_is_decided_after_encode_once(monkeypatch):
    from types import SimpleNamespace
    from qwen_engine.adapters import qwen21_diffusers as qd
    adapter = cli._make_adapter()
    with pytest.raises(EngineError):
        adapter._apply_joint_chunking()  # denoise before encode must not guess
    applied = []
    monkeypatch.setattr(qd, "_apply_mlp_chunking",
                        lambda torch, transformer, tokens: applied.append(tokens) or 60)
    adapter._torch, adapter._pipeline = object(), SimpleNamespace(transformer=object())
    adapter._joint_tokens = qd.MLP_CHUNK_ABOVE_TOKENS
    adapter._apply_joint_chunking()
    assert applied == []  # exactly at the threshold is not above it
    adapter._joint_tokens = qd.MLP_CHUNK_ABOVE_TOKENS + 1
    adapter._apply_joint_chunking()
    assert applied == [qd.MLP_CHUNK_TOKENS]
    assert adapter._mlp_chunk_tokens == qd.MLP_CHUNK_TOKENS
    assert adapter._mlp_chunked_blocks == 60
    applied.clear()
    adapter._apply_joint_chunking()  # already chunked: never re-wrapped
    assert applied == []


_VAE_PIXEL_SHAPE = (1, 4, 1, 1024, 1024)  # pinned preprocessed reference input


class _Latent:
    """Minimal tensor witness: shape/dtype/device plus detach/to only."""
    def __init__(self, tag, shape=(1, 64, 1, 2, 2), device="cpu", dtype="fp16"):
        self.tag, self.shape, self.device, self.dtype = tag, tuple(shape), device, dtype
    def detach(self):
        return self
    def to(self, device=None, dtype=None):
        return _Latent(self.tag, self.shape, device, dtype)


class _Pending:
    def __init__(self, tag):
        self.tag = tag
    def unsqueeze(self, dimension):
        assert dimension == 2  # the pinned call shape
        # Pixels are not latents: the pinned VAE input for a reference is
        # (1, 4, 1, 1024, 1024) while the fake latent is (1, 64, 1, 2, 2);
        # an equal-shape fake would hide the pixel/latent confusion.
        return _Latent(self.tag, shape=_VAE_PIXEL_SHAPE)


class _RecordingImageProcessor:
    def __init__(self):
        self.preprocessed = []
    def preprocess(self, image, width=None, height=None):
        self.preprocessed.append((image.mode, width, height))
        return _Pending(tag=f"pre-{len(self.preprocessed)}")


class _VaeWitnessPipe:
    """Pinned-shape witness for the load-phase encode and the per-call
    override: bound signature ``_encode_vae_image(image, generator)``, a
    recording ``image_processor``, deterministic per-call latents."""
    def __init__(self):
        self.calls = []
        self.latent_channels = 64
        self.image_processor = _RecordingImageProcessor()
    def _encode_vae_image(self, image, generator):
        self.calls.append((image, generator))
        return _Latent(f"encoded-{len(self.calls)}")


class _DriftPipe(_VaeWitnessPipe):
    def _encode_vae_image(self, image, generator, scale=1.0):
        return None


def _install_fake_diffusers_calc(monkeypatch):
    """Stub ONLY the module the lazy calculate_dimensions import targets;
    the fake echoes the pinned square target for any ratio."""
    import sys
    import types
    leaf = types.ModuleType("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
    leaf.calculate_dimensions = lambda area, ratio: (int(area ** 0.5), int(area ** 0.5), None)
    for name in ("diffusers", "diffusers.pipelines", "diffusers.pipelines.qwenimage21"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.qwenimage21.pipeline_qwenimage21", leaf)


class _GradMode:
    """Witness that the load-phase encode runs grad-free: counts entries
    into the fake torch.no_grad() context."""
    def __init__(self):
        self.entered = 0
    def __call__(self):
        grad = self
        class _Ctx:
            def __enter__(self):
                grad.entered += 1
            def __exit__(self, *exc):
                return False
        return _Ctx()


def _stub_torch():
    from types import SimpleNamespace
    return SimpleNamespace(float16="fp16", float32="fp32", no_grad=_GradMode())


def test_encode_reference_latents_uses_the_pinned_preprocessing(tmp_path, monkeypatch):
    from PIL import Image as PILImage
    _install_fake_diffusers_calc(monkeypatch)
    pipe = _VaeWitnessPipe()
    adapter = cli._make_adapter()
    images = [PILImage.new("RGB", (64, 32), (200, 10, 10)),
              PILImage.new("RGBA", (32, 64), (10, 10, 200, 255))]
    torch_stub = _stub_torch()
    latents = adapter._encode_reference_latents(pipe, images, torch_stub, ("cuda", 0))
    # Both inputs reach the VAE as RGBA resized to the pinned
    # output_resolution target; the generator is None (argmax sampling), so
    # the latents are generator-independent.
    assert pipe.image_processor.preprocessed == [("RGBA", 1024, 1024), ("RGBA", 1024, 1024)]
    assert [generator for _, generator in pipe.calls] == [None, None]
    assert [(tensor.device, tensor.dtype) for tensor, _ in pipe.calls] == [
        (("cuda", 0), "fp16"), (("cuda", 0), "fp16")]
    # Stored off-card, in pipeline order, as (input PIXEL shape, latent)
    # pairs: the per-call override can only ever witness the pipeline's
    # pixel tensor, and here the pixel (1, 4, 1, 1024, 1024) genuinely
    # differs from the latent (1, 64, 1, 2, 2) — the shapes the GPU run
    # produced, where an equal-shape comparison rejected every run.
    assert [(shape, latent.tag) for shape, latent in latents] == [
        ((1, 4, 1, 1024, 1024), "encoded-1"),
        ((1, 4, 1, 1024, 1024), "encoded-2")]
    assert all(latent.device == "cpu" for _, latent in latents)
    # The pinned __call__ carries @torch.no_grad(); _encode_vae_image does
    # not, so the load phase must enter it per encode — traced encoder
    # activations (~14 GiB per 1024² reference on the V100) OOM the SECOND
    # reference before the transformer is even placed.
    assert torch_stub.no_grad.entered == 2


def test_encode_reference_latents_refuses_drift_and_bad_shape(tmp_path, monkeypatch):
    from PIL import Image as PILImage
    _install_fake_diffusers_calc(monkeypatch)
    adapter = cli._make_adapter()
    images = [PILImage.new("RGB", (64, 32), (1, 2, 3))]
    with pytest.raises(EngineError, match="signature"):
        adapter._encode_reference_latents(_DriftPipe(), images, _stub_torch(), ("cuda", 0))

    class _BadShape(_VaeWitnessPipe):
        def _encode_vae_image(self, image, generator):
            return _Latent("wide", shape=(4, 64, 1, 2, 2))

    with pytest.raises(EngineError, match="unexpected VAE latent shape"):
        adapter._encode_reference_latents(_BadShape(), images, _stub_torch(), ("cuda", 0))


def test_reference_latents_override_serves_stored_latents_in_order():
    pipe = _VaeWitnessPipe()
    adapter = cli._make_adapter()
    adapter._pipeline = pipe
    adapter._reference_vae_latents = (
        (_VAE_PIXEL_SHAPE, _Latent("first")),
        (_VAE_PIXEL_SHAPE, _Latent("second")))
    adapter._install_reference_latents_hook()
    assert "_encode_vae_image" in vars(pipe)
    served = [pipe._encode_vae_image(
        _Latent(f"in-{i}", shape=_VAE_PIXEL_SHAPE, device=("cuda", 0), dtype="fp16"),
        None) for i in range(2)]
    assert [latent.tag for latent in served] == ["first", "second"]
    assert all(latent.device == ("cuda", 0) and latent.dtype == "fp16"
               for latent in served)
    with pytest.raises(EngineError, match="more reference VAE encodes"):
        pipe._encode_vae_image(_Latent("in-3", shape=_VAE_PIXEL_SHAPE), None)
    adapter._remove_reference_latents_hook()
    assert "_encode_vae_image" not in vars(pipe)
    adapter._remove_reference_latents_hook()  # idempotent


def test_override_refuses_geometry_drift():
    pipe = _VaeWitnessPipe()
    adapter = cli._make_adapter()
    adapter._pipeline = pipe
    # Two stored pairs: a refused encode still POPs (the run dies on the
    # refusal, but the box must not be double-spent).
    adapter._reference_vae_latents = ((_VAE_PIXEL_SHAPE, _Latent("stored")),
                                      (_VAE_PIXEL_SHAPE, _Latent("next")))
    adapter._install_reference_latents_hook()
    with pytest.raises(EngineError, match="preprocessing disagrees"):
        pipe._encode_vae_image(_Latent("tall", shape=(1, 4, 1, 512, 512)), None)
    # A LATENT-shaped input — the identity the buggy pixel-vs-latent
    # comparison could never satisfy — must also refuse: only the pixel
    # geometry is the witness.
    with pytest.raises(EngineError, match="preprocessing disagrees"):
        pipe._encode_vae_image(_Latent("latent-shaped", shape=(1, 64, 1, 2, 2)), None)
    adapter._remove_reference_latents_hook()


def test_override_refuses_missing_method_missing_latents_drift_and_stale():
    from types import SimpleNamespace
    adapter = cli._make_adapter()
    adapter._pipeline = SimpleNamespace()  # no _encode_vae_image at all
    with pytest.raises(EngineError, match="pinned _encode_vae_image"):
        adapter._install_reference_latents_hook()
    pipe = _VaeWitnessPipe()
    adapter._pipeline = pipe
    adapter._reference_vae_latents = None
    with pytest.raises(EngineError, match="no VAE latents prepared"):
        adapter._install_reference_latents_hook()
    drift = cli._make_adapter()
    drift._pipeline = _DriftPipe()
    drift._reference_vae_latents = ((_VAE_PIXEL_SHAPE, _Latent("a")),)
    with pytest.raises(EngineError, match="signature"):
        drift._install_reference_latents_hook()
    adapter._reference_vae_latents = ((_VAE_PIXEL_SHAPE, _Latent("a")),)
    pipe._encode_vae_image = lambda image, generator: None  # stale override in vars
    with pytest.raises(EngineError, match="already installed"):
        adapter._install_reference_latents_hook()


@needs_references
def test_reference_denoise_wraps_inner_call_and_always_removes(monkeypatch, tmp_path):
    class Inner:
        def __init__(self, pipe, order, consume):
            self.pipe, self.order, self.consume = pipe, order, consume
        def denoise(self, config, conditioning):
            self.order.append("override" if "_encode_vae_image" in vars(self.pipe) else "bare")
            for _ in range(self.consume):
                self.pipe._encode_vae_image(
                    _Latent("in", shape=_VAE_PIXEL_SHAPE), None)
            if conditioning == "boom":
                raise EngineError("inner refusal")
            return ["latents"]

    pipe = _VaeWitnessPipe()
    order = []
    adapter = cli._make_adapter()
    adapter._pipeline = pipe
    adapter._joint_tokens = 100
    adapter._reference_vae_latents = ((_VAE_PIXEL_SHAPE, _Latent("a")),
                                      (_VAE_PIXEL_SHAPE, _Latent("b")))
    monkeypatch.setattr(adapter, "_apply_joint_chunking", lambda: order.append("chunk"))
    monkeypatch.setattr(adapter, "_release_vae", lambda: None)
    monkeypatch.setattr(adapter, "_release_denoiser", lambda: None)
    config = _reference_config(tmp_path)

    adapter._inner = Inner(pipe, order, consume=2)
    assert adapter.denoise(config, "cond") == ["latents"]
    assert order == ["chunk", "override"]
    assert "_encode_vae_image" not in vars(pipe) and adapter._reference_latents_hook is None
    assert adapter._reference_latents_box is None

    adapter._inner = Inner(pipe, order, consume=1)  # a truncated consumption
    with pytest.raises(EngineError, match="consumed 1 of 2"):
        adapter.denoise(config, "cond")
    assert "_encode_vae_image" not in vars(pipe)    # removed even on refusal

    adapter._inner = Inner(pipe, order, consume=0)
    with pytest.raises(EngineError, match="inner"):
        adapter.denoise(config, "boom")
    assert "_encode_vae_image" not in vars(pipe)    # the finally always removes


@needs_references
def test_encode_derives_joint_tokens_and_facts_merge_them(tmp_path):
    class Inner:
        def encode(self, config):
            return "encoding"

        def calibration_facts(self):
            return {"prompt_tokens": 25, "negative_prompt_tokens": None,
                    "cfg_active": False, "encoder_dtype": "fp16",
                    "execution_dtype": "fp16", "use_kv_cache": True,
                    "embedding_cache_hit": True, "references": 2}

    adapter = cli._make_adapter()
    adapter._inner = Inner()
    adapter._joint_reference_tokens = lambda config: 2048
    config = _reference_config(tmp_path, references=("a" * 64, "b" * 64))
    assert adapter.encode(config) == "encoding"
    assert adapter._joint_tokens == 25 + 2048 + config.latent_tokens
    assert adapter.calibration_facts()["joint_tokens"] == adapter._joint_tokens
    adapter._joint_tokens = None
    assert "joint_tokens" not in adapter.calibration_facts()


@needs_references
def test_t2i_denoise_path_never_touches_reference_hooks(tmp_path, monkeypatch):
    class Inner:
        def denoise(self, config, conditioning):
            assert "_encode_vae_image" not in vars(pipe)
            return ["latents"]

    pipe = _VaeWitnessPipe()
    adapter = cli._make_adapter()
    adapter._pipeline = pipe
    adapter._inner = Inner()
    monkeypatch.setattr(adapter, "_release_vae", lambda: None)
    monkeypatch.setattr(adapter, "_release_denoiser", lambda: None)
    assert adapter.denoise(_config(tmp_path), None) == ["latents"]
    assert pipe.calls == [] and pipe.image_processor.preprocessed == []


@needs_references
def test_close_drops_reference_latents():
    adapter = cli._make_adapter()
    adapter._reference_vae_latents = ((_VAE_PIXEL_SHAPE, _Latent("a")),)
    adapter._reference_latents_box = [_Latent("a")]
    adapter.close()
    assert adapter._reference_vae_latents is None
    assert adapter._reference_latents_box is None


# -- load-phase residency order over the REAL supplied bridge --------------------


def _load_env(tmp_path, monkeypatch, references, quant=None):
    """One full loader.load over fakes, real bundle/manifest/bridge.

    Returns (adapter, order, pipe): ORDER records every component
    load/encode so the residency order is directly assertable, and the
    real SuppliedQwen21Adapter attachment proves the relaxed reference-run
    component gate accepts a VAE that already left the card."""
    import hashlib
    import io
    import json
    from types import SimpleNamespace
    from PIL import Image as PILImage
    from qwen_engine import reference_store
    from qwen_engine.adapters import qwen21_diffusers as qd
    from qwen_engine.adapters import supplied_qwen21 as sp
    from qwen_engine.local_bundle import (BASE_PROFILE, LOCAL_REFERENCE_LAYOUT,
                                          resolve_local_bundle)
    from qwen_engine.manifest import load_manifest

    order = []
    root = tmp_path / "models"
    artifacts = []
    for role, names in LOCAL_REFERENCE_LAYOUT.items():
        for name in names:
            raw = f"{role}:{name}".encode()
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            artifacts.append({"role": role, "repo": BASE_PROFILE.repo,
                              "revision": BASE_PROFILE.revision, "filename": name,
                              "sha256": hashlib.sha256(raw).hexdigest(),
                              "bytes": len(raw)})
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "artifacts": artifacts}))

    dev = SimpleNamespace(type="cpu")

    class _Param:
        is_meta = False
        device = dev
        dtype = "fp16"

    class _FakeTransformer:
        training = False

        def __init__(self):
            self.config = SimpleNamespace(in_channels=64, out_channels=64,
                                          context_in_dim=16)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            order.append("load-transformer")
            return cls()
        def to(self, **kwargs):
            return self
        def eval(self):
            return self
        def parameters(self):
            return iter([_Param()])
        def modules(self):
            return iter([self])
        def set_attention_backend(self, name):
            self.backend = name

    class _FakeVae(_FakeTransformer):
        def __init__(self):
            self.config = SimpleNamespace(z_dim=64)
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            order.append("load-vae")
            return cls()
        def enable_tiling(self, **kwargs):
            pass
        def disable_tiling(self):
            pass

    class _FakeScheduler:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    class _FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    class _FakePipe:
        def __init__(self, scheduler, vae, text_encoder, processor, transformer,
                     sample_sigmas=None):
            self.scheduler, self.vae, self.text_encoder = scheduler, vae, text_encoder
            self.processor, self.transformer = processor, transformer
            self.vae_scale_factor = 16
            self.latent_channels = vae.config.z_dim if vae is not None else 64
            self.image_processor = _RecordingImageProcessor()
            self._execution_device = dev
            self.encode_calls = []
        def _encode_vae_image(self, image, generator):
            order.append("encode")
            self.encode_calls.append((image, generator))
            return _Latent("reference-latent")

    torch = SimpleNamespace(float16="fp16", float32="fp32", no_grad=_GradMode(),
                            device=lambda kind, index: (kind, index))
    runtime = SimpleNamespace(matches_cuda_device=lambda **kwargs: True,
                              sha256="f" * 64)
    monkeypatch.setattr(qd, "_libraries", lambda: (
        torch, _FakePipe, _FakeTransformer, _FakeVae, _FakeScheduler, _FakeProcessor))
    monkeypatch.setattr(sp, "_libraries", lambda: (
        torch, SimpleNamespace(), (_FakePipe, _FakeTransformer, _FakeVae, _FakeScheduler)))
    monkeypatch.setattr(qd, "_require_v100",
                        lambda _torch: SimpleNamespace(uuid="GPU-fake"))
    monkeypatch.setattr(qd, "attach_cuda", lambda _torch: None)
    monkeypatch.setattr(qd, "observe_generation_runtime", lambda: runtime)
    _install_fake_diffusers_calc(monkeypatch)

    values = dict(manifest_path=str(manifest_path), models_dir=str(root),
                  width=256, height=256)
    if quant == "q8":
        from qwen_engine import q8_artifact as qa_mod
        from qwen_engine.adapters import q8_loader
        from qwen_engine.config import Quant
        monkeypatch.setattr(
            qa_mod, "resolve_q8_artifact",
            lambda path: SimpleNamespace(identity_sha256="e" * 64))
        monkeypatch.setattr(q8_loader, "validate_q8_environment", lambda: None)
        # the GGUF-parameter witness needs real gguf/torch; the placement it
        # enforces is represented here by the fake transformer's params
        monkeypatch.setattr(q8_loader, "validate_q8_parameters",
                            lambda model, device, dtype: None)
        monkeypatch.setattr(
            q8_loader, "load_q8_transformer",
            lambda *args: (order.append("load-q8-transformer"),
                           _FakeTransformer())[1])
        values.update(quant=Quant.Q8, q8_manifest_path=str(root / "q8.json"))
    if references:
        store = tmp_path / "store"
        store.mkdir()
        stream = io.BytesIO()
        PILImage.new("RGB", (48, 32), (9, 99, 200)).save(stream, format="PNG")
        values.update(references=(reference_store.put_bytes(store, stream.getvalue()),),
                      references_dir=str(store))
    adapter = cli._make_adapter()
    assets = resolve_local_bundle(load_manifest(manifest_path), root).to_assets()
    adapter.load(_config(tmp_path, **values), assets)
    return adapter, order, adapter._pipeline


@needs_references
def test_reference_load_encodes_before_the_transformer_and_attaches(tmp_path, monkeypatch):
    adapter, order, pipe = _load_env(tmp_path, monkeypatch, references=True)
    # The measured OOM fix, witnessed: the VAE encodes alone, leaves the
    # card, and only then does the transformer load.
    assert order == ["load-vae", "encode", "load-transformer"]
    assert len(adapter._reference_vae_latents) == 1
    assert pipe.vae is None                      # VAE off the card at sampling load-in
    assert pipe.transformer is not None
    assert pipe.encode_calls[0][1] is None       # argmax: generator-independent
    # The REAL bridge attached with a non-resident VAE (relaxed reference gate).
    assert adapter._inner._state == "loaded"
    adapter.close()


@needs_references
def test_t2i_load_order_and_attachment_are_unchanged(tmp_path, monkeypatch):
    adapter, order, pipe = _load_env(tmp_path, monkeypatch, references=False)
    assert order == ["load-transformer", "load-vae"]   # unchanged T2I sequence
    assert pipe.vae is not None                        # resident, as before
    assert adapter._reference_vae_latents is None
    assert adapter._inner._state == "loaded"
    adapter.close()


@needs_references
@pytest.mark.parametrize("quant,references,expect", [
    (None, False, ("t2i-on", True)),
    (None, True, ("reference-fp16-off", False)),
    ("q8", True, ("q8-on", True)),
    ("q8", False, ("t2i-on", True)),
])
def test_kv_cache_policy_is_automatic_from_references_and_quant(
        tmp_path, monkeypatch, quant, references, expect):
    """V100 measured policy: FP16+KV-on OOMs the first denoise step of a
    reference run; KV-off fits (95 s), Q8+KV-on fits (58 s), Turbo FP16
    KV-off fits (19 s); T2I keeps KV-on. No user-facing field exists."""
    from types import SimpleNamespace
    from qwen_engine.adapters import qwen21_diffusers as qd
    policy, use_kv = expect
    stub = SimpleNamespace(metadata={})
    monkeypatch.setattr(qd, "current", lambda: stub)
    adapter, _order, _pipe = _load_env(tmp_path, monkeypatch, references, quant=quant)
    assert adapter._inner.use_kv_cache is use_kv
    assert stub.metadata["kv_cache_policy"] == policy
    adapter.close()


def _fake_torch(name, total):
    from types import SimpleNamespace
    properties = SimpleNamespace(major=7, minor=0, name=name, total_memory=total)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_properties=lambda index: properties,
                           get_arch_list=lambda: ["sm_70"])
    return SimpleNamespace(version=SimpleNamespace(cuda="12.6"), cuda=cuda)


@pytest.mark.parametrize("name, total", [("Tesla V100-SXM2-16GB", 16945512448),
                                         ("Tesla V100-SXM2-32GB", 34089730048),
                                         ("Tesla V100S-PCIE-32GB", 34089730048)])
def test_the_pipeline_accepts_16gb_and_32gb_v100s(name, total):
    from qwen_engine.adapters import qwen21_diffusers as qd
    assert qd._require_v100(_fake_torch(name, total)).total_memory == total


def test_the_pipeline_refuses_a_v100_sized_unlike_any_reviewed_card():
    from qwen_engine.adapters import qwen21_diffusers as qd
    from qwen_engine.errors import HardwareAbsentError
    with pytest.raises(HardwareAbsentError, match="V100 \\(16GB or 32GB\\)"):
        qd._require_v100(_fake_torch("Tesla V100-SXM2-16GB", 24 * 1024**3))
