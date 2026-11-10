"""Exactness bridge + collector gates. ML-free: no torch/diffusers import.

The collector's real sampling is exercised only on hardware; these tests pin
the refusal rules: a record missing any exactness field never reaches the
memory policy, a legacy record stays loadable but ineligible, and a failed
measurement never writes a calibration file.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from qwen_engine import calibration as cal
from qwen_engine import memory
from qwen_engine.bench import (
    EXACT_OBSERVATION_FIELDS,
    BenchSchemaError,
    CalibrationLog,
    CalibrationObservation,
)
from qwen_engine.config import GenerationConfig
from qwen_engine.mem_preflight import MemorySnapshot
from qwen_engine.memory import DeviceIdentity, ResourceEvidence

GiB = 1024**3
UUID = "GPU-01234567-0123-0123-0123-012345678901"


def exact_observation(**overrides):
    values = dict(
        observed_at=1000.0, phase="denoise", model_revision="a" * 40,
        backend="sdpa", quant="fp16", residency="resident",
        vae_tiled_decode=False, cfg_duplicated=False,
        tokens=4096, resolution=(1024, 1024),
        peak_gpu_allocated_bytes=13 * GiB,
        peak_gpu_reserved_bytes=14 * GiB,
        peak_host_rss_bytes=2 * GiB,
        prompt_tokens=25,
        steps=40, negative_prompt_tokens=10, cfg_active=True,
        encoder_dtype="fp16", execution_dtype="fp16", use_kv_cache=True,
        step_index=None, embedding_cache_hit=False,
        artifact_set_sha256="b" * 64, runtime_identity_sha256="c" * 64,
        device_uuid=UUID, device_architecture="sm_70",
        driver_version="580.178.04", cuda_build="12.6",
        peak_gpu_used_bytes=15 * GiB,
        peak_host_tree_bytes=3 * GiB,
    )
    values.update(overrides)
    return CalibrationObservation(**values)


def legacy_observation():
    """A record shape from before the exactness extension."""
    return CalibrationObservation(
        observed_at=900.0, phase="load", model_revision="a" * 40,
        backend="sdpa", quant="fp16", residency="resident",
        tokens=4096, resolution=(1024, 1024),
        peak_gpu_allocated_bytes=13 * GiB,
        peak_gpu_reserved_bytes=14 * GiB,
        peak_host_rss_bytes=2 * GiB,
        prompt_tokens=25,
    )


# -- schema round-trip -------------------------------------------------------


def test_exact_record_round_trips_through_the_calibration_file():
    log = CalibrationLog(observations=(exact_observation(),))
    again = CalibrationLog.from_dict(json.loads(log.to_json()))
    assert again.observations == log.observations
    payload = json.loads(log.to_json())["observations"][0]
    assert payload["peak_gpu_used_bytes"] == 15 * GiB
    assert payload["device_uuid"] == UUID


def test_legacy_record_still_loads_and_reports_every_missing_field():
    log = CalibrationLog(observations=(legacy_observation(),))
    again = CalibrationLog.from_dict(json.loads(log.to_json()))
    missing = cal.missing_exact_fields(again.observations[0])
    # The legacy record did measure reserved/RSS peaks; everything else that
    # makes a record exact (device-wide used, tree peak, identities, config)
    # is absent and must be named, never defaulted.
    assert missing == tuple(sorted(
        set(EXACT_OBSERVATION_FIELDS) - {
            "negative_prompt_tokens", "step_index", "prompt_tokens",
            "peak_gpu_reserved_bytes", "peak_host_rss_bytes"}))
    with pytest.raises(cal.CalibrationError, match="not exact evidence"):
        cal.policy_descriptor(again.observations[0])
    with pytest.raises(cal.CalibrationError, match="not exact evidence"):
        cal.calibration_evidence(again.observations[0])


@pytest.mark.parametrize("field", [
    name for name in EXACT_OBSERVATION_FIELDS
    if name not in ("negative_prompt_tokens", "step_index")])
def test_every_absent_exactness_field_is_named(field):
    observation = exact_observation(**{field: None})
    assert cal.missing_exact_fields(observation) == (field,)
    with pytest.raises(cal.CalibrationError, match=field):
        cal.calibration_evidence(observation)


def test_inactive_cfg_needs_no_negative_length_but_active_cfg_does():
    inactive = exact_observation(cfg_active=False, negative_prompt_tokens=None)
    assert cal.missing_exact_fields(inactive) == ()
    active = exact_observation(cfg_active=True, negative_prompt_tokens=None)
    assert cal.missing_exact_fields(active) == ("negative_prompt_tokens",)
    # The schema itself refuses the contradictory inactive+length record.
    with pytest.raises(BenchSchemaError, match="inactive CFG"):
        exact_observation(cfg_active=False, negative_prompt_tokens=10)


def test_footprint_ordering_is_enforced_at_the_schema_boundary():
    with pytest.raises(BenchSchemaError, match="device-wide"):
        exact_observation(peak_gpu_reserved_bytes=16 * GiB,
                          peak_gpu_used_bytes=15 * GiB)
    with pytest.raises(BenchSchemaError, match="owned-process-tree"):
        exact_observation(peak_host_rss_bytes=4 * GiB,
                          peak_host_tree_bytes=3 * GiB)


# -- bridge ------------------------------------------------------------------


def test_bridge_maps_every_field_and_the_policy_consumes_the_result():
    observation = exact_observation()
    descriptor = cal.policy_descriptor(observation)
    assert descriptor.phase == "denoise"
    assert descriptor.step_index is None
    assert descriptor.latent_tokens == 4096
    assert descriptor.artifact_set_sha256 == "b" * 64
    assert descriptor.runtime_identity_sha256 == "c" * 64
    evidence = cal.calibration_evidence(observation)
    device = DeviceIdentity(UUID, "sm_70", "580.178.04", "12.6")
    assert evidence.device == device
    assert evidence.peak_gpu_used_bytes == 15 * GiB
    assert evidence.peak_host_tree_bytes == 3 * GiB

    resources = ResourceEvidence(
        captured_at=1000.0,
        host=MemorySnapshot(64 * GiB, "unlimited"),
        device=device, cuda_kernel_verified=True,
        gpu_total_bytes=16 * GiB, gpu_available_bytes=16 * GiB,
    )
    decision = memory.assess_memory(
        descriptor, resources, (evidence,), host_headroom_bytes=0,
        gpu_headroom_bytes=0, max_resource_age_s=5,
        max_calibration_age_s=60, now=1000.0)
    assert decision.allowed, decision.reasons
    assert decision.required_gpu_bytes == 15 * GiB
    assert decision.required_host_bytes == 3 * GiB


def test_single_step_and_whole_loop_phases_never_match():
    whole = exact_observation()
    step = exact_observation(phase="denoise-step", step_index=2)
    whole_evidence = cal.calibration_evidence(whole)
    step_evidence = cal.calibration_evidence(step)
    device = whole_evidence.device
    resources = ResourceEvidence(
        1000.0, MemorySnapshot(64 * GiB, "unlimited"), device, True,
        16 * GiB, 16 * GiB)
    decision = memory.assess_memory(
        whole_evidence.descriptor, resources, (step_evidence,),
        host_headroom_bytes=0, gpu_headroom_bytes=0,
        max_resource_age_s=5, max_calibration_age_s=60, now=1000.0)
    assert not decision.allowed
    assert "exact_calibration_missing" in decision.reasons


def test_unreviewed_descriptor_values_are_refused_by_the_bridge():
    with pytest.raises(cal.CalibrationError, match="policy descriptor"):
        cal.policy_descriptor(exact_observation(backend="auto"))
    with pytest.raises(cal.CalibrationError, match="policy descriptor"):
        cal.policy_descriptor(exact_observation(execution_dtype="bf16"))


def test_device_identity_requires_all_four_measured_labels():
    observation = exact_observation(driver_version=None)
    assert "driver_version" in cal.missing_exact_fields(observation)
    with pytest.raises(cal.CalibrationError):
        cal.device_identity(observation)


# -- collector gates ---------------------------------------------------------


def _config(**overrides):
    values = dict(
        prompt="a red cube", output_dir="/tmp/out", cache_dir="/tmp/cache",
        manifest_path="/tmp/manifest.json", models_dir="/tmp/models",
    )
    values.update(overrides)
    return GenerationConfig(**values)


def test_reviewed_device_gate_compares_values_not_types():
    # Regression: comparing type(major)/type(minor) refuses every real GPU.
    good = SimpleNamespace(major=7, minor=0, name="Tesla V100-SXM2-16GB",
                           total_memory=16945512448)
    cal._require_reviewed_device(good)
    for bad in (
        SimpleNamespace(major=8, minor=0, name=good.name,
                        total_memory=good.total_memory),
        SimpleNamespace(major=7, minor=5, name=good.name,
                        total_memory=good.total_memory),
        SimpleNamespace(major=7, minor=0, name="Tesla T4",
                        total_memory=good.total_memory),
        SimpleNamespace(major=7, minor=0, name=good.name,
                        total_memory=8 * GiB),
        SimpleNamespace(major=7, minor=0, name=good.name,
                        total_memory=17 * GiB),
        SimpleNamespace(major=7, minor=0, name=good.name,
                        total_memory=24 * GiB),
        SimpleNamespace(major=7, minor=0, name=good.name,
                        total_memory=33 * GiB),
    ):
        with pytest.raises(cal.CalibrationError, match="reviewed Tesla V100"):
            cal._require_reviewed_device(bad)


@pytest.mark.parametrize("name", ["Tesla V100-SXM2-32GB", "Tesla V100-PCIE-32GB",
                                  "Tesla V100S-PCIE-32GB"])
def test_the_32gb_v100_cards_are_reviewed_devices(name):
    # What the driver reports for a 32GB V100: just under 32 GiB.
    cal._require_reviewed_device(SimpleNamespace(major=7, minor=0, name=name,
                                                 total_memory=34089730048))


def _facts(**overrides):
    values = dict(prompt_tokens=25, negative_prompt_tokens=None,
                  cfg_active=False, encoder_dtype="fp16",
                  execution_dtype="fp16", use_kv_cache=True,
                  embedding_cache_hit=False)
    values.update(overrides)
    return values


@pytest.mark.parametrize("change", [
    {"embedding_cache_hit": None},
    {"embedding_cache_hit": 1},
    {"prompt_tokens": 0},
    {"prompt_tokens": True},
    {"use_kv_cache": None},
    {"encoder_dtype": "auto"},
    {"execution_dtype": "bf16"},
    {"cfg_active": True, "negative_prompt_tokens": None},
    {"cfg_active": False, "negative_prompt_tokens": 3},
])
def test_adapter_facts_are_a_hard_coherence_gate(change):
    with pytest.raises(cal.CalibrationError):
        cal.validate_adapter_facts(_facts(**change), _config())


def test_adapter_facts_accept_both_cfg_modes():
    assert cal.validate_adapter_facts(_facts(), _config())["cfg_active"] is False
    active = _config(negative_prompt="blurry", true_cfg_scale=2.0)
    assert cal.validate_adapter_facts(
        _facts(cfg_active=True, negative_prompt_tokens=12), active)[
            "negative_prompt_tokens"] == 12


def _collector_stub(records, facts, config):
    collector = cal.CalibrationCollector.__new__(cal.CalibrationCollector)
    collector._config = config
    collector._records = records
    collector._facts = facts
    collector._runtime_sha256 = "c" * 64
    collector._device = DeviceIdentity(UUID, "sm_70", "580.178.04", "12.6")
    collector._revision = "a" * 40
    collector._artifact_set_sha256 = "b" * 64
    collector._current_phase = None
    collector._failure = None
    collector._finished = False
    return collector


def _record(**overrides):
    values = dict(
        phase="denoise", observed_at=1000.0,
        peak_gpu_allocated=13 * GiB, peak_gpu_reserved=14 * GiB,
        window=cal._Window(15 * GiB, 2 * GiB, 3 * GiB, 12),
    )
    values.update(overrides)
    return cal._PhaseRecord(**values)


def test_collector_builds_one_exact_observation_per_phase():
    config = _config()
    collector = _collector_stub(
        [_record(phase="load"), _record(phase="denoise")], _facts(), config)
    observations = collector._build_observations()
    assert [o.phase for o in observations] == ["load", "denoise"]
    assert all(o.model_revision == "a" * 40 for o in observations)
    assert all(o.runtime_identity_sha256 == "c" * 64 for o in observations)
    assert all(o.peak_gpu_used_bytes == 15 * GiB for o in observations)
    assert all(o.peak_host_tree_bytes == 3 * GiB for o in observations)
    assert all(o.resolution == (1024, 1024) for o in observations)
    assert all(o.cfg_active is False for o in observations)
    assert all(o.negative_prompt_tokens is None for o in observations)
    for observation in observations:
        assert cal.missing_exact_fields(observation) == ()


def test_collector_refuses_to_write_when_a_peak_was_not_measured(tmp_path):
    class _Sampler:
        def stop(self):
            pass

    config = _config()
    collector = _collector_stub(
        [_record(peak_gpu_reserved=None)], _facts(), config)
    collector._sampler = _Sampler()
    collector._nvml = None
    target = tmp_path / "calibration.json"
    collector.path = target
    with pytest.raises(cal.CalibrationError, match="peak_gpu_reserved_bytes"):
        collector.finish()
    assert not target.exists()


def test_collector_refuses_an_unmeasured_cache_decision(tmp_path):
    class _Sampler:
        def stop(self):
            pass

    config = _config()
    collector = _collector_stub(
        [_record()], _facts(embedding_cache_hit=None), config)
    collector._sampler = _Sampler()
    collector._nvml = None
    target = tmp_path / "calibration.json"
    collector.path = target
    with pytest.raises(cal.CalibrationError, match="coherent"):
        collector.finish()
    assert not target.exists()
