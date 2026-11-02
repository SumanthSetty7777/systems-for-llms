"""Exercise the actual preflight script without CUDA or external subprocesses."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "run_modal.py"
spec = importlib.util.spec_from_file_location("distributed_preflight_test", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _execute(monkeypatch, tmp_path, *, topology=None, identities=None,
             gpu_count=2, bf16=True, peer_error=False):
    calls, selected = [], []

    def native_bf16(*, including_emulation):
        assert including_emulation is False
        return bf16

    def peer_access(first, second):
        if peer_error:
            raise RuntimeError("Peer capability query is unavailable")
        return first != second

    cuda = SimpleNamespace(
        device_count=lambda: gpu_count,
        set_device=lambda index: selected.append(index),
        is_bf16_supported=native_bf16,
        get_device_properties=lambda index: SimpleNamespace(
            name="NVIDIA H100", uuid=f"GPU-test-{index}", total_memory=80 * 1024**3),
        get_device_capability=lambda index: (9, 0),
        can_device_access_peer=peer_access,
    )
    fake_torch = SimpleNamespace(cuda=cuda, __version__="2.11.0", version=SimpleNamespace(cuda="13.0"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def command(args, *, capture_output, text, timeout):
        assert capture_output and text and timeout > 0
        calls.append(args)
        response = topology if args[1] == "topo" else identities
        if isinstance(response, Exception):
            raise response
        if response is None:
            response = (0, "topology available\n" if args[1] == "topo" else "GPU-test-0, NVIDIA H100, 0000:01:00.0\n", "")
        code, stdout, stderr = response
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", command)
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(sys, "argv", ["-c", str(output), "2"])
    exec(compile(launcher.PREFLIGHT, "<distributed-preflight>", "exec"), {"__name__": "__main__"})
    return json.loads(output.read_text()), calls, selected


def test_successful_preflight_preserves_topology_and_torch_devices(monkeypatch, tmp_path):
    report, calls, selected = _execute(monkeypatch, tmp_path)
    assert selected == [0, 1] and len(calls) == 2
    assert report["nvidia_smi_topology"] == "topology available\n"
    assert report["nvidia_smi_diagnostics"]["topology"]["status"] == "completed"
    assert report["nvidia_smi_diagnostics"]["identities"]["returncode"] == 0
    assert report["warnings"] == []
    assert report["devices"][1] == {
        "index": 1, "name": "NVIDIA H100", "uuid": "GPU-test-1",
        "memory_bytes": 80 * 1024**3, "capability": [9, 0],
    }


def test_failed_topology_with_empty_stderr_does_not_block_cuda_gate(monkeypatch, tmp_path):
    report, calls, _ = _execute(monkeypatch, tmp_path, topology=(1, "Topology query not supported\n", ""))
    diagnostic = report["nvidia_smi_diagnostics"]["topology"]
    assert diagnostic["returncode"] == 1 and diagnostic["status"] != "completed"
    assert diagnostic["stdout"] == "Topology query not supported\n" and diagnostic["stderr"] == ""
    assert diagnostic["command"] == ["nvidia-smi", "topo", "-m"]
    assert report["nvidia_smi_topology"] is None
    assert len(calls) == 2 and report["warnings"]
    assert report["devices"][0]["uuid"] == "GPU-test-0"


@pytest.mark.parametrize("failure,status", [
    (FileNotFoundError("nvidia-smi is unavailable"), "unavailable"),
    (subprocess.TimeoutExpired(["nvidia-smi", "topo", "-m"], 30,
                               output=b"partial topology\n", stderr=b"timed out\n"), "timeout"),
])
def test_optional_diagnostic_missing_or_timed_out_preserves_torch_identity(monkeypatch, tmp_path, failure, status):
    report, calls, _ = _execute(monkeypatch, tmp_path, topology=failure, identities=failure)
    for name in ("topology", "identities"):
        diagnostic = report["nvidia_smi_diagnostics"][name]
        assert diagnostic["status"] == status and diagnostic["returncode"] is None
        assert isinstance(diagnostic["stdout"], str) and isinstance(diagnostic["stderr"], str)
    assert report["nvidia_smi_topology"] is None and report["nvidia_smi_identities"] is None
    assert len(calls) == 2 and report["warnings"]
    assert [device["uuid"] for device in report["devices"]] == ["GPU-test-0", "GPU-test-1"]


def test_failed_identity_query_is_recorded_without_erasing_torch_identity(monkeypatch, tmp_path):
    report, _, _ = _execute(monkeypatch, tmp_path, identities=(9, "", "Identity query unavailable"))
    diagnostic = report["nvidia_smi_diagnostics"]["identities"]
    assert diagnostic["returncode"] == 9 and diagnostic["stderr"] == "Identity query unavailable"
    assert report["nvidia_smi_identities"] is None and report["warnings"]
    assert report["devices"][0]["name"] == "NVIDIA H100"


def test_unavailable_peer_capability_is_unknown_instead_of_false(monkeypatch, tmp_path):
    report, _, _ = _execute(monkeypatch, tmp_path, peer_error=True)
    assert report["peer_access"] == [[True, None], [None, True]]
    assert report["warnings"]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_wrong_cuda_count_remains_fatal(monkeypatch, tmp_path, count):
    with pytest.raises(RuntimeError, match="Expected exactly 2 CUDA GPUs"):
        _execute(monkeypatch, tmp_path, gpu_count=count)
    assert not (tmp_path / "preflight.json").exists()


def test_missing_native_bf16_remains_fatal(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="lacks native BF16 support"):
        _execute(monkeypatch, tmp_path, bf16=False)
    assert not (tmp_path / "preflight.json").exists()
