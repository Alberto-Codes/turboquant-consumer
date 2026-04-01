r"""Experiment 029 -- End-to-end vLLM serving benchmarks.

3-layer benchmark protocol measuring TQ4 KV cache compression overhead
in a real vLLM serving context.  Automates server lifecycle, VRAM
monitoring via nvidia-smi polling, and multi-run reproducibility.

Layers:
    1. Offline throughput  (``vllm bench throughput``)  — pure engine overhead
    2. Online saturation   (``vllm bench serve --request-rate inf``) — peak capacity
    3. Online latency sweep (``vllm bench serve --request-rate 1..16``) — QPS profile

Reproducibility: 3 independent runs per configuration (fresh server restart
each run for online layers).  Reports median with P25/P75 range.

Usage:
    # TQ4 Molmo2-4B
    uv run python experiments/experiment_029_vllm_serving_benchmark.py \
        --model allenai/Molmo2-4B --tag molmo2-4b-tq4 \
        --attention-backend CUSTOM --enforce-eager --trust-remote-code

    # TQ4 Llama-3.1-8B
    uv run python experiments/experiment_029_vllm_serving_benchmark.py \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama-8b-tq4 \
        --attention-backend CUSTOM --enforce-eager

    # Baseline (no TQ4, separate venv)
    uv run python experiments/experiment_029_vllm_serving_benchmark.py \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama-8b-baseline \
        --vllm-binary /tmp/baseline-venv/bin/vllm --enforce-eager

    # Fused eager (TQ4_USE_FUSED_PAGED=1)
    TQ4_USE_FUSED_PAGED=1 uv run python experiments/experiment_029_vllm_serving_benchmark.py \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama-8b-fused-eager \
        --attention-backend CUSTOM --enforce-eager --skip-offline

    # Fused + CUDA graphs (no --enforce-eager)
    TQ4_USE_FUSED_PAGED=1 uv run python experiments/experiment_029_vllm_serving_benchmark.py \
        --model meta-llama/Llama-3.1-8B-Instruct --tag llama-8b-fused-cg \
        --attention-backend CUSTOM --skip-offline

See Also:
    ``experiments/experiment_028_decode_oom_validation.py``:
        VRAM measurement pattern and server health check.
    ``experiments/experiment_020_fused_paged_decode_benchmark.py``:
        Kernel-level fused benchmark reference.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_GPU_UTIL = 0.85
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_INPUT_LEN = 512
DEFAULT_OUTPUT_LEN = 128
NUM_PROMPTS_OFFLINE = 200
NUM_PROMPTS_SATURATION = 200
NUM_PROMPTS_SWEEP = 100
NUM_WARMUPS = 10
DEFAULT_RUNS = 3
SWEEP_RATES = [1, 2, 4, 8, 16]
HEALTH_TIMEOUT_S = 300
VRAM_POLL_S = 0.1
SERVER_PORT = 8000


# ── VRAM Monitor ──────────────────────────────────────────────────────


class VRAMMonitor:
    """Background nvidia-smi poller that records peak GPU VRAM (MiB)."""

    def __init__(self, interval_s: float = VRAM_POLL_S) -> None:
        """Initialize with polling interval in seconds."""
        self._interval = interval_s
        self._peak = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin background VRAM polling."""
        self._running = True
        self._peak = _get_vram_mib()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        while self._running:
            val = _get_vram_mib()
            if val > 0:
                self._peak = max(self._peak, val)
            time.sleep(self._interval)

    def stop(self) -> int:
        """Stop polling and return peak VRAM in MiB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self._peak

    @property
    def peak_mib(self) -> int:
        """Current peak VRAM reading in MiB."""
        return self._peak


def _get_vram_mib() -> int:
    """Single nvidia-smi VRAM reading (first GPU, MiB)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(result.stdout.strip().split("\n")[0])
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return -1


# ── Server Management ─────────────────────────────────────────────────

_active_server: subprocess.Popen[str] | None = None


def _cleanup_server() -> None:
    """Atexit handler — terminate any leftover server process."""
    global _active_server, _server_log_fh
    if _active_server and _active_server.poll() is None:
        print("\n[cleanup] Stopping vLLM server...")
        _active_server.terminate()
        try:
            _active_server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _active_server.kill()
            _active_server.wait()
        _active_server = None
    if _server_log_fh:
        _server_log_fh.close()
        _server_log_fh = None


atexit.register(_cleanup_server)


_server_log_fh: Any = None


def _start_server(
    vllm_binary: str,
    model: str,
    *,
    attention_backend: str | None = None,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_num_batched_tokens: int | None = None,
    trust_remote_code: bool = False,
    port: int = SERVER_PORT,
    extra_env: dict[str, str] | None = None,
    log_dir: Path | None = None,
    tag: str = "server",
    run_id: int = 0,
) -> subprocess.Popen[str]:
    """Start a vLLM serve process and register it for cleanup."""
    global _active_server, _server_log_fh

    cmd = [
        vllm_binary,
        "serve",
        model,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
    ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    env = {**os.environ, **(extra_env or {})}
    print(f"[server] Starting: {' '.join(cmd)}")

    # Redirect to log file — piping to PIPE deadlocks when the buffer fills.
    base = log_dir or Path("experiments/logs")
    log_path = base / f"experiment-029-{tag}-server-run{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _server_log_fh = open(log_path, "w")  # noqa: SIM115
    print(f"[server] Log: {log_path}")

    proc = subprocess.Popen(
        cmd,
        stdout=_server_log_fh,
        stderr=subprocess.STDOUT,
        env=env,
    )
    _active_server = proc
    return proc


def _wait_for_health(base_url: str, timeout_s: int = HEALTH_TIMEOUT_S) -> None:
    """Block until the vLLM /v1/models endpoint responds 200."""
    global _active_server
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # Check if server process has exited
        if _active_server and _active_server.poll() is not None:
            raise RuntimeError(
                f"Server exited with code {_active_server.returncode} during startup"
            )
        try:
            resp = requests.get(f"{base_url}/models", timeout=5)
            if resp.ok:
                models = [m["id"] for m in resp.json()["data"]]
                print(f"[server] Ready — serving: {models}")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"Server not ready after {timeout_s}s")


def _stop_server(proc: subprocess.Popen[str]) -> None:
    """Gracefully terminate a vLLM server process."""
    global _active_server, _server_log_fh
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if _active_server is proc:
        _active_server = None
    if _server_log_fh:
        _server_log_fh.close()
        _server_log_fh = None
    print("[server] Stopped")


# ── Benchmark Runners ─────────────────────────────────────────────────


def _run_offline_throughput(
    vllm_binary: str,
    model: str,
    run_id: int,
    tag: str,
    result_dir: Path,
    *,
    attention_backend: str | None = None,
    enforce_eager: bool = False,
    gpu_memory_utilization: float = DEFAULT_GPU_UTIL,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    trust_remote_code: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Layer 1: offline throughput via ``vllm bench throughput``."""
    out_file = result_dir / f"experiment-029-{tag}-offline-run{run_id}.json"

    cmd = [
        vllm_binary,
        "bench",
        "throughput",
        "--model",
        model,
        "--dataset-name",
        "random",
        "--num-prompts",
        str(NUM_PROMPTS_OFFLINE),
        "--random-input-len",
        str(DEFAULT_INPUT_LEN),
        "--random-output-len",
        str(DEFAULT_OUTPUT_LEN),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--output-json",
        str(out_file),
    ]
    if attention_backend:
        cmd += ["--attention-backend", attention_backend]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    env = {**os.environ, **(extra_env or {})}

    print(f"\n[L1-offline] Run {run_id}: {' '.join(cmd[:6])}...")
    vram = VRAMMonitor()
    vram.start()

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    elapsed = time.perf_counter() - t0
    peak_vram = vram.stop()

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "unknown")[-2000:]
        print(f"[L1-offline] Run {run_id} FAILED (rc={proc.returncode})")
        print(msg)
        return {"error": msg, "run_id": run_id}

    data: dict[str, Any] = {}
    if out_file.exists():
        data = json.loads(out_file.read_text())

    data["run_id"] = run_id
    data["wall_time_s"] = round(elapsed, 2)
    data["peak_vram_mib"] = peak_vram
    # Re-write with augmented metadata
    out_file.write_text(json.dumps(data, indent=2, default=str))

    tput = data.get("output_tokens_per_second", data.get("tokens_per_second", "?"))
    print(
        f"[L1-offline] Run {run_id}: {tput} tok/s, "
        f"peak VRAM={peak_vram} MiB, elapsed={elapsed:.1f}s"
    )
    return data


def _run_online_benchmark(
    vllm_binary: str,
    model: str,
    *,
    num_prompts: int,
    request_rate: float | str,
    run_id: int,
    tag: str,
    label: str,
    result_dir: Path,
    host: str = "127.0.0.1",
    port: int = SERVER_PORT,
) -> dict[str, Any]:
    """Run ``vllm bench serve`` for a single rate/run combination."""
    rate_str = (
        "inf"
        if request_rate == float("inf") or request_rate == "inf"
        else str(request_rate)
    )
    out_filename = f"experiment-029-{tag}-{label}-rate{rate_str}-run{run_id}.json"

    cmd = [
        vllm_binary,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--random-input-len",
        str(DEFAULT_INPUT_LEN),
        "--random-output-len",
        str(DEFAULT_OUTPUT_LEN),
        "--request-rate",
        rate_str,
        "--num-warmups",
        str(NUM_WARMUPS),
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "25,50,75,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        out_filename,
    ]

    print(f"  [{label}] Run {run_id}, rate={rate_str}: starting")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "unknown")[-2000:]
        print(f"  [{label}] Run {run_id} FAILED (rc={proc.returncode})")
        print(msg)
        return {
            "error": msg,
            "run_id": run_id,
            "request_rate": rate_str,
        }

    out_file = result_dir / out_filename
    data: dict[str, Any] = {}
    if out_file.exists():
        data = json.loads(out_file.read_text())

    data["run_id"] = run_id
    data["request_rate"] = rate_str
    data["wall_time_s"] = round(elapsed, 2)

    tput = data.get("output_throughput", data.get("total_token_throughput", "?"))
    ttft = data.get("median_ttft_ms", "?")
    tpot = data.get("median_tpot_ms", "?")
    print(
        f"  [{label}] Run {run_id}: throughput={tput}, "
        f"TTFT_p50={ttft}ms, TPOT_p50={tpot}ms, elapsed={elapsed:.1f}s"
    )
    return data


# ── Aggregation ───────────────────────────────────────────────────────


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _aggregate_metric(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Median, P25, P75 for *key* across *runs*."""
    values = []
    for r in runs:
        v = r.get(key)
        if v is not None and not isinstance(v, str) and "error" not in r:
            values.append(float(v))
    return {
        "median": round(_safe_median(values), 3),
        "p25": round(_quantile(values, 0.25), 3),
        "p75": round(_quantile(values, 0.75), 3),
        "n": len(values),
        "raw": [round(v, 3) for v in values],
    }


def _aggregate_runs(runs: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    """Aggregate multiple runs into summary statistics."""
    agg: dict[str, Any] = {
        "n_runs": len(runs),
        "n_ok": sum(1 for r in runs if "error" not in r),
    }
    for m in metrics:
        agg[m] = _aggregate_metric(runs, m)
    return agg


# ── Main Pipeline ─────────────────────────────────────────────────────


def _build_extra_env() -> dict[str, str]:
    """Propagate TQ4-specific env vars to subprocesses."""
    env: dict[str, str] = {}
    for var in ("TQ4_USE_FUSED_PAGED", "TQ4_K_BITS", "TQ4_V_BITS"):
        if os.environ.get(var):
            env[var] = os.environ[var]
    return env


def _run_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Orchestrate the full 3-layer benchmark protocol."""
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    health_url = f"http://127.0.0.1:{args.port}/v1"
    extra_env = _build_extra_env()

    results: dict[str, Any] = {
        "experiment": "029-vllm-serving-benchmarks",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": args.tag,
        "model": args.model,
        "config": {
            "attention_backend": args.attention_backend,
            "enforce_eager": args.enforce_eager,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "trust_remote_code": args.trust_remote_code,
            "vllm_binary": args.vllm_binary,
            "runs": args.runs,
            "tq4_use_fused_paged": os.environ.get("TQ4_USE_FUSED_PAGED", "0"),
            "tq4_k_bits": os.environ.get("TQ4_K_BITS"),
            "tq4_v_bits": os.environ.get("TQ4_V_BITS"),
        },
    }

    # ── Layer 1: Offline throughput ──
    if not args.skip_offline:
        print(f"\n{'=' * 60}")
        print("LAYER 1: Offline Throughput")
        print(f"{'=' * 60}")

        offline_runs: list[dict[str, Any]] = []
        for run_id in range(1, args.runs + 1):
            data = _run_offline_throughput(
                args.vllm_binary,
                args.model,
                run_id,
                args.tag,
                result_dir,
                attention_backend=args.attention_backend,
                enforce_eager=args.enforce_eager,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code,
                extra_env=extra_env,
            )
            offline_runs.append(data)

        results["layer1_offline"] = {
            "runs": offline_runs,
            "aggregate": _aggregate_runs(
                offline_runs,
                [
                    "elapsed_time",
                    "num_requests",
                    "total_num_tokens",
                    "requests_per_second",
                    "output_tokens_per_second",
                    "total_tokens_per_second",
                    "peak_vram_mib",
                ],
            ),
        }

    # ── Layers 2 & 3: Online benchmarks (fresh server per run) ──
    if not args.skip_online:
        saturation_runs: list[dict[str, Any]] = []
        sweep_runs: dict[str, list[dict[str, Any]]] = {str(r): [] for r in SWEEP_RATES}
        vram_peaks: list[int] = []

        for run_id in range(1, args.runs + 1):
            print(f"\n{'=' * 60}")
            print(f"ONLINE RUN {run_id}/{args.runs} (fresh server)")
            print(f"{'=' * 60}")

            server = _start_server(
                args.vllm_binary,
                args.model,
                attention_backend=args.attention_backend,
                enforce_eager=args.enforce_eager,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                trust_remote_code=args.trust_remote_code,
                port=args.port,
                extra_env=extra_env,
                log_dir=result_dir,
                tag=args.tag,
                run_id=run_id,
            )

            try:
                _wait_for_health(health_url)
                vram = VRAMMonitor()
                vram.start()

                # Layer 2: Saturation (rate=inf)
                if not args.skip_saturation:
                    print("\n--- Layer 2: Online Saturation (rate=inf) ---")
                    data = _run_online_benchmark(
                        args.vllm_binary,
                        args.model,
                        num_prompts=NUM_PROMPTS_SATURATION,
                        request_rate="inf",
                        run_id=run_id,
                        tag=args.tag,
                        label="L2-sat",
                        result_dir=result_dir,
                        port=args.port,
                    )
                    saturation_runs.append(data)

                # Layer 3: Latency sweep
                if not args.skip_sweep:
                    print("\n--- Layer 3: Latency Sweep ---")
                    for rate in SWEEP_RATES:
                        data = _run_online_benchmark(
                            args.vllm_binary,
                            args.model,
                            num_prompts=NUM_PROMPTS_SWEEP,
                            request_rate=rate,
                            run_id=run_id,
                            tag=args.tag,
                            label="L3-sweep",
                            result_dir=result_dir,
                            port=args.port,
                        )
                        sweep_runs[str(rate)].append(data)

                peak = vram.stop()
                vram_peaks.append(peak)
                print(f"\n[vram] Run {run_id} peak VRAM: {peak} MiB")

            finally:
                _stop_server(server)

        # Store online results
        online_metrics = [
            "request_throughput",
            "output_throughput",
            "total_token_throughput",
            "mean_ttft_ms",
            "median_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "mean_itl_ms",
            "median_itl_ms",
        ]

        if saturation_runs:
            results["layer2_saturation"] = {
                "runs": saturation_runs,
                "aggregate": _aggregate_runs(saturation_runs, online_metrics),
            }

        if any(sweep_runs.values()):
            results["layer3_sweep"] = {}
            for rate_str, runs_at_rate in sweep_runs.items():
                if runs_at_rate:
                    results["layer3_sweep"][f"rate_{rate_str}"] = {
                        "runs": runs_at_rate,
                        "aggregate": _aggregate_runs(runs_at_rate, online_metrics),
                    }

        if vram_peaks:
            valid_peaks = [p for p in vram_peaks if p > 0]
            results["peak_vram_mib"] = {
                "values": vram_peaks,
                "max": max(valid_peaks) if valid_peaks else -1,
                "median": (
                    round(_safe_median([float(v) for v in valid_peaks]), 0)
                    if valid_peaks
                    else -1
                ),
            }

    return results


# ── CLI ───────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for Experiment 029."""
    parser = argparse.ArgumentParser(
        description="Experiment 029: End-to-end vLLM serving benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--tag", required=True, help="Run tag (e.g. 'llama-8b-tq4')")
    parser.add_argument(
        "--vllm-binary",
        default="vllm",
        help="Path to vllm binary (default: 'vllm' from PATH)",
    )
    parser.add_argument("--attention-backend", default=None, help="e.g. CUSTOM for TQ4")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_UTIL,
    )
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Required for VLMs (e.g. Molmo2: 4096+)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Independent runs per config (default: 3)",
    )
    parser.add_argument(
        "--result-dir",
        default="experiments/logs",
        help="Directory for JSON results",
    )
    parser.add_argument(
        "--skip-offline",
        action="store_true",
        help="Skip Layer 1 (offline throughput)",
    )
    parser.add_argument(
        "--skip-online",
        action="store_true",
        help="Skip Layers 2+3 (online benchmarks)",
    )
    parser.add_argument(
        "--skip-saturation",
        action="store_true",
        help="Skip Layer 2 (saturation)",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip Layer 3 (latency sweep)",
    )

    args = parser.parse_args()

    print("Experiment 029: End-to-End vLLM Serving Benchmarks")
    print(f"{'=' * 60}")
    print(f"Model:   {args.model}")
    print(f"Tag:     {args.tag}")
    print(f"Runs:    {args.runs}")
    print(
        f"Config:  backend={args.attention_backend}, eager={args.enforce_eager}, "
        f"gpu_util={args.gpu_memory_utilization}, max_len={args.max_model_len}"
    )
    fused = os.environ.get("TQ4_USE_FUSED_PAGED")
    if fused:
        print(f"Fused:   TQ4_USE_FUSED_PAGED={fused}")
    layers = []
    if not args.skip_offline:
        layers.append("L1-offline")
    if not args.skip_online:
        if not args.skip_saturation:
            layers.append("L2-saturation")
        if not args.skip_sweep:
            layers.append("L3-sweep")
    print(f"Layers:  {', '.join(layers) or 'none'}")
    print(f"{'=' * 60}")

    if not layers:
        print("Nothing to run — all layers skipped.")
        sys.exit(0)

    results = _run_suite(args)

    out_path = Path(args.result_dir) / f"experiment-029-{args.tag}-summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print(f"Summary saved to {out_path}")
    print(f"Per-run JSONs in {args.result_dir}/experiment-029-{args.tag}-*")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
