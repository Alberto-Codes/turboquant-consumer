## Experiment: 006 — AMD ROCm GPU validation on Radeon 890M (gfx1150)

**Date:** 2026-03-26
**Hardware:** Radeon 890M iGPU (32 GB shared DDR5, gfx1150 RDNA 3.5), Ryzen AI 9 HX 370 (12C/24T), 64 GB DDR5
**OS:** Bazzite 43 (immutable Fedora), kernel 6.17.7
**Container:** `rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0` via Podman
**ROCm:** 7.1, HIP 7.1.25424
**Override:** `HSA_OVERRIDE_GFX_VERSION=11.0.0` (gfx1150 → gfx1100)

### Hypothesis

TurboQuant's core quantization pipeline (Lloyd-Max, TurboQuantMSE, CompressedDynamicCache) should produce correct results on AMD Radeon 890M via ROCm with the HSA architecture override, since all operations are standard PyTorch tensor math with no CUDA-specific kernels.

### Setup

- Podman container with GPU passthrough (`/dev/kfd`, `/dev/dri`)
- `--security-opt=label=disable` required (SELinux blocks `hipMalloc`)
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` environment variable
- Tests run on both CPU and GPU, results compared

### Results

#### Phase 0: GPU Detection

| Check | Result |
|-------|--------|
| `torch.cuda.is_available()` | True |
| Device name | AMD Radeon Graphics |
| VRAM reported | 32.0 GB |
| GCN arch (spoofed) | gfx1100 |
| Compute units | 8 |
| HIP version | 7.1.25424 |

#### Phase 0.4: Basic GPU Compute

| Test | Result |
|------|--------|
| 1000x1000 matmul | Correct |
| 2000x2000 matmul | Correct |
| CPU/GPU agreement (atol=1e-4) | True |
| Max absolute difference | 0.004 (normal fp divergence) |

#### Phase 0.5: Test Suite on CPU in Container

| Result | Count |
|--------|-------|
| Passed | 62/62 |
| Time | 12.4s |

#### Phase 0.6: Cross-Device GPU Correctness

| Operation | CPU vs GPU | Detail |
|-----------|-----------|--------|
| Lloyd-Max quantize | **Bit-identical** | All indices match exactly |
| Lloyd-Max dequantize | **Exact match** | atol=1e-6 |
| TurboQuantMSE indices | **Bit-identical** | All indices match exactly |
| TurboQuantMSE norms | **Match** | atol=1e-5 |
| TurboQuantMSE decompress | **cosine 1.000002** | Effectively identical |
| CompressedDynamicCache keys | **cosine 0.9951** | Expected from lossy quantization |
| CompressedDynamicCache values | **cosine 0.9952** | Expected from lossy quantization |
| 8-layer 512-token stress test | **Clean** | No NaN, no Inf, 76.6 MB GPU |

### Key Findings

1. **SELinux was the blocker, not ROCm.** All HSA override values crashed
   with `Memory critical error — Memory in use` until `--security-opt=label=disable`
   was added to the Podman command. This is specific to Bazzite/Fedora immutable
   distributions with SELinux enforcing. The ROCm stack itself works correctly.

2. **HSA override introduces zero quantization error.** Lloyd-Max indices and
   TurboQuantMSE indices are bit-identical between CPU and GPU. The gfx1150 →
   gfx1100 architecture spoof does not affect computation correctness for
   standard PyTorch tensor operations.

3. **Cache cosine similarities match RTX 4090 results.** The 0.995 key/value
   cosine similarities from CompressedDynamicCache on the 890M match what we
   see on the 4090 — the compression loss is from TurboQuant's quantization,
   not from the GPU platform.

4. **ROCm 7.11 preview has native gfx1150 wheels.** Available at
   `https://repo.amd.com/rocm/whl/gfx1150/` — installs `torch-2.9.1+rocm7.11.0`
   with `rocm-sdk-libraries-gfx1150-7.11.0`. Not tested for compute correctness
   yet (the 7.1 + override path was validated first).

5. **Memory architecture confirmed.** 32 GB shared VRAM from DDR5 system RAM.
   ~90 GB/s bandwidth vs ~1 TB/s on 4090. Expect 11-16x slower for
   bandwidth-bound operations.

### Blocker Resolution Log

| Attempt | Override | Container Flag | Result |
|---------|----------|---------------|--------|
| 1 | 11.0.0 | (default) | `Memory critical error` crash |
| 2 | 11.0.2 | (default) | Crash |
| 3 | 11.0.1 | (default) | Crash |
| 4 | 11.5.0 | (default) | Crash |
| 5 | 11.5.1 | (default) | Crash |
| 6 | 11.0.0 | `--security-opt=label=disable` | **Works** |

### Working Podman Command

```bash
podman run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  --security-opt=label=disable \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -v ~/Projects/turboquant-consumer:/workspace:z \
  -w /workspace \
  docker.io/rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.8.0
```

### Next Steps

1. **Create `Containerfile`** — codify the working setup for reproducibility.
2. **Add cross-device pytest fixtures** — permanent CPU vs GPU validation in CI.
3. **Explore `torch.compile(mode="default")`** — potential free speedups on ROCm.
4. **End-to-end Molmo2-4B inference** — full model inference on the 890M.
