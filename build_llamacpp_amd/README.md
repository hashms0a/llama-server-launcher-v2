# build_llamacpp_amd.sh

Build [llama.cpp](https://github.com/ggml-org/llama.cpp) from source with **ROCm/HIP** and/or **Vulkan** acceleration on AMD GPUs.

Written for **Ubuntu 26.04 LTS** and a **Radeon RX 7900 XTX** (RDNA3, `gfx1100`, 24 GB), but it auto-detects the GPU architecture and works on other AMD cards and other Debian-based releases.

---

## Quick start

```bash
chmod +x build_llamacpp_amd.sh
./build_llamacpp_amd.sh
```

Every prompt has a 5-second countdown and a sensible default, so you can also just start it and walk away.

---

## What it does

1. Asks which backend(s) to build — ROCm, Vulkan, or both (default).
2. Checks system dependencies and offers to install what's missing.
3. Verifies GPU access (`render`/`video` groups, `rocminfo`, `vulkaninfo`).
4. Detects the GPU architecture (`gfx1100` on the 7900 XTX).
5. Fetches llama.cpp via `git clone` or a snapshot tarball.
6. Configures and builds each backend in its own tree.
7. Copies the binaries into `bin-rocm/` and `bin-vulkan/`.
8. Runs `llama-cli --list-devices` to confirm the GPU is visible.

---

## Dependencies

### Checked automatically

The script probes every item below and offers to install anything missing. You do **not** need to install these by hand.

#### Always required

| Package | Why |
|---|---|
| `build-essential` | GCC/G++ and make |
| `cmake` | build system |
| `git` | cloning the repository |
| `curl` | tarball fallback download |
| `libcurl4-openssl-dev` | `-DLLAMA_CURL=ON`, lets `llama-server`/`llama-cli` pull GGUFs with `-hf` |
| `pciutils` | GPU identification |

#### ROCm / HIP backend

| Package | Why |
|---|---|
| `rocm` (or `rocm-hip-sdk` / `rocm-dev`) | HIP compiler, hipBLAS/rocBLAS, `rocminfo`, `rocm-smi` |

Detected by looking for the `hipconfig` and `rocminfo` commands rather than by package name, so an existing AMD-repo install is recognised correctly.

> **Ubuntu 26.04 note:** ROCm now ships in the Ubuntu archive (universe component), so `sudo apt install rocm` is all that's needed — no `amdgpu-install`, no external repo, and permissions are configured by the packages. The archive version is ROCm 7.1, which trails upstream by a few point releases. If you need newer kernels, install from AMD's official instructions instead; the script detects either.

Expect several GB of downloads and disk usage for the ROCm stack.

#### Vulkan backend

| Package | Why |
|---|---|
| `libvulkan-dev` | Vulkan headers and loader dev files |
| `glslc` | **required** — compiles ggml's compute shaders at build time |
| `spirv-tools` | SPIR-V utilities |
| `vulkan-tools` | `vulkaninfo`, used for verification |
| `mesa-vulkan-drivers` | RADV, the runtime driver for Radeon cards |

### Not checked automatically

- **`amdgpu` kernel driver** — in-tree on Ubuntu 26.04, nothing to install. If `rocminfo` reports no agents, this is the first thing to look at.
- **Group membership** — the script warns if you're missing `render` or `video` but won't change it for you:
  ```bash
  sudo usermod -aG render,video $USER   # then log out and back in
  ```
  Not needed with the Ubuntu archive ROCm packages; still needed with AMD's own packages.
- **`rocwmma-dev`** — only if you enable `ROCWMMA_FATTN` (see below).
- **Disk space** — budget ~10 GB for ROCm plus a few GB for the build trees.

### Manual install (if you'd rather not use the prompt)

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl libcurl4-openssl-dev pciutils
sudo apt install -y rocm                                    # ROCm backend
sudo apt install -y libvulkan-dev glslc spirv-tools \
                    vulkan-tools mesa-vulkan-drivers        # Vulkan backend
```

---

## Configuration

Variables at the top of the script:

| Variable | Default | Meaning |
|---|---|---|
| `GPU_TARGET` | `auto` | HIP architecture. `auto` detects via `amdgpu-arch`/`rocminfo`; set explicitly (e.g. `gfx1100`) to skip detection |
| `GPU_TARGET_FALLBACK` | `gfx1100` | Used when detection fails — RX 7900 XTX/XT/GRE |
| `JOBS` | `$(nproc)` | Parallel compile jobs. HIP uses ~1.5–2 GB RAM per job; lower this if the build gets OOM-killed |
| `CLEAN_FIRST` | `1` | `1` wipes objects first; set `0` for much faster re-runs |
| `BUILD_SHARED_LIBS` | `OFF` | Static binaries, no `libggml*.so` alongside them |
| `ROCWMMA_FATTN` | `OFF` | rocWMMA flash attention. Needs `rocwmma-dev`; can speed up long-context work on RDNA3 |
| `REPO_BRANCH` | `master` | Branch used for the tarball download |

---

## Prompts

**Backend** (default 3)
1. ROCm/HIP only
2. Vulkan only
3. Both, in separate build trees

**Source** (default 1)
1. `git clone` — full history
2. Snapshot tarball — faster, no history

**Targets** (default 2)
1. Everything
2. `llama-cli`, `llama-gguf-split`, `llama-server`, `llama-bench`

---

## Output layout

```
llama.cpp/
├── build-rocm/     # ROCm CMake tree
├── build-vulkan/   # Vulkan CMake tree
├── bin-rocm/       # llama-cli, llama-server, llama-bench, llama-gguf-split
└── bin-vulkan/     # same set, Vulkan build
```

The two backends are built separately on purpose. Compiling both into one binary works, but the same GPU then enumerates twice, and separate trees let you benchmark one against the other and keep a working fallback if a ROCm upgrade breaks something.

---

## Running

```bash
./llama.cpp/bin-rocm/llama-server   -m model.gguf -ngl 99 -c 8192
./llama.cpp/bin-vulkan/llama-server -m model.gguf -ngl 99 -c 8192
```

`-ngl 99` offloads all layers to the GPU. With 24 GB you can fully offload most 24B–32B models at Q4_K_M, or a 70B at low quant with partial offload.

Compare the two backends on identical work:

```bash
./llama.cpp/bin-rocm/llama-bench   -m model.gguf -ngl 99
./llama.cpp/bin-vulkan/llama-bench -m model.gguf -ngl 99
```

On RDNA3, ROCm is typically ahead on prompt processing while Vulkan is often competitive on token generation.

Useful environment variables:

- `HIP_VISIBLE_DEVICES=0` — restrict which GPU the ROCm build uses
- `GGML_VK_VISIBLE_DEVICES=0` — same idea for Vulkan
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` — only needed for unsupported RDNA3 cards; **not** required for the 7900 XTX

---

## Troubleshooting

**`ROCm clang not found`** — the ROCm stack isn't installed or isn't on `PATH`. Check `hipconfig -R`; install with `sudo apt install rocm`.

**`rocminfo` finds no agents** — the amdgpu driver isn't loaded or you lack `/dev/kfd` access. Check `lsmod | grep amdgpu` and your group membership, then reboot.

**Build is OOM-killed** — lower `JOBS`. HIP template instantiation is memory-hungry.

**`glslc: command not found`** — `sudo apt install glslc`.

**`vulkaninfo` shows no device** — install `mesa-vulkan-drivers` and confirm the amdgpu driver is loaded.

**GPU not listed at runtime** — run `llama-cli --list-devices` from the relevant `bin-*` directory; it prints what each backend can actually see.

**Rebuilds take forever** — set `CLEAN_FIRST=0`.
