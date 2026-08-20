#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# ================================================================
# llama.cpp build script for AMD GPUs (ROCm/HIP + Vulkan)
# Target system: Ubuntu 26.04 LTS "Resolute Raccoon"
# Target GPU:    Radeon RX 7900 XTX (RDNA3, gfx1100, 24 GB)
# ================================================================

# ---------------- User-tunable settings ----------------

# GPU architecture to compile HIP kernels for. gfx1100 = RX 7900 XTX/XT/GRE.
# Leave as "auto" to detect it from the installed GPU at build time.
GPU_TARGET="auto"
GPU_TARGET_FALLBACK="gfx1100"

# Parallel compile jobs. HIP kernel compilation is RAM-hungry (~1.5-2 GB/job),
# so lower this if the build gets OOM-killed.
JOBS="$(nproc)"

# 1 = wipe object files before building (slow, matches a fresh build)
# 0 = incremental rebuild (much faster when re-running this script)
CLEAN_FIRST=1

# Build static binaries (no libggml*.so next to the executables).
BUILD_SHARED_LIBS="OFF"

# rocWMMA-accelerated flash attention. Needs the rocwmma-dev package.
# Off by default because it is not always packaged; turn on if you have it.
ROCWMMA_FATTN="OFF"

REPO_BRANCH="master"
TARBALL_URL="https://github.com/ggml-org/llama.cpp/archive/refs/heads/${REPO_BRANCH}.tar.gz"

# --- Helper: prompt with a visible per-second countdown ---
# Usage: read_with_countdown <seconds> <prompt_text>
# Captures a single keypress into the global REPLY_INPUT.
# Returns 0 if a key was pressed, 1 if it timed out.
read_with_countdown() {
    local timeout="$1"
    local prompt="$2"
    local remaining="$timeout"
    local input=""
    REPLY_INPUT=""

    while [ "$remaining" -gt 0 ]; do
        # \r returns to start of line so the countdown overwrites in place.
        # Trailing spaces clear any leftover characters.
        printf "\r%s [%ds] " "$prompt" "$remaining"
        if read -t 1 -n 1 -r input; then
            REPLY_INPUT="$input"
            printf "\n"
            return 0
        fi
        remaining=$((remaining - 1))
    done
    printf "\r%s [0s] \n" "$prompt"
    return 1   # timed out
}

# --- 0. Choose which backend(s) to build ---
echo "----------------------------------------------------------------"
echo "Select Backend"
echo "----------------------------------------------------------------"
echo "1. ROCm / HIP only     (best throughput on RDNA3, needs ROCm installed)"
echo "2. Vulkan only         (driver-only, works without the ROCm stack)"
echo "3. Both, in separate build trees (Default)"
echo ""

if read_with_countdown 5 "Enter your choice (1, 2 or 3), defaults to '3'"; then
    backend_choice="$REPLY_INPUT"
else
    echo "No input received within 5 seconds. Defaulting to option 3."
    backend_choice="3"
fi
[[ -z "$backend_choice" ]] && backend_choice="3"

case "$backend_choice" in
    1) BUILD_ROCM=1; BUILD_VULKAN=0 ;;
    2) BUILD_ROCM=0; BUILD_VULKAN=1 ;;
    *) BUILD_ROCM=1; BUILD_VULKAN=1 ;;
esac

echo ""
echo "ROCm build:   $([ "$BUILD_ROCM" -eq 1 ] && echo yes || echo no)"
echo "Vulkan build: $([ "$BUILD_VULKAN" -eq 1 ] && echo yes || echo no)"

# --- 1. Check dependencies and install if needed ---
echo ""
echo "----------------------------------------------------------------"
echo "Build Environment Setup"
echo "----------------------------------------------------------------"

# Common toolchain packages
DEPENDENCIES=(pciutils build-essential cmake git curl libcurl4-openssl-dev)

# Vulkan SDK bits: headers, loader, the glslc shader compiler that ggml-vulkan
# needs at build time, and the Mesa RADV driver used at runtime.
if [ "$BUILD_VULKAN" -eq 1 ]; then
    DEPENDENCIES+=(libvulkan-dev vulkan-tools glslc spirv-tools mesa-vulkan-drivers)
fi

MISSING_DEPS=()
for dep in "${DEPENDENCIES[@]}"; do
    if dpkg -s "$dep" >/dev/null 2>&1; then
        echo " [ OK ]      $dep"
    else
        echo " [ MISSING ] $dep"
        MISSING_DEPS+=("$dep")
    fi
done

# ROCm is checked by tooling rather than package name, because the stack can
# come either from the Ubuntu archive (26.04 ships ROCm 7.1 in universe) or
# from AMD's repo.radeon.com packages, which use different package names.
if [ "$BUILD_ROCM" -eq 1 ]; then
    if command -v hipconfig >/dev/null 2>&1 && command -v rocminfo >/dev/null 2>&1; then
        echo " [ OK ]      ROCm / HIP toolchain ($(hipconfig --version 2>/dev/null | head -n1))"
    else
        echo " [ MISSING ] ROCm / HIP toolchain"
        # Prefer the leanest meta-package that is actually available.
        ROCM_PKG=""
        for candidate in rocm-hip-sdk rocm-dev rocm; do
            if apt-cache show "$candidate" >/dev/null 2>&1; then
                ROCM_PKG="$candidate"
                break
            fi
        done
        if [ -n "$ROCM_PKG" ]; then
            echo "             -> will install '$ROCM_PKG'"
            MISSING_DEPS+=("$ROCM_PKG")
        else
            echo "             -> no ROCm package found in your apt sources."
            echo "                On Ubuntu 26.04 'rocm' lives in the universe component;"
            echo "                enable it with: sudo add-apt-repository universe"
        fi
    fi
fi
echo ""

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "All required dependencies are already installed. Skipping installation."
else
    echo "The following required dependencies are missing:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo " - $dep"
    done
    echo ""
    echo "Note: the ROCm stack is large (several GB) and may take a while."
    echo ""

    # Prompt with a 5 second visible countdown. If no 'y' is given
    # (timeout or other key), skip the installation.
    if read_with_countdown 5 "Run 'sudo apt update' and install the missing dependencies now? (y/n)"; then
        :
    else
        echo "No input received within 5 seconds. Defaulting to skip."
    fi

    if [[ $REPLY_INPUT =~ ^[Yy]$ ]]; then
        echo "Updating package lists..."
        sudo apt update

        echo "Installing dependencies..."
        sudo apt install "${MISSING_DEPS[@]}" -y
        echo "Dependencies installed."
    else
        echo "Skipping dependency installation. Ensure you have the necessary tools installed."
    fi
fi

# --- 1b. GPU access sanity checks ---
if [ "$BUILD_ROCM" -eq 1 ]; then
    echo ""
    echo "Checking GPU device access..."
    # The Ubuntu archive packages set up permissions automatically, but AMD's
    # own packages still expect membership in render/video.
    for grp in render video; do
        if getent group "$grp" >/dev/null 2>&1 && ! id -nG "$USER" | grep -qw "$grp"; then
            echo " [ WARN ] $USER is not in the '$grp' group."
            echo "          Fix with: sudo usermod -aG $grp $USER   (then log out and back in)"
        fi
    done
    if command -v rocminfo >/dev/null 2>&1; then
        if rocminfo >/dev/null 2>&1; then
            echo " [ OK ]   rocminfo can talk to the GPU."
        else
            echo " [ WARN ] rocminfo failed. The amdgpu kernel driver may not be loaded,"
            echo "          or the current user lacks /dev/kfd access. The HIP build will"
            echo "          still compile, but nothing will run until this is resolved."
        fi
    fi
fi

# --- 1c. Resolve the HIP GPU target ---
if [ "$BUILD_ROCM" -eq 1 ] && [ "$GPU_TARGET" = "auto" ]; then
    detected=""
    if command -v amdgpu-arch >/dev/null 2>&1; then
        detected="$(amdgpu-arch 2>/dev/null | head -n1 || true)"
    fi
    if [ -z "$detected" ] && command -v rocminfo >/dev/null 2>&1; then
        detected="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]\+' || true)"
    fi
    if [ -n "$detected" ]; then
        GPU_TARGET="$detected"
        echo " [ OK ]   Detected GPU architecture: $GPU_TARGET"
    else
        GPU_TARGET="$GPU_TARGET_FALLBACK"
        echo " [ INFO ] Could not detect GPU architecture. Falling back to $GPU_TARGET (RX 7900 XTX)."
    fi
fi

# --- 2. Clone the repository ---
echo ""
echo "----------------------------------------------------------------"
echo "Cloning llama.cpp..."
echo "----------------------------------------------------------------"

if [ -d "llama.cpp" ]; then
    echo "Directory 'llama.cpp' already exists. Skipping clone."
    # Optional: git -C llama.cpp pull
else
    echo "1. git clone (full history, needed to contribute or check out old commits) (Default)"
    echo "2. Download snapshot tarball (no git history, usually much faster)"
    echo ""

    # Prompt with a 5 second visible countdown. If no input is given (timeout)
    # or Enter is pressed, use the default (option 1 = git clone).
    if read_with_countdown 5 "Enter your choice (1 or 2), defaults to '1'"; then
        clone_choice="$REPLY_INPUT"
    else
        echo "No input received within 5 seconds. Defaulting to option 1."
        clone_choice="1"
    fi

    # Treat empty input (e.g. just pressing Enter) as the default as well
    if [[ -z "$clone_choice" ]]; then
        clone_choice="1"
    fi

    echo ""

    if [[ "$clone_choice" == "2" ]]; then
        echo "Downloading snapshot from $TARBALL_URL ..."

        # Prefer wget, fall back to curl (curl is already a listed dependency).
        if command -v wget >/dev/null 2>&1; then
            wget -O llama.cpp.tar.gz "$TARBALL_URL"
        elif command -v curl >/dev/null 2>&1; then
            curl -L -o llama.cpp.tar.gz "$TARBALL_URL"
        else
            echo "Error: neither wget nor curl is available to download the tarball."
            exit 1
        fi

        echo "Extracting..."
        # The archive unpacks to llama.cpp-<branch>/, so rename it afterwards.
        rm -rf "llama.cpp-${REPO_BRANCH}"
        tar -xzf llama.cpp.tar.gz
        mv "llama.cpp-${REPO_BRANCH}" llama.cpp
        rm -f llama.cpp.tar.gz

        echo "Snapshot ready in 'llama.cpp' (no git history)."
    else
        git clone https://github.com/ggml-org/llama.cpp
    fi
fi

# --- 3. Pick which targets to build ---
echo ""
echo "----------------------------------------------------------------"
echo "Select Build Option"
echo "----------------------------------------------------------------"
echo "1. Build everything"
echo "2. Build only llama-cli, llama-gguf-split, llama-server, llama-bench (Default)"
echo ""

if read_with_countdown 5 "Enter your choice (1 or 2), defaults to '2'"; then
    build_choice="$REPLY_INPUT"
else
    echo "No input received within 5 seconds. Defaulting to option 2."
    build_choice="2"
fi
[[ -z "$build_choice" ]] && build_choice="2"

TARGETS=(llama-cli llama-gguf-split llama-server llama-bench)

CLEAN_FLAG=()
[ "$CLEAN_FIRST" -eq 1 ] && CLEAN_FLAG=(--clean-first)

# --- 4a. ROCm / HIP build ---
if [ "$BUILD_ROCM" -eq 1 ]; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "Configuring CMake (ROCm / HIP, target $GPU_TARGET)..."
    echo "----------------------------------------------------------------"

    # Locate the ROCm installation and its bundled LLVM. ggml's HIP backend must
    # be compiled with ROCm's own clang, not the system clang.
    ROCM_PATH="$(hipconfig -R 2>/dev/null || true)"
    [ -z "$ROCM_PATH" ] && ROCM_PATH="/opt/rocm"
    HIP_LLVM_BIN="$(hipconfig -l 2>/dev/null || true)"
    [ -z "$HIP_LLVM_BIN" ] && HIP_LLVM_BIN="${ROCM_PATH}/llvm/bin"

    if [ ! -x "${HIP_LLVM_BIN}/clang" ]; then
        echo "Error: ROCm clang not found at ${HIP_LLVM_BIN}/clang"
        echo "       Install the ROCm stack (e.g. 'sudo apt install rocm') and re-run."
        exit 1
    fi
    echo "Using ROCM_PATH=$ROCM_PATH"
    echo "Using HIP compiler: ${HIP_LLVM_BIN}/clang"

    # Flags:
    # -DGGML_HIP=ON: enable the HIP/ROCm backend (replaces the old GGML_HIPBLAS)
    # -DGPU_TARGETS: the gfx arch to emit code for; gfx1100 = RX 7900 XTX
    # -DBUILD_SHARED_LIBS=OFF: build static libraries
    # -DLLAMA_CURL=ON: enable CURL support for downloading models
    HIPCXX="${HIP_LLVM_BIN}/clang" HIP_PATH="${ROCM_PATH}" \
    cmake llama.cpp -B llama.cpp/build-rocm \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIP=ON \
        -DGPU_TARGETS="$GPU_TARGET" \
        -DAMDGPU_TARGETS="$GPU_TARGET" \
        -DGGML_HIP_ROCWMMA_FATTN="$ROCWMMA_FATTN" \
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
        -DLLAMA_CURL=ON

    echo ""
    echo "----------------------------------------------------------------"
    echo "Building ROCm backend (Release, -j $JOBS)..."
    echo "----------------------------------------------------------------"

    if [[ "$build_choice" == "1" ]]; then
        cmake --build llama.cpp/build-rocm --config Release -j "$JOBS" "${CLEAN_FLAG[@]}"
    else
        cmake --build llama.cpp/build-rocm --config Release -j "$JOBS" "${CLEAN_FLAG[@]}" --target "${TARGETS[@]}"
    fi
fi

# --- 4b. Vulkan build ---
if [ "$BUILD_VULKAN" -eq 1 ]; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "Configuring CMake (Vulkan)..."
    echo "----------------------------------------------------------------"

    if ! command -v glslc >/dev/null 2>&1; then
        echo "Error: 'glslc' not found. The Vulkan backend needs it to compile shaders."
        echo "       Install it with: sudo apt install glslc"
        exit 1
    fi
    if command -v vulkaninfo >/dev/null 2>&1; then
        if vulkaninfo --summary >/dev/null 2>&1; then
            echo " [ OK ]   Vulkan loader sees a device."
        else
            echo " [ WARN ] vulkaninfo found no working device. Check that mesa-vulkan-drivers"
            echo "          is installed and the amdgpu driver is loaded."
        fi
    fi

    # Flags:
    # -DGGML_VULKAN=ON: enable the Vulkan compute backend (RADV on this card)
    cmake llama.cpp -B llama.cpp/build-vulkan \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON \
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
        -DLLAMA_CURL=ON

    echo ""
    echo "----------------------------------------------------------------"
    echo "Building Vulkan backend (Release, -j $JOBS)..."
    echo "----------------------------------------------------------------"

    if [[ "$build_choice" == "1" ]]; then
        cmake --build llama.cpp/build-vulkan --config Release -j "$JOBS" "${CLEAN_FLAG[@]}"
    else
        cmake --build llama.cpp/build-vulkan --config Release -j "$JOBS" "${CLEAN_FLAG[@]}" --target "${TARGETS[@]}"
    fi
fi

# --- 5. Copy the binaries ---
echo ""
echo "----------------------------------------------------------------"
echo "Copying binaries..."
echo "----------------------------------------------------------------"

copy_binaries() {
    local build_dir="$1"
    local dest="$2"

    mkdir -p "$dest"
    for target in "${TARGETS[@]}"; do
        if [ -f "${build_dir}/bin/${target}" ]; then
            cp "${build_dir}/bin/${target}" "$dest/"
            echo "Success! '$target' has been copied to '$dest'."
        else
            echo "Warning: Binary '$target' not found in ${build_dir}/bin."
        fi
    done
    # Static builds have no shared ggml libs; copy them if this was a shared build.
    if [ "$BUILD_SHARED_LIBS" = "ON" ]; then
        cp "${build_dir}"/bin/*.so "$dest/" 2>/dev/null || true
    fi
}

[ "$BUILD_ROCM" -eq 1 ]   && copy_binaries "llama.cpp/build-rocm"   "llama.cpp/bin-rocm"
[ "$BUILD_VULKAN" -eq 1 ] && copy_binaries "llama.cpp/build-vulkan" "llama.cpp/bin-vulkan"

# --- 6. Verify ---
echo ""
echo "----------------------------------------------------------------"
echo "Verifying GPU detection..."
echo "----------------------------------------------------------------"

if [ "$BUILD_ROCM" -eq 1 ] && [ -x "llama.cpp/bin-rocm/llama-cli" ]; then
    echo "ROCm backend:"
    ./llama.cpp/bin-rocm/llama-cli --list-devices 2>&1 | sed 's/^/  /' || \
        echo "  (device listing failed - check the warnings above)"
fi

if [ "$BUILD_VULKAN" -eq 1 ] && [ -x "llama.cpp/bin-vulkan/llama-cli" ]; then
    echo "Vulkan backend:"
    ./llama.cpp/bin-vulkan/llama-cli --list-devices 2>&1 | sed 's/^/  /' || \
        echo "  (device listing failed - check the warnings above)"
fi

echo ""
echo "Build complete."
echo ""
echo "Run a model with, for example:"
[ "$BUILD_ROCM" -eq 1 ]   && echo "  ./llama.cpp/bin-rocm/llama-server   -m model.gguf -ngl 99 -c 8192"
[ "$BUILD_VULKAN" -eq 1 ] && echo "  ./llama.cpp/bin-vulkan/llama-server -m model.gguf -ngl 99 -c 8192"
[ "$BUILD_ROCM" -eq 1 ] && [ "$BUILD_VULKAN" -eq 1 ] && \
    echo "  Compare them with llama-bench from each directory on the same GGUF."
