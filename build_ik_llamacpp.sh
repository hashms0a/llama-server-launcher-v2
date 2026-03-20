#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

REPO_URL="https://github.com/ikawrakow/ik_llama.cpp"
REPO_DIR="ik_llama.cpp"

# --- 1. Ask the user to install dependencies ---
echo "----------------------------------------------------------------"
echo "Build Environment Setup (ik_llama.cpp)"
echo "----------------------------------------------------------------"
echo "The following dependencies are required to build ik_llama.cpp:"
echo " - build-essential"
echo " - git"
echo " - cmake"
echo " - curl"
echo " - libcurl4-openssl-dev"
echo " - libgomp1"
echo " - pciutils"
echo ""

read -p "Do you want to run 'sudo apt update' and install these dependencies now? (y/n) " -n 1 -r
echo    # move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Updating package lists..."
    sudo apt update

    echo "Installing dependencies..."
    sudo apt install -y build-essential git cmake curl libcurl4-openssl-dev libgomp1 pciutils
    echo "Dependencies installed."
else
    echo "Skipping dependency installation. Ensure you have the necessary tools installed."
fi

# --- 1b. Ensure CMake >= 3.25 (required for CUDA C++20 support) ---
CMAKE_MIN_MAJOR=3
CMAKE_MIN_MINOR=25

CURRENT_CMAKE_VER=$(cmake --version 2>/dev/null | head -n1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
CMAKE_MAJOR=$(echo "$CURRENT_CMAKE_VER" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CURRENT_CMAKE_VER" | cut -d. -f2)

if [ -z "$CURRENT_CMAKE_VER" ] || [ "$CMAKE_MAJOR" -lt "$CMAKE_MIN_MAJOR" ] || \
   { [ "$CMAKE_MAJOR" -eq "$CMAKE_MIN_MAJOR" ] && [ "$CMAKE_MINOR" -lt "$CMAKE_MIN_MINOR" ]; }; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "CMake Upgrade Required"
    echo "----------------------------------------------------------------"
    echo "ik_llama.cpp with CUDA needs CMake >= 3.25 (CUDA C++20 support)."
    echo "Detected: ${CURRENT_CMAKE_VER:-none}"
    echo ""
    read -p "Install latest CMake from the Kitware APT repository? (y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt install -y ca-certificates gpg wget
        wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
            gpg --dearmor | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

        CODENAME=$(lsb_release -cs)
        echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $CODENAME main" | \
            sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

        sudo apt update
        sudo apt install -y cmake
        echo "CMake upgraded to: $(cmake --version | head -n1)"
    else
        echo "ERROR: CMake >= 3.25 is required. Please upgrade manually."
        exit 1
    fi
else
    echo ""
    echo "CMake version OK: $CURRENT_CMAKE_VER (>= 3.25)"
fi

# --- 2. Clone the repository ---
echo ""
echo "----------------------------------------------------------------"
echo "Cloning ik_llama.cpp..."
echo "----------------------------------------------------------------"

if [ -d "$REPO_DIR" ]; then
    echo "Directory '$REPO_DIR' already exists. Skipping clone."
    # Optional: git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL"
fi

# --- 3. Configure CMake (GPU / CUDA) ---
echo ""
echo "----------------------------------------------------------------"
echo "Configuring CMake (GPU build with CUDA)..."
echo "----------------------------------------------------------------"

# Remove stale build directory from a previous failed configure
if [ -d "$REPO_DIR/build" ]; then
    echo "Removing previous build directory..."
    rm -rf "$REPO_DIR/build"
fi
# Flags:
# -DGGML_NATIVE=ON:  Enable native CPU optimizations (AVX2, etc.)
# -DGGML_CUDA=ON:    Enable CUDA support (requires NVIDIA drivers + CUDA Toolkit)
cmake "$REPO_DIR" -B "$REPO_DIR/build" \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=ON

# --- 4. Build the project ---
echo ""
echo "----------------------------------------------------------------"
echo "Select Build Option"
echo "----------------------------------------------------------------"
echo "1. Build everything (Default)"
echo "2. Build only llama-cli, llama-gguf-split, and llama-server"
echo ""

read -p "Enter your choice (1 or 2): " build_choice

echo ""
echo "----------------------------------------------------------------"
echo "Building project (Release mode)..."
echo "----------------------------------------------------------------"

if [[ "$build_choice" == "2" ]]; then
    cmake --build "$REPO_DIR/build" --config Release -j$(nproc) --clean-first --target llama-cli llama-gguf-split llama-server
else
    cmake --build "$REPO_DIR/build" --config Release -j$(nproc) --clean-first
fi

# --- 5. Copy the binaries ---
echo ""
echo "----------------------------------------------------------------"
echo "Copying binaries..."
echo "----------------------------------------------------------------"

targets=("llama-cli" "llama-gguf-split" "llama-server")

for target in "${targets[@]}"; do
    if [ -f "$REPO_DIR/build/bin/$target" ]; then
        cp "$REPO_DIR/build/bin/$target" "$REPO_DIR/"
        echo "Success! '$target' has been copied to the '$REPO_DIR' folder."
    else
        echo "Warning: Binary '$target' not found in build output."
    fi
done

echo ""
echo "Build complete."
