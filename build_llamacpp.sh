#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# --- 1. Ask the user to install dependencies ---
echo "----------------------------------------------------------------"
echo "Build Environment Setup"
echo "----------------------------------------------------------------"
echo "The following dependencies are required to build llama.cpp:"
echo " - pciutils"
echo " - build-essential"
echo " - cmake"
echo " - curl"
echo " - libcurl4-openssl-dev"
echo ""

read -p "Do you want to run 'sudo apt update' and install these dependencies now? (y/n) " -n 1 -r
echo    # move to a new line

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Updating package lists..."
    sudo apt update
    
    echo "Installing dependencies..."
    sudo apt install pciutils build-essential cmake curl libcurl4-openssl-dev -y
    echo "Dependencies installed."
else
    echo "Skipping dependency installation. Ensure you have the necessary tools installed."
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
    git clone https://github.com/ggml-org/llama.cpp
fi

# --- 3. Configure CMake ---
echo ""
echo "----------------------------------------------------------------"
echo "Configuring CMake..."
echo "----------------------------------------------------------------"
# Flags:
# -DBUILD_SHARED_LIBS=OFF: Build static libraries
# -DGGML_CUDA=ON: Enable CUDA support (Requires NVCC/Nvidia Toolkit)
# -DLLAMA_CURL=ON: Enable CURL support for downloading models
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON

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
    # Option 2: Build specific targets
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split llama-server
else
    # Option 1: Build everything (Default)
    cmake --build llama.cpp/build --config Release -j --clean-first
fi

# --- 5. Copy the binaries ---
echo ""
echo "----------------------------------------------------------------"
echo "Copying binaries..."
echo "----------------------------------------------------------------"

# Define the targets to copy
targets=("llama-cli" "llama-gguf-split" "llama-server")

for target in "${targets[@]}"; do
    if [ -f "llama.cpp/build/bin/$target" ]; then
        cp "llama.cpp/build/bin/$target" llama.cpp/
        echo "Success! '$target' has been copied to the 'llama.cpp' folder."
    else
        echo "Warning: Binary '$target' not found in build output."
    fi
done

echo ""
echo "Build complete."
