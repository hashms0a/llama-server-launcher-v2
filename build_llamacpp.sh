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
echo "Building project (Release mode)..."
echo "----------------------------------------------------------------"
# -j: Uses all available cores
# --clean-first: Clean build directory before building
cmake --build llama.cpp/build --config Release -j --clean-first

# --- 5. Copy the binary ---
echo ""
echo "----------------------------------------------------------------"
echo "Copying binary..."
echo "----------------------------------------------------------------"

if [ -f "llama.cpp/build/bin/llama-server" ]; then
    cp llama.cpp/build/bin/llama-server llama.cpp/
    echo "Success! 'llama-server' has been copied to the 'llama.cpp' folder."
else
    echo "Error: Binary 'llama-server' not found in build output."
    exit 1
fi

echo ""
echo "Build complete."
