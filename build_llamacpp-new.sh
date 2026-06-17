#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

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

# --- 1. Check dependencies and install if needed ---
echo "----------------------------------------------------------------"
echo "Build Environment Setup"
echo "----------------------------------------------------------------"

# List of required dependencies (apt package names)
DEPENDENCIES=(pciutils build-essential cmake curl libcurl4-openssl-dev)

# Determine which dependencies are missing using dpkg
MISSING_DEPS=()
for dep in "${DEPENDENCIES[@]}"; do
    if dpkg -s "$dep" >/dev/null 2>&1; then
        echo " [ OK ]      $dep"
    else
        echo " [ MISSING ] $dep"
        MISSING_DEPS+=("$dep")
    fi
done
echo ""

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "All required dependencies are already installed. Skipping installation."
else
    echo "The following required dependencies are missing:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo " - $dep"
    done
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
echo "1. Build everything"
echo "2. Build only llama-cli, llama-gguf-split, and llama-server (Default)"
echo ""

# Prompt with a 5 second visible countdown. If no input is given (timeout)
# or Enter is pressed, use the default (option 2).
if read_with_countdown 5 "Enter your choice (1 or 2), defaults to '2'"; then
    build_choice="$REPLY_INPUT"
else
    echo "No input received within 5 seconds. Defaulting to option 2."
    build_choice="2"
fi

# Treat empty input (e.g. just pressing Enter) as the default as well
if [[ -z "$build_choice" ]]; then
    build_choice="2"
fi

echo ""
echo "----------------------------------------------------------------"
echo "Building project (Release mode)..."
echo "----------------------------------------------------------------"

if [[ "$build_choice" == "1" ]]; then
    # Option 1: Build everything
    cmake --build llama.cpp/build --config Release -j --clean-first
else
    # Option 2 (Default): Build specific targets
    cmake --build llama.cpp/build --config Release -j --clean-first --target llama-cli llama-gguf-split llama-server
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
