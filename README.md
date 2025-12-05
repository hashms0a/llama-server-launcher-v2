
# Llama Server Launcher

A Tkinter GUI for launching `llama-server` from the llama.cpp project with common parameters. This tool simplifies the process of running LLM models locally by providing a user-friendly interface for configuring server options, managing model settings, and handling server processes.

<img width="1408" height="878" alt="2025-12-05_17-47" src="https://github.com/user-attachments/assets/7be98875-ecb4-4e87-b582-422b70af6c2e" />

## Features

- **Persistent Settings**: Saves and restores settings per GGUF file
- **Process Management**: Launch server in terminal or background with kill functionality
- **Command Preview**: Shows the exact command that will be executed
- **Model Search**: Filter GGUF files in your model directory
- **OpenAI-Compatible API**: Full support for OpenAI API endpoints
- **Multimodal Support**: Configure mmproj for vision models
- **Comprehensive Parameters**: Access to most llama-server options

## Requirements

- Python 3.6+
- Tkinter (usually included with Python)
- `llama-server` executable from llama.cpp
- GGUF model files

## Installation

### Option 1: Automated Build (Recommended)

Use the provided `build_llamacpp.sh` script to automatically build llama.cpp:

1. **Download the build script**:
   Save the `build_llamacpp.sh` file to your preferred location

2. **Make the script executable**:
   ```bash
   chmod +x build_llamacpp.sh
   ```

3. **Run the build script**:
   ```bash
   ./build_llamacpp.sh
   ```

The script will:
- Ask to install required dependencies (pciutils, build-essential, cmake, curl, libcurl4-openssl-dev)
- Clone the llama.cpp repository (or update if it already exists)
- Configure CMake with CUDA and CURL support enabled
- Build the project in Release mode using all available CPU cores
- Copy the `llama-server` binary to the `llama.cpp` folder

### Option 2: Manual Installation

1. **Install dependencies manually**:
   ```bash
   sudo apt update
   sudo apt install pciutils build-essential cmake curl libcurl4-openssl-dev
   ```

2. **Clone llama.cpp**:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   ```

3. **Build with CMake**:
   ```bash
   cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
   cmake --build build --config Release -j
   ```

4. **Copy the binary**:
   ```bash
   cp build/bin/llama-server .
   ```

3. **Download the launcher**:
   Save the `llama_server_launcher_v5.py` file to your preferred location

4. **Make executable** (optional):
   ```bash
   chmod +x llama_server_launcher_v5.py
   ```

## Usage

### Basic Usage

1. **Launch the GUI**:
   ```bash
   python3 llama_server_launcher_v5.py
   ```

2. **Configure llama-server path**:
   - Click "Browse..." next to "Llama Server Executable"
   - Navigate to your `llama-server` binary (usually in `~/llama.cpp/`)
   - Default path: `~/llama.cpp/llama-server`

3. **Select your model directory**:
   - Click "Browse..." next to "GGUF Directory"
   - Select the folder containing your GGUF model files
   - Use the search box to filter models

4. **Configure parameters**:
   - **Common Parameters Tab**: Network, performance, and sampling settings
   - **Additional Parameters Tab**: API settings, multimodal, batch options, and flags

5. **Launch the server**:
   - Choose "Run in Terminal" or "Run in Background"
   - Click "Launch Server"
   - The API endpoint information will be displayed

### Key Configuration Options

#### Network Settings
- **Host**: `0.0.0.0` (all interfaces) or `localhost`
- **Port**: Default `8033`

#### Performance Settings
- **GPU Layers (-ngl)**: Number of layers to offload to GPU (`99` = all)
- **Threads**: Number of CPU threads (`-1` = auto)
- **Context Size**: Context window size in tokens

#### Sampling Settings
- **Temperature**: Controls randomness (0.0-2.0)
- **Top P**: Nucleus sampling threshold (0.0-1.0)
- **Top K**: Top-k filtering (0 = disabled)
- **Min P**: Minimum probability threshold

#### OpenAI-Compatible API
- **API Key**: Optional authentication key
- **Model Alias**: Name returned by `/v1/models`
- **Chat Template**: Custom Jinja2 template or file

### Advanced Features

#### Saving/Loading Settings
- Settings are automatically saved per model
- Select a model to load its saved configuration
- Click "Save Settings" to manually save current configuration

#### Command Preview
- View the exact command that will be executed
- Copy command to clipboard for manual execution
- Updates automatically when parameters change

#### Process Management
- **Terminal Mode**: Opens in a terminal emulator (gnome-terminal, konsole, etc.)
- **Background Mode**: Runs as a background process with PID tracking
- **Kill Server**: Stop running instances (background or all via pkill)

## API Endpoints

Once running, the server provides OpenAI-compatible endpoints:

- **Chat Completions**: `/v1/chat/completions`
- **Text Completions**: `/v1/completions`
- **List Models**: `/v1/models`
- **Embeddings**: `/v1/embeddings`
- **Health Check**: `/health`

## Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8033/v1",
    api_key="not-needed"  # Optional if API key is set
)

response = client.chat.completions.create(
    model="your-model-alias",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## Configuration File

Settings are stored in `~/.llama_server_launcher_config.json`:
- Last used paths and directories
- Per-model parameter settings
- Default values for all options

## Troubleshooting

### Build Issues
- Ensure you have an NVIDIA GPU and CUDA toolkit installed for `DGGML_CUDA=ON`
- Remove `-DGGML_CUDA=ON` from the build script if you don't have CUDA
- Make sure you have sufficient RAM for the build process

### Server won't start
1. Verify `llama-server` path is correct and executable
2. Check that the selected GGUF file exists
3. Ensure required dependencies are installed

### Permission denied
```bash
chmod +x /path/to/llama-server
```

### No terminal emulator found
The launcher tries multiple terminal emulators. If none are found:
- Use "Run in Background" mode instead
- Install a supported terminal (gnome-terminal, konsole, xfce4-terminal, xterm)

### Model not loading
1. Check the GGUF file is valid
2. Verify context size is appropriate for the model
3. Ensure sufficient RAM/VRAM for the model size

## Tips

- **GPU Offloading**: Set `-ngl` to `99` for maximum GPU acceleration
- **Context Size**: Larger contexts require more RAM
- **Batch Size**: Leave empty for default unless you know what you're doing
- **API Key**: Use for production deployments
- **Model Alias**: Set to a friendly name for easier API usage

## Supported Terminal Emulators

- gnome-terminal
- konsole
- xfce4-terminal
- xterm

## License

This launcher is provided as-is for use with llama.cpp. Please refer to the llama.cpp license for the server component.
