# Llama Server Launcher

A Tkinter GUI for launching `llama-server` from the llama.cpp project with common parameters. This tool simplifies the process of running LLM models locally by providing a user-friendly interface for configuring server options, managing model settings, and handling server processes.

<img width="762" height="896" alt="2025-12-06_10-55" src="https://github.com/user-attachments/assets/a7378451-a7e0-4c91-a22d-e62b4942eb6a" />

## Features

- **Persistent Settings**: Saves and restores settings per GGUF file
- **Process Management**: Launch server in terminal or background with kill functionality
- **Command Preview**: Shows the exact command that will be executed
- **Model Search**: Filter GGUF files in your model directory
- **Model Information**: Automatic display of model metadata (architecture, layers, context size, quantization)
- **OpenAI-Compatible API**: Full support for OpenAI API endpoints
- **Multimodal Support**: Configure mmproj for vision models
- **Comprehensive Parameters**: Access to most llama-server options
- **Favorites System**: Manage and organize models with custom notes
  - **★ Toggle Button**: Click to add/remove model from favorites
  - **Star Indicator**: Favorites show with ★ prefix in dropdown and appear first
  - **Notes**: Add custom notes for each model (e.g., "Best for coding", "24GB VRAM")
  - **Auto-save**: Notes save on Enter key or when clicking elsewhere
  - **Auto-favorite**: Adding a note automatically marks model as favorite
  - **Persistence**: Favorites and notes saved to config file

## Requirements

- Python 3.6+
- Tkinter (usually included with Python)
- `llama-server` executable from llama.cpp
- GGUF model files

## Installation

### 1. Create Virtual Environment

Create and activate a Python virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies

Install the required Python packages for model metadata analysis:

```bash
pip install -r requirements.txt --break-system-packages
```

Or install manually:
```bash
pip install gguf
```

*Note: The `gguf` library is lightweight and recommended for metadata extraction. If unavailable, the launcher will still work but won't display model information.*

### 3. Build llama.cpp

#### Option A: Automated Build (Recommended)

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

#### Option B: Manual Installation

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

### 4. Download the Launcher

Save the `llama_server_launcher_v6.py` file to your preferred location

5. **Make executable** (optional):
   ```bash
   chmod +x llama_server_launcher_v6.py
   ```

## Usage

### Basic Usage

1. **Launch the GUI**:
   ```bash
   python3 llama_server_launcher_v6.py
   ```

2. **Configure llama-server path**:
   - Click "Browse..." next to "Llama Server Executable"
   - Navigate to your `llama-server` binary (usually in `~/llama.cpp/`)
   - Default path: `~/llama.cpp/llama-server`

3. **Select your model directory**:
   - Click "Browse..." next to "GGUF Directory"
   - Select the folder containing your GGUF model files
   - Use the search box to filter models

4. **View model information**:
   - Select a model from the dropdown
   - Model metadata will be displayed (architecture, layers, context size, quantization, file size)
   - Information is extracted automatically if `gguf` library is installed

5. **Configure parameters**:
   - **Common Parameters Tab**: Network, performance, and sampling settings
   - **Additional Parameters Tab**: API settings, multimodal, batch options, and flags

6. **Launch the server**:
   - Choose "Run in Terminal" or "Run in Background"
   - Click "Launch Server"
   - The API endpoint information will be displayed

### Key Configuration Options

#### Network Settings
- **Host**: `0.0.0.0` (all interfaces) or `localhost`
- **Port**: Default `8033` (updated from v5)

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

### Model Information Display

When you select a GGUF model, the launcher automatically analyzes and displays:
- **Architecture**: Model architecture (e.g., llama, mistral, mixtral)
- **Layers**: Number of transformer layers
- **Context**: Context window size (displayed in K for large values)
- **Size**: File size in GB
- **Quant**: Quantization type (e.g., Q4_K_M, Q8_0, F16)

*Note: Model analysis requires the `gguf` Python library. Without it, only file size will be shown.*

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
    base_url="http://localhost:8033/v1",  # Note: Default port changed to 8033
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
- Favorites and notes (new in this version)

## Troubleshooting

### Model Information Not Showing
- Install the `gguf` library: `pip install gguf`
- Or install `llama-cpp-python` as an alternative: `pip install llama-cpp-python`
- Without these libraries, only file size will be displayed

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
- **Model Analysis**: Install `gguf` library for automatic model metadata display
- **Favorites**: Use the star button to quickly access frequently used models

## Supported Terminal Emulators

- gnome-terminal
- konsole
- xfce4-terminal
- xterm

## License

This launcher is provided as-is for use with llama.cpp. Please refer to the llama.cpp license for the server component.
