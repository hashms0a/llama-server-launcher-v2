#!/usr/bin/env python3
"""
Llama Server Launcher - A Tkinter GUI for launching llama-server with common parameters.
Supports persistent settings per GGUF file, process management, and command preview.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import subprocess
import signal
import shlex
import threading
from pathlib import Path
from typing import Optional, Any, Dict

# Try to import llama-cpp-python for GGUF analysis
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Try to import gguf library for direct metadata reading (more reliable)
try:
    from gguf import GGUFReader
    GGUF_READER_AVAILABLE = True
except ImportError:
    GGUF_READER_AVAILABLE = False

# Configuration file path
CONFIG_FILE = os.path.expanduser("~/.llama_server_launcher_config.json")
DEFAULT_LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/llama-server")


class SafeVar:
    """Wrapper for Tkinter variables that handles empty/invalid values gracefully."""

    @staticmethod
    def get_int(var: tk.StringVar, default: int = 0) -> int:
        """Safely get integer value from a StringVar."""
        try:
            value = var.get().strip()
            if value == "" or value == "-":
                return default
            return int(value)
        except (tk.TclError, ValueError):
            return default

    @staticmethod
    def get_float(var: tk.StringVar, default: float = 0.0) -> float:
        """Safely get float value from a StringVar."""
        try:
            value = var.get().strip()
            if value == "" or value == "-" or value == ".":
                return default
            return float(value)
        except (tk.TclError, ValueError):
            return default

    @staticmethod
    def get_str(var: tk.StringVar, default: str = "") -> str:
        """Safely get string value from a StringVar."""
        try:
            return var.get().strip()
        except tk.TclError:
            return default


class LlamaServerLauncher:
    """Main application class for Llama Server Launcher."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Llama Server Launcher")
        self.root.geometry("750x850")
        self.root.minsize(700, 750)

        # Process handle for background server
        self.server_process: Optional[subprocess.Popen] = None

        # Load configuration
        self.config = self.load_config()

        # Initialize variables
        self.init_variables()

        # Build UI
        self.build_ui()

        # Load last used directory and settings
        self.load_last_session()

        # Bind cleanup on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_variables(self):
        """Initialize all Tkinter variables."""
        # Path variables
        self.llama_server_path_var = tk.StringVar(value=self.config.get("llama_server_path", DEFAULT_LLAMA_SERVER_PATH))
        self.gguf_dir_var = tk.StringVar(value=self.config.get("last_gguf_dir", ""))
        self.selected_gguf_var = tk.StringVar()

        # Common parameters (using StringVar for safe handling)
        self.host_var = tk.StringVar(value="0.0.0.0")
        self.port_var = tk.StringVar(value="8033")
        self.ngl_var = tk.StringVar(value="99")
        self.ncmoe_var = tk.StringVar(value="")
        self.jinja_var = tk.BooleanVar(value=True)
        self.threads_var = tk.StringVar(value="-1")
        self.ctx_size_var = tk.StringVar(value="8192")
        self.temp_var = tk.StringVar(value="0.7")
        self.min_p_var = tk.StringVar(value="0.0")
        self.top_p_var = tk.StringVar(value="0.9")
        self.top_k_var = tk.StringVar(value="40")
        self.presence_penalty_var = tk.StringVar(value="0.0")

        # Additional parameters
        self.mmproj_path_var = tk.StringVar(value="")
        self.batch_size_var = tk.StringVar(value="")
        self.ubatch_size_var = tk.StringVar(value="")
        self.n_predict_var = tk.StringVar(value="")
        self.rope_freq_base_var = tk.StringVar(value="")
        self.rope_freq_scale_var = tk.StringVar(value="")
        self.repeat_penalty_var = tk.StringVar(value="")
        self.frequency_penalty_var = tk.StringVar(value="")
        self.flash_attn_var = tk.BooleanVar(value=False)
        self.mlock_var = tk.BooleanVar(value=False)
        self.no_mmap_var = tk.BooleanVar(value=False)
        self.cont_batching_var = tk.BooleanVar(value=True)
        self.metrics_var = tk.BooleanVar(value=False)
        self.verbose_var = tk.BooleanVar(value=False)
        self.log_disable_var = tk.BooleanVar(value=False)
        self.parallel_var = tk.StringVar(value="")
        self.custom_args_var = tk.StringVar(value="")

        # OpenAI-compatible API settings
        self.api_key_var = tk.StringVar(value="")
        self.model_alias_var = tk.StringVar(value="")
        self.chat_template_var = tk.StringVar(value="")
        self.chat_template_file_var = tk.StringVar(value="")

        # Run mode
        self.run_in_terminal_var = tk.BooleanVar(value=True)

        # Available GGUF files list
        self.gguf_files: list[str] = []

        # Search filter for models
        self.model_search_var = tk.StringVar()

        # Model analysis info
        self.model_info_var = tk.StringVar(value="No model selected")
        self.analysis_thread: Optional[threading.Thread] = None
        self.current_analysis_path: Optional[str] = None
        self.last_analysis_result: Optional[Dict[str, Any]] = None

    def build_ui(self):
        """Build the main user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Llama server path section
        self.build_server_path_section(main_frame)

        # GGUF file selection section
        self.build_gguf_section(main_frame)

        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Common parameters tab
        common_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(common_tab, text="Common Parameters")
        self.build_common_params_tab(common_tab)

        # Additional parameters tab
        additional_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(additional_tab, text="Additional Parameters")
        self.build_additional_params_tab(additional_tab)

        # Command preview section
        self.build_command_preview_section(main_frame)

        # Control buttons section
        self.build_control_buttons(main_frame)

        # Status bar
        self.build_status_bar(main_frame)

    def build_server_path_section(self, parent):
        """Build the llama-server path selection section."""
        frame = ttk.LabelFrame(parent, text="Llama Server Executable", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        path_frame = ttk.Frame(frame)
        path_frame.pack(fill=tk.X)

        ttk.Entry(path_frame, textvariable=self.llama_server_path_var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="Browse...", command=self.browse_llama_server).pack(side=tk.LEFT, padx=(5, 0))

    def build_gguf_section(self, parent):
        """Build the GGUF file selection section."""
        frame = ttk.LabelFrame(parent, text="Model Selection", padding="5")
        frame.pack(fill=tk.X, pady=(0, 10))

        # Directory selection
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(dir_frame, text="GGUF Directory:").pack(side=tk.LEFT)
        ttk.Entry(dir_frame, textvariable=self.gguf_dir_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_gguf_dir).pack(side=tk.LEFT)

        # Search box
        search_frame = ttk.Frame(frame)
        search_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        search_entry = ttk.Entry(search_frame, textvariable=self.model_search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.model_search_var.trace_add("write", self.filter_gguf_list)
        ttk.Button(search_frame, text="Clear", command=lambda: self.model_search_var.set("")).pack(side=tk.LEFT)

        # GGUF file dropdown
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X)

        ttk.Label(file_frame, text="Select Model:").pack(side=tk.LEFT)
        self.gguf_combo = ttk.Combobox(file_frame, textvariable=self.selected_gguf_var, state="readonly", width=60)
        self.gguf_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.gguf_combo.bind("<<ComboboxSelected>>", self.on_gguf_selected)
        ttk.Button(file_frame, text="Refresh", command=self.refresh_gguf_list).pack(side=tk.LEFT)

        # Model info display
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))

        self.model_info_label = ttk.Label(info_frame, textvariable=self.model_info_var, foreground="gray", font=("TkDefaultFont", 9))
        self.model_info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def build_common_params_tab(self, parent):
        """Build the common parameters tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Network settings
        network_frame = ttk.LabelFrame(scrollable_frame, text="Network Settings", padding="5")
        network_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(network_frame, "Host (--host):", self.host_var, "IP address to bind (0.0.0.0 for all)", 0)
        self.create_param_row(network_frame, "Port (--port):", self.port_var, "Port number (default: 8033)", 1)

        # GPU/Performance settings
        perf_frame = ttk.LabelFrame(scrollable_frame, text="Performance Settings", padding="5")
        perf_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(perf_frame, "GPU Layers (-ngl):", self.ngl_var, "Number of layers to offload to GPU (99 = all)", 0)
        self.create_param_row(perf_frame, "MoE Experts (-ncmoe):", self.ncmoe_var, "Number of MoE experts to use (leave empty for default)", 1)
        self.create_param_row(perf_frame, "Threads (--threads):", self.threads_var, "Number of threads (-1 = auto)", 2)
        self.create_param_row(perf_frame, "Context Size (--ctx-size):", self.ctx_size_var, "Context window size in tokens", 3)

        # Sampling settings
        sampling_frame = ttk.LabelFrame(scrollable_frame, text="Sampling Settings", padding="5")
        sampling_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(sampling_frame, "Temperature (--temp):", self.temp_var, "Sampling temperature (0.0-2.0)", 0)
        self.create_param_row(sampling_frame, "Min P (--min-p):", self.min_p_var, "Minimum probability threshold (0.0-1.0)", 1)
        self.create_param_row(sampling_frame, "Top P (--top-p):", self.top_p_var, "Top-p sampling probability (0.0-1.0)", 2)
        self.create_param_row(sampling_frame, "Top K (--top-k):", self.top_k_var, "Top-k filtering (0 = disabled)", 3)
        self.create_param_row(sampling_frame, "Presence Penalty:", self.presence_penalty_var, "Presence penalty (-2.0 to 2.0)", 4)
        self.create_param_row(sampling_frame, "Repeat Penalty:", self.repeat_penalty_var, "Repeat penalty (1.0 = disabled)", 5)

        # Flags
        flags_frame = ttk.LabelFrame(scrollable_frame, text="Flags", padding="5")
        flags_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(flags_frame, text="Enable Jinja templates (--jinja)", variable=self.jinja_var).pack(anchor=tk.W)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def build_additional_params_tab(self, parent):
        """Build the additional parameters tab."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # OpenAI-compatible API settings
        api_frame = ttk.LabelFrame(scrollable_frame, text="OpenAI-Compatible API Settings", padding="5")
        api_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(api_frame, "API Key (--api-key):", self.api_key_var, "API key for authentication (optional)", 0)
        self.create_param_row(api_frame, "Model Alias (--alias):", self.model_alias_var, "Model name returned by /v1/models", 1)
        self.create_param_row(api_frame, "Chat Template (--chat-template):", self.chat_template_var, "Jinja2 chat template (optional)", 2)

        template_file_row = ttk.Frame(api_frame)
        template_file_row.pack(fill=tk.X, pady=2)
        ttk.Label(template_file_row, text="Chat Template File:", width=25).pack(side=tk.LEFT)
        ttk.Entry(template_file_row, textvariable=self.chat_template_file_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(template_file_row, text="Browse...", command=self.browse_chat_template_file).pack(side=tk.LEFT)

        ttk.Label(api_frame, text="Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings", foreground="gray").pack(anchor=tk.W, pady=(5, 0))

        # Multimodal settings
        mm_frame = ttk.LabelFrame(scrollable_frame, text="Multimodal Settings", padding="5")
        mm_frame.pack(fill=tk.X, pady=(0, 10))

        mmproj_row = ttk.Frame(mm_frame)
        mmproj_row.pack(fill=tk.X, pady=2)
        ttk.Label(mmproj_row, text="MMProj Path (--mmproj):", width=25).pack(side=tk.LEFT)
        ttk.Entry(mmproj_row, textvariable=self.mmproj_path_var, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(mmproj_row, text="Browse...", command=self.browse_mmproj).pack(side=tk.LEFT)

        # Batch settings
        batch_frame = ttk.LabelFrame(scrollable_frame, text="Batch Settings", padding="5")
        batch_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(batch_frame, "Batch Size (-b):", self.batch_size_var, "Logical batch size (leave empty for default)", 0)
        self.create_param_row(batch_frame, "Micro Batch (-ub):", self.ubatch_size_var, "Physical batch size (leave empty for default)", 1)
        self.create_param_row(batch_frame, "Parallel Slots (--parallel):", self.parallel_var, "Number of parallel sequences", 2)

        # Generation settings
        gen_frame = ttk.LabelFrame(scrollable_frame, text="Generation Settings", padding="5")
        gen_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(gen_frame, "Max Predict (-n):", self.n_predict_var, "Max tokens to predict (-1 = infinite)", 0)
        self.create_param_row(gen_frame, "Frequency Penalty:", self.frequency_penalty_var, "Frequency penalty (0.0-2.0)", 1)

        # RoPE settings
        rope_frame = ttk.LabelFrame(scrollable_frame, text="RoPE Settings", padding="5")
        rope_frame.pack(fill=tk.X, pady=(0, 10))

        self.create_param_row(rope_frame, "RoPE Freq Base:", self.rope_freq_base_var, "RoPE frequency base (leave empty for default)", 0)
        self.create_param_row(rope_frame, "RoPE Freq Scale:", self.rope_freq_scale_var, "RoPE frequency scale (leave empty for default)", 1)

        # Additional flags
        flags_frame = ttk.LabelFrame(scrollable_frame, text="Additional Flags", padding="5")
        flags_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(flags_frame, text="Flash Attention (-fa)", variable=self.flash_attn_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Lock memory (--mlock)", variable=self.mlock_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Disable mmap (--no-mmap)", variable=self.no_mmap_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Continuous batching (-cb)", variable=self.cont_batching_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Enable metrics (--metrics)", variable=self.metrics_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Verbose output (--verbose)", variable=self.verbose_var).pack(anchor=tk.W)
        ttk.Checkbutton(flags_frame, text="Disable logging (--log-disable)", variable=self.log_disable_var).pack(anchor=tk.W)

        # Custom arguments
        custom_frame = ttk.LabelFrame(scrollable_frame, text="Custom Arguments", padding="5")
        custom_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(custom_frame, text="Additional arguments (space-separated):").pack(anchor=tk.W)
        ttk.Entry(custom_frame, textvariable=self.custom_args_var, width=60).pack(fill=tk.X, pady=(5, 0))

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_param_row(self, parent, label: str, var: tk.StringVar, tooltip: str, row: int):
        """Create a parameter input row with label and entry."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)

        lbl = ttk.Label(frame, text=label, width=25)
        lbl.pack(side=tk.LEFT)

        entry = ttk.Entry(frame, textvariable=var, width=15)
        entry.pack(side=tk.LEFT, padx=(5, 10))

        ttk.Label(frame, text=tooltip, foreground="gray").pack(side=tk.LEFT)

    def build_command_preview_section(self, parent):
        """Build the command preview section."""
        frame = ttk.LabelFrame(parent, text="Command Preview", padding="5")
        frame.pack(fill=tk.X, pady=(10, 0))

        # Text widget for command display
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.X)

        self.command_text = tk.Text(text_frame, height=8, wrap=tk.WORD, font=("Courier", 9))
        self.command_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.command_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.command_text.configure(yscrollcommand=scrollbar.set)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(btn_frame, text="Update Preview", command=self.update_command_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Copy Command", command=self.copy_command).pack(side=tk.LEFT)

    def build_control_buttons(self, parent):
        """Build the control buttons section."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=(10, 0))

        # Run mode selection
        mode_frame = ttk.LabelFrame(frame, text="Run Mode", padding="5")
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Radiobutton(mode_frame, text="Run in Terminal", variable=self.run_in_terminal_var, value=True).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Run in Background", variable=self.run_in_terminal_var, value=False).pack(side=tk.LEFT, padx=(10, 0))

        # Control buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.RIGHT)

        self.launch_btn = ttk.Button(btn_frame, text="Launch Server", command=self.launch_server, style="Accent.TButton")
        self.launch_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.kill_btn = ttk.Button(btn_frame, text="Kill Server", command=self.kill_server)
        self.kill_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(btn_frame, text="Save Settings", command=self.save_current_settings).pack(side=tk.LEFT)

    def build_status_bar(self, parent):
        """Build the status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))

    def browse_llama_server(self):
        """Browse for llama-server executable."""
        path = filedialog.askopenfilename(
            title="Select llama-server executable",
            initialdir=os.path.dirname(SafeVar.get_str(self.llama_server_path_var)) or os.path.expanduser("~")
        )
        if path:
            self.llama_server_path_var.set(path)
            self.save_config()

    def browse_gguf_dir(self):
        """Browse for GGUF files directory."""
        path = filedialog.askdirectory(
            title="Select GGUF Files Directory",
            initialdir=SafeVar.get_str(self.gguf_dir_var) or os.path.expanduser("~")
        )
        if path:
            self.gguf_dir_var.set(path)
            self.refresh_gguf_list()
            self.save_config()

    def browse_mmproj(self):
        """Browse for mmproj file."""
        path = filedialog.askopenfilename(
            title="Select MMProj File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir=SafeVar.get_str(self.gguf_dir_var) or os.path.expanduser("~")
        )
        if path:
            self.mmproj_path_var.set(path)

    def browse_chat_template_file(self):
        """Browse for chat template file."""
        path = filedialog.askopenfilename(
            title="Select Chat Template File",
            filetypes=[("Jinja2 templates", "*.jinja *.jinja2 *.j2"), ("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~")
        )
        if path:
            self.chat_template_file_var.set(path)

    def refresh_gguf_list(self):
        """Refresh the list of GGUF files in the selected directory."""
        gguf_dir = SafeVar.get_str(self.gguf_dir_var)
        if not gguf_dir or not os.path.isdir(gguf_dir):
            self.gguf_files = []
            self.gguf_combo["values"] = []
            return

        self.gguf_files = sorted([
            f for f in os.listdir(gguf_dir)
            if f.lower().endswith(".gguf")
        ])

        # Apply current filter
        self.filter_gguf_list()

        if self.gguf_files:
            self.status_var.set(f"Found {len(self.gguf_files)} GGUF file(s)")
        else:
            self.status_var.set("No GGUF files found in directory")

    def filter_gguf_list(self, *args):
        """Filter the GGUF files list based on search text."""
        search_text = SafeVar.get_str(self.model_search_var).lower()

        if not search_text:
            # No filter, show all files
            filtered = self.gguf_files
        else:
            # Filter files containing all search terms (space-separated)
            search_terms = search_text.split()
            filtered = [
                f for f in self.gguf_files
                if all(term in f.lower() for term in search_terms)
            ]

        self.gguf_combo["values"] = filtered

        # Update status
        if search_text and self.gguf_files:
            self.status_var.set(f"Showing {len(filtered)} of {len(self.gguf_files)} model(s)")

    def _run_gguf_analysis(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze a GGUF model file to extract metadata.
        Runs in a worker thread to avoid blocking the UI.

        Uses gguf library for direct file reading (preferred) or falls back to llama-cpp-python.
        Returns a dictionary with model information or an error key.
        """
        result = {
            "path": model_path,
            "filename": os.path.basename(model_path),
        }

        try:
            # Get file size
            file_size_bytes = os.path.getsize(model_path)
            result["file_size_bytes"] = file_size_bytes
            result["file_size_gb"] = round(file_size_bytes / (1024 ** 3), 2)

            # Try gguf library first (most reliable - reads file directly)
            if GGUF_READER_AVAILABLE:
                try:
                    result = self._analyze_with_gguf_reader(model_path, result)
                    return result
                except Exception as e:
                    # Fall through to try llama-cpp-python
                    pass

            # Try llama-cpp-python as fallback
            if LLAMA_CPP_AVAILABLE:
                try:
                    result = self._analyze_with_llama_cpp(model_path, result)
                    return result
                except Exception as e:
                    result["warning"] = f"Metadata extraction failed: {str(e)}"
                    return result

            # No analysis libraries available
            if not GGUF_READER_AVAILABLE and not LLAMA_CPP_AVAILABLE:
                result["warning"] = "Install 'gguf' or 'llama-cpp-python' for metadata"

        except FileNotFoundError:
            result["error"] = "File not found"
        except PermissionError:
            result["error"] = "Permission denied"
        except Exception as e:
            result["error"] = f"Analysis failed: {str(e)}"

        return result

    def _analyze_with_gguf_reader(self, model_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze GGUF using the gguf library (reads file directly without loading model).
        This is the most reliable method.
        """
        reader = GGUFReader(model_path)

        # Helper function to extract field value
        def get_field_value(field_name: str, default=None):
            """Extract a single value from a GGUF field."""
            if field_name not in reader.fields:
                return default
            field = reader.fields[field_name]
            try:
                # Get the data parts
                if hasattr(field, 'parts') and field.parts:
                    data = field.parts[-1]
                    if hasattr(data, 'tolist'):
                        val = data.tolist()
                        if isinstance(val, list):
                            if len(val) == 0:
                                return default
                            # Single element array - return the value directly
                            if len(val) == 1:
                                return val[0]
                            # Multi-element array - check if it's an ASCII string
                            # ASCII strings are typically longer and contain printable chars
                            try:
                                int_vals = [int(x) for x in val]
                                # If all values are valid ASCII printable range or common control chars
                                if all(0 <= x < 256 for x in int_vals):
                                    decoded = bytes(int_vals).decode('utf-8', errors='ignore')
                                    # Return as string if it looks like text (has letters)
                                    if any(c.isalpha() for c in decoded):
                                        return decoded
                            except (ValueError, TypeError, OverflowError):
                                pass
                            # Otherwise return the list as-is
                            return val
                        # Decode bytes to string
                        if isinstance(val, bytes):
                            return val.decode('utf-8', errors='ignore')
                        return val
                    elif isinstance(data, bytes):
                        return data.decode('utf-8', errors='ignore')
                    else:
                        return data
                # Alternative: try data attribute
                elif hasattr(field, 'data'):
                    data = field.data
                    if isinstance(data, bytes):
                        return data.decode('utf-8', errors='ignore')
                    return data
            except Exception:
                pass
            return default

        # Architecture
        arch = get_field_value("general.architecture", "unknown")
        if isinstance(arch, bytes):
            arch = arch.decode('utf-8', errors='ignore')
        # Safety: if still a list of ints, convert to string
        if isinstance(arch, list):
            try:
                arch = bytes(int(x) for x in arch).decode('utf-8', errors='ignore')
            except (ValueError, TypeError):
                arch = str(arch)
        result["architecture"] = str(arch) if arch else "unknown"

        # Model name
        model_name = get_field_value("general.name", result["filename"])
        result["model_name"] = str(model_name) if model_name else result["filename"]

        # Context length - try architecture-specific key first
        ctx_length = get_field_value(f"{arch}.context_length")
        if ctx_length is None:
            ctx_length = get_field_value("general.context_length")
        # Search for any context_length key
        if ctx_length is None:
            for key in reader.fields.keys():
                if 'context_length' in key:
                    ctx_length = get_field_value(key)
                    if ctx_length is not None:
                        break
        result["context_length"] = ctx_length if ctx_length is not None else "unknown"

        # Layer count (block_count) - try architecture-specific key first
        layer_count = get_field_value(f"{arch}.block_count")
        if layer_count is None:
            # Search for any block_count key
            for key in reader.fields.keys():
                if 'block_count' in key:
                    layer_count = get_field_value(key)
                    if layer_count is not None:
                        break
        result["layer_count"] = layer_count if layer_count is not None else "unknown"

        # Embedding length - search for it
        embed_length = get_field_value(f"{arch}.embedding_length")
        if embed_length is None:
            for key in reader.fields.keys():
                if 'embedding_length' in key:
                    embed_length = get_field_value(key)
                    if embed_length is not None:
                        break
        result["embedding_length"] = embed_length if embed_length is not None else "unknown"

        # Head count - search for it
        head_count = get_field_value(f"{arch}.attention.head_count")
        if head_count is None:
            for key in reader.fields.keys():
                if 'head_count' in key:
                    head_count = get_field_value(key)
                    if head_count is not None:
                        break
        result["head_count"] = head_count if head_count is not None else "unknown"

        # Quantization - always use filename heuristic as it's most reliable
        quant = self._guess_quantization(result["filename"])
        if quant == "unknown":
            # Try file_type as fallback
            file_type = get_field_value("general.file_type")
            if file_type is not None:
                quant = f"type_{file_type}"
        result["quantization"] = quant

        return result

    def _analyze_with_llama_cpp(self, model_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze GGUF using llama-cpp-python (fallback method).
        Note: This may fail for some models.
        """
        llm = Llama(
            model_path=model_path,
            vocab_only=True,
            verbose=False,
            n_ctx=0,
            n_gpu_layers=0,
        )

        # Extract metadata from the model
        metadata = llm.metadata if hasattr(llm, 'metadata') else {}

        # Architecture
        result["architecture"] = metadata.get("general.architecture", "unknown")

        # Model name from metadata
        result["model_name"] = metadata.get("general.name", result["filename"])

        # Context length
        ctx_key = f"{result['architecture']}.context_length"
        result["context_length"] = metadata.get(ctx_key, metadata.get("general.context_length", "unknown"))

        # Layer count
        layer_key = f"{result['architecture']}.block_count"
        result["layer_count"] = metadata.get(layer_key, "unknown")

        # Embedding length
        embed_key = f"{result['architecture']}.embedding_length"
        result["embedding_length"] = metadata.get(embed_key, "unknown")

        # Head count
        head_key = f"{result['architecture']}.attention.head_count"
        result["head_count"] = metadata.get(head_key, "unknown")

        # Quantization type
        result["quantization"] = metadata.get("general.quantization_version", self._guess_quantization(result["filename"]))

        # Vocab size
        if hasattr(llm, 'n_vocab'):
            result["vocab_size"] = llm.n_vocab()

        # Clean up
        del llm

        return result

    def _guess_quantization(self, filename: str) -> str:
        """Guess quantization type from filename."""
        filename_upper = filename.upper()
        quant_types = [
            "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
            "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
            "Q3_K_M", "Q3_K_S", "Q3_K_L",
            "Q2_K", "IQ4_XS", "IQ3_XXS", "IQ2_XXS",
            "F16", "F32", "BF16"
        ]
        for qt in quant_types:
            if qt in filename_upper or qt.replace("_", "-") in filename_upper:
                return qt
        return "unknown"

    def _update_ui_after_analysis(self, result: Dict[str, Any]):
        """Update UI elements after GGUF analysis completes."""
        if "error" in result:
            info_text = f"⚠ {result.get('filename', 'Unknown')}: {result['error']}"
            self.model_info_var.set(info_text)
            return

        # Build info string
        parts = []

        # Architecture
        if result.get("architecture") and result["architecture"] != "unknown":
            parts.append(f"Arch: {result['architecture']}")

        # Layers
        if result.get("layer_count") and result["layer_count"] != "unknown":
            parts.append(f"Layers: {result['layer_count']}")

        # Context length
        if result.get("context_length") and result["context_length"] != "unknown":
            ctx = result["context_length"]
            if isinstance(ctx, int) and ctx >= 1024:
                ctx_display = f"{ctx // 1024}K"
            else:
                ctx_display = str(ctx)
            parts.append(f"Ctx: {ctx_display}")

        # File size
        if result.get("file_size_gb"):
            parts.append(f"Size: {result['file_size_gb']} GB")

        # Quantization
        if result.get("quantization") and result["quantization"] != "unknown":
            parts.append(f"Quant: {result['quantization']}")

        if parts:
            info_text = " │ ".join(parts)
        else:
            info_text = f"Size: {result.get('file_size_gb', '?')} GB"

        # Add warning if llama-cpp-python not available
        if result.get("warning"):
            info_text = f"Size: {result.get('file_size_gb', '?')} GB │ ⚠ {result['warning']}"

        self.model_info_var.set(info_text)

        # Store analysis result for potential use (e.g., setting max layers)
        self.last_analysis_result = result

    def _start_analysis(self, model_path: str):
        """Start GGUF analysis in a background thread."""
        # Cancel any ongoing analysis for a different file
        if self.current_analysis_path != model_path:
            self.current_analysis_path = model_path
            self.model_info_var.set("Analyzing model...")

            def analysis_worker():
                result = self._run_gguf_analysis(model_path)
                # Schedule UI update on main thread
                if self.current_analysis_path == model_path:  # Still relevant
                    self.root.after(0, lambda: self._update_ui_after_analysis(result))

            self.analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
            self.analysis_thread.start()

    def on_gguf_selected(self, event=None):
        """Handle GGUF file selection."""
        selected = SafeVar.get_str(self.selected_gguf_var)
        if selected:
            full_path = os.path.join(SafeVar.get_str(self.gguf_dir_var), selected)
            self.load_settings_for_model(full_path)

            # Default model alias to GGUF filename (without extension) if not set
            if not SafeVar.get_str(self.model_alias_var):
                # Remove .gguf extension
                model_name = selected[:-5] if selected.lower().endswith(".gguf") else selected
                self.model_alias_var.set(model_name)

            # Start GGUF analysis in background
            self._start_analysis(full_path)

            self.update_command_preview()
            self.status_var.set(f"Selected: {selected}")

    def get_model_full_path(self) -> str:
        """Get the full path of the selected model."""
        gguf_dir = SafeVar.get_str(self.gguf_dir_var)
        selected = SafeVar.get_str(self.selected_gguf_var)
        if gguf_dir and selected:
            return os.path.join(gguf_dir, selected)
        return ""

    def build_command(self) -> list[str]:
        """Build the command arguments list."""
        cmd = [SafeVar.get_str(self.llama_server_path_var)]

        model_path = self.get_model_full_path()
        if model_path:
            cmd.extend(["-m", model_path])

        # Common parameters
        host = SafeVar.get_str(self.host_var)
        if host:
            cmd.extend(["--host", host])

        port = SafeVar.get_str(self.port_var)
        if port:
            cmd.extend(["--port", port])

        ngl = SafeVar.get_str(self.ngl_var)
        if ngl:
            cmd.extend(["-ngl", ngl])

        ncmoe = SafeVar.get_str(self.ncmoe_var)
        if ncmoe:
            cmd.extend(["-ncmoe", ncmoe])

        if self.jinja_var.get():
            cmd.append("--jinja")

        threads = SafeVar.get_str(self.threads_var)
        if threads:
            cmd.extend(["--threads", threads])

        ctx_size = SafeVar.get_str(self.ctx_size_var)
        if ctx_size:
            cmd.extend(["--ctx-size", ctx_size])

        temp = SafeVar.get_str(self.temp_var)
        if temp:
            cmd.extend(["--temp", temp])

        min_p = SafeVar.get_str(self.min_p_var)
        if min_p:
            cmd.extend(["--min-p", min_p])

        top_p = SafeVar.get_str(self.top_p_var)
        if top_p:
            cmd.extend(["--top-p", top_p])

        top_k = SafeVar.get_str(self.top_k_var)
        if top_k:
            cmd.extend(["--top-k", top_k])

        presence_penalty = SafeVar.get_str(self.presence_penalty_var)
        if presence_penalty:
            cmd.extend(["--presence-penalty", presence_penalty])

        # Additional parameters
        mmproj = SafeVar.get_str(self.mmproj_path_var)
        if mmproj:
            cmd.extend(["--mmproj", mmproj])

        # OpenAI-compatible API settings
        api_key = SafeVar.get_str(self.api_key_var)
        if api_key:
            cmd.extend(["--api-key", api_key])

        model_alias = SafeVar.get_str(self.model_alias_var)
        if model_alias:
            cmd.extend(["--alias", model_alias])

        chat_template = SafeVar.get_str(self.chat_template_var)
        if chat_template:
            cmd.extend(["--chat-template", chat_template])

        chat_template_file = SafeVar.get_str(self.chat_template_file_var)
        if chat_template_file:
            cmd.extend(["--chat-template-file", chat_template_file])

        batch_size = SafeVar.get_str(self.batch_size_var)
        if batch_size:
            cmd.extend(["-b", batch_size])

        ubatch_size = SafeVar.get_str(self.ubatch_size_var)
        if ubatch_size:
            cmd.extend(["-ub", ubatch_size])

        parallel = SafeVar.get_str(self.parallel_var)
        if parallel:
            cmd.extend(["--parallel", parallel])

        n_predict = SafeVar.get_str(self.n_predict_var)
        if n_predict:
            cmd.extend(["-n", n_predict])

        repeat_penalty = SafeVar.get_str(self.repeat_penalty_var)
        if repeat_penalty:
            cmd.extend(["--repeat-penalty", repeat_penalty])

        frequency_penalty = SafeVar.get_str(self.frequency_penalty_var)
        if frequency_penalty:
            cmd.extend(["--frequency-penalty", frequency_penalty])

        rope_freq_base = SafeVar.get_str(self.rope_freq_base_var)
        if rope_freq_base:
            cmd.extend(["--rope-freq-base", rope_freq_base])

        rope_freq_scale = SafeVar.get_str(self.rope_freq_scale_var)
        if rope_freq_scale:
            cmd.extend(["--rope-freq-scale", rope_freq_scale])

        # Flags
        if self.flash_attn_var.get():
            cmd.extend(["-fa", "on"])

        if self.mlock_var.get():
            cmd.append("--mlock")

        if self.no_mmap_var.get():
            cmd.append("--no-mmap")

        if self.cont_batching_var.get():
            cmd.append("-cb")

        if self.metrics_var.get():
            cmd.append("--metrics")

        if self.verbose_var.get():
            cmd.append("--verbose")

        if self.log_disable_var.get():
            cmd.append("--log-disable")

        # Custom arguments
        custom = SafeVar.get_str(self.custom_args_var)
        if custom:
            cmd.extend(custom.split())

        return cmd

    def build_command_string(self) -> str:
        """Build a formatted command string for display."""
        cmd = self.build_command()
        if len(cmd) < 2:
            return "# No model selected"

        def quote_if_needed(s: str) -> str:
            """Quote string if it contains special characters."""
            if not s:
                return s
            # Use shlex.quote for proper shell escaping
            if any(c in s for c in ' \t\n\'"\\$`!'):
                return shlex.quote(s)
            return s

        # Format with line continuations
        lines = [cmd[0]]  # Executable
        i = 1
        while i < len(cmd):
            if cmd[i].startswith("-"):
                if i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
                    lines.append(f"  {cmd[i]} {quote_if_needed(cmd[i + 1])}")
                    i += 2
                else:
                    lines.append(f"  {cmd[i]}")
                    i += 1
            else:
                lines.append(f"  {quote_if_needed(cmd[i])}")
                i += 1

        return " \\\n".join(lines)

    def update_command_preview(self):
        """Update the command preview text."""
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert("1.0", self.build_command_string())

    def copy_command(self):
        """Copy the command to clipboard."""
        cmd = self.build_command_string()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status_var.set("Command copied to clipboard")

    def launch_server(self):
        """Launch the llama-server."""
        # Validate
        server_path = SafeVar.get_str(self.llama_server_path_var)
        if not server_path or not os.path.isfile(server_path):
            messagebox.showerror("Error", "Please select a valid llama-server executable")
            return

        if not os.access(server_path, os.X_OK):
            messagebox.showerror("Error", "llama-server is not executable. Run: chmod +x " + server_path)
            return

        model_path = self.get_model_full_path()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid GGUF model file")
            return

        # Save settings before launching
        self.save_current_settings()

        cmd = self.build_command()

        if self.run_in_terminal_var.get():
            # Run in terminal
            try:
                # Try different terminal emulators
                terminals = [
                    ["gnome-terminal", "--", "bash", "-c"],
                    ["konsole", "-e", "bash", "-c"],
                    ["xfce4-terminal", "-e", "bash -c"],
                    ["xterm", "-e", "bash", "-c"],
                ]

                # Use shlex.quote for proper shell escaping of all arguments
                cmd_str = " ".join(shlex.quote(c) for c in cmd)
                full_cmd_str = f'{cmd_str}; echo "\\nPress Enter to close..."; read'

                launched = False
                for term_cmd in terminals:
                    try:
                        if term_cmd[0] == "gnome-terminal":
                            subprocess.Popen(term_cmd + [full_cmd_str])
                        else:
                            subprocess.Popen(term_cmd + [full_cmd_str])
                        launched = True
                        break
                    except FileNotFoundError:
                        continue

                if launched:
                    self.status_var.set("Server launched in terminal")
                    self.show_api_info()
                else:
                    messagebox.showerror("Error", "No supported terminal emulator found")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to launch server: {e}")
        else:
            # Run in background
            try:
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
                self.status_var.set(f"Server launched in background (PID: {self.server_process.pid})")
                self.show_api_info()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to launch server: {e}")

    def show_api_info(self):
        """Show OpenAI-compatible API endpoint information."""
        host = SafeVar.get_str(self.host_var) or "localhost"
        port = SafeVar.get_str(self.port_var) or "8033"
        api_key = SafeVar.get_str(self.api_key_var)
        model_alias = SafeVar.get_str(self.model_alias_var) or "default"

        # Use localhost for display if bound to all interfaces
        display_host = "localhost" if host == "0.0.0.0" else host
        base_url = f"http://{display_host}:{port}"

        info = f"""Server starting at: {base_url}

═══ OpenAI-Compatible API Endpoints ═══

• Chat Completions:  {base_url}/v1/chat/completions
• Text Completions:  {base_url}/v1/completions
• List Models:       {base_url}/v1/models
• Embeddings:        {base_url}/v1/embeddings
• Health Check:      {base_url}/health

═══ Connection Settings ═══

Base URL:    {base_url}/v1
Model Name:  {model_alias}
API Key:     {"Required: " + api_key if api_key else "Not required"}

═══ Python Example ═══

from openai import OpenAI

client = OpenAI(
    base_url="{base_url}/v1",
    api_key="{api_key if api_key else 'not-needed'}"
)

response = client.chat.completions.create(
    model="{model_alias}",
    messages=[{{"role": "user", "content": "Hello!"}}]
)
print(response.choices[0].message.content)
"""

        # Create info dialog
        info_window = tk.Toplevel(self.root)
        info_window.title("API Endpoint Information")
        info_window.geometry("550x580")
        info_window.transient(self.root)

        # Text widget with scrollbar
        frame = ttk.Frame(info_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        text = tk.Text(frame, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)

        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text.insert("1.0", info)
        text.configure(state="disabled")

        # Buttons
        btn_frame = ttk.Frame(info_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def copy_base_url():
            self.root.clipboard_clear()
            self.root.clipboard_append(f"{base_url}/v1")
            self.status_var.set("Base URL copied to clipboard")

        ttk.Button(btn_frame, text="Copy Base URL", command=copy_base_url).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Close", command=info_window.destroy).pack(side=tk.RIGHT)

    def kill_server(self):
        """Kill the llama-server process."""
        # First try to kill our own process
        if self.server_process is not None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.status_var.set("Background server stopped")
                self.server_process = None
                return
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process = None
                self.status_var.set("Background server killed")
                return
            except Exception as e:
                self.status_var.set(f"Error stopping server: {e}")

        # If no background process, ask about using pkill
        result = messagebox.askyesno(
            "Kill llama-server",
            "No background server found. Do you want to kill ALL running llama-server processes?\n\n"
            "This will use 'pkill llama-server' to terminate any running instances.",
            icon="warning"
        )

        if result:
            try:
                subprocess.run(["pkill", "-f", "llama-server"], check=False)
                self.status_var.set("Sent kill signal to all llama-server processes")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to kill processes: {e}")

    def get_current_settings(self) -> dict:
        """Get all current settings as a dictionary."""
        return {
            "host": SafeVar.get_str(self.host_var),
            "port": SafeVar.get_str(self.port_var),
            "ngl": SafeVar.get_str(self.ngl_var),
            "ncmoe": SafeVar.get_str(self.ncmoe_var),
            "jinja": self.jinja_var.get(),
            "threads": SafeVar.get_str(self.threads_var),
            "ctx_size": SafeVar.get_str(self.ctx_size_var),
            "temp": SafeVar.get_str(self.temp_var),
            "min_p": SafeVar.get_str(self.min_p_var),
            "top_p": SafeVar.get_str(self.top_p_var),
            "top_k": SafeVar.get_str(self.top_k_var),
            "presence_penalty": SafeVar.get_str(self.presence_penalty_var),
            "mmproj": SafeVar.get_str(self.mmproj_path_var),
            "api_key": SafeVar.get_str(self.api_key_var),
            "model_alias": SafeVar.get_str(self.model_alias_var),
            "chat_template": SafeVar.get_str(self.chat_template_var),
            "chat_template_file": SafeVar.get_str(self.chat_template_file_var),
            "batch_size": SafeVar.get_str(self.batch_size_var),
            "ubatch_size": SafeVar.get_str(self.ubatch_size_var),
            "parallel": SafeVar.get_str(self.parallel_var),
            "n_predict": SafeVar.get_str(self.n_predict_var),
            "repeat_penalty": SafeVar.get_str(self.repeat_penalty_var),
            "frequency_penalty": SafeVar.get_str(self.frequency_penalty_var),
            "rope_freq_base": SafeVar.get_str(self.rope_freq_base_var),
            "rope_freq_scale": SafeVar.get_str(self.rope_freq_scale_var),
            "flash_attn": self.flash_attn_var.get(),
            "mlock": self.mlock_var.get(),
            "no_mmap": self.no_mmap_var.get(),
            "cont_batching": self.cont_batching_var.get(),
            "metrics": self.metrics_var.get(),
            "verbose": self.verbose_var.get(),
            "log_disable": self.log_disable_var.get(),
            "custom_args": SafeVar.get_str(self.custom_args_var),
            "run_in_terminal": self.run_in_terminal_var.get(),
        }

    def apply_settings(self, settings: dict):
        """Apply settings from a dictionary."""
        self.host_var.set(settings.get("host", "0.0.0.0"))
        self.port_var.set(settings.get("port", "8033"))
        self.ngl_var.set(settings.get("ngl", "99"))
        self.ncmoe_var.set(settings.get("ncmoe", ""))
        self.jinja_var.set(settings.get("jinja", True))
        self.threads_var.set(settings.get("threads", "-1"))
        self.ctx_size_var.set(settings.get("ctx_size", "8192"))
        self.temp_var.set(settings.get("temp", "0.7"))
        self.min_p_var.set(settings.get("min_p", "0.0"))
        self.top_p_var.set(settings.get("top_p", "0.9"))
        self.top_k_var.set(settings.get("top_k", "40"))
        self.presence_penalty_var.set(settings.get("presence_penalty", "0.0"))
        self.mmproj_path_var.set(settings.get("mmproj", ""))
        self.api_key_var.set(settings.get("api_key", ""))
        self.model_alias_var.set(settings.get("model_alias", ""))
        self.chat_template_var.set(settings.get("chat_template", ""))
        self.chat_template_file_var.set(settings.get("chat_template_file", ""))
        self.batch_size_var.set(settings.get("batch_size", ""))
        self.ubatch_size_var.set(settings.get("ubatch_size", ""))
        self.parallel_var.set(settings.get("parallel", ""))
        self.n_predict_var.set(settings.get("n_predict", ""))
        self.repeat_penalty_var.set(settings.get("repeat_penalty", ""))
        self.frequency_penalty_var.set(settings.get("frequency_penalty", ""))
        self.rope_freq_base_var.set(settings.get("rope_freq_base", ""))
        self.rope_freq_scale_var.set(settings.get("rope_freq_scale", ""))
        self.flash_attn_var.set(settings.get("flash_attn", False))
        self.mlock_var.set(settings.get("mlock", False))
        self.no_mmap_var.set(settings.get("no_mmap", False))
        self.cont_batching_var.set(settings.get("cont_batching", True))
        self.metrics_var.set(settings.get("metrics", False))
        self.verbose_var.set(settings.get("verbose", False))
        self.log_disable_var.set(settings.get("log_disable", False))
        self.custom_args_var.set(settings.get("custom_args", ""))
        self.run_in_terminal_var.set(settings.get("run_in_terminal", True))

    def save_current_settings(self):
        """Save current settings for the selected model."""
        model_path = self.get_model_full_path()
        if not model_path:
            self.status_var.set("No model selected to save settings for")
            return

        settings = self.get_current_settings()

        if "model_settings" not in self.config:
            self.config["model_settings"] = {}

        self.config["model_settings"][model_path] = settings
        self.save_config()
        self.status_var.set(f"Settings saved for {os.path.basename(model_path)}")

    def load_settings_for_model(self, model_path: str):
        """Load saved settings for a specific model."""
        if "model_settings" in self.config and model_path in self.config["model_settings"]:
            self.apply_settings(self.config["model_settings"][model_path])
            self.status_var.set(f"Loaded settings for {os.path.basename(model_path)}")
        else:
            # Apply defaults
            self.apply_settings({})
            self.status_var.set(f"Using default settings for {os.path.basename(model_path)}")

    def load_config(self) -> dict:
        """Load configuration from file."""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        return {}

    def save_config(self):
        """Save configuration to file."""
        try:
            self.config["llama_server_path"] = SafeVar.get_str(self.llama_server_path_var)
            self.config["last_gguf_dir"] = SafeVar.get_str(self.gguf_dir_var)
            self.config["last_selected_gguf"] = SafeVar.get_str(self.selected_gguf_var)

            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_last_session(self):
        """Load the last session settings."""
        # Load last GGUF directory
        last_dir = self.config.get("last_gguf_dir", "")
        if last_dir and os.path.isdir(last_dir):
            self.gguf_dir_var.set(last_dir)
            self.refresh_gguf_list()

            # Select last used GGUF if still available
            last_gguf = self.config.get("last_selected_gguf", "")
            if last_gguf and last_gguf in self.gguf_files:
                self.selected_gguf_var.set(last_gguf)
                self.on_gguf_selected()

    def on_close(self):
        """Handle application close."""
        self.save_config()

        # Warn about background process
        if self.server_process is not None:
            result = messagebox.askyesno(
                "Server Running",
                "A background server is still running. Do you want to stop it before closing?",
                icon="warning"
            )
            if result:
                self.kill_server()

        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()

    # Try to set a modern theme
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "alt" in available_themes:
            style.theme_use("alt")
    except Exception:
        pass

    app = LlamaServerLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
