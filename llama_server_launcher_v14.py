#!/usr/bin/env python3
"""
Llama Server Launcher - A PyQt6 GUI for launching llama-server with common parameters.
Supports persistent settings per GGUF file, process management, and command preview.
"""

import sys
import os
import json
import subprocess
import shlex
import tempfile
import threading
from datetime import datetime
from typing import Optional, Any, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox, QRadioButton,
    QComboBox, QPlainTextEdit, QListWidget, QScrollArea,
    QFileDialog, QMessageBox, QFrame, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

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
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llama_server_launcher_config.json")
DEFAULT_LLAMA_SERVER_PATH = os.path.expanduser("~/llama.cpp/llama-server")


class SafeVar:
    """Helper that works with QLineEdit widgets to safely extract values."""

    @staticmethod
    def get_int(widget: QLineEdit, default: int = 0) -> int:
        """Safely get integer value from a QLineEdit."""
        try:
            value = widget.text().strip()
            if value == "" or value == "-":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def get_float(widget: QLineEdit, default: float = 0.0) -> float:
        """Safely get float value from a QLineEdit."""
        try:
            value = widget.text().strip()
            if value == "" or value == "-" or value == ".":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def get_str(widget: QLineEdit, default: str = "") -> str:
        """Safely get string value from a QLineEdit."""
        try:
            return widget.text().strip()
        except (TypeError, AttributeError):
            return default


class AnalysisSignal(QObject):
    """Signal class for thread-safe UI updates from analysis worker."""
    finished = pyqtSignal(dict)


class LlamaServerLauncher(QMainWindow):
    """Main application class for Llama Server Launcher."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama Server Launcher")
        self.setGeometry(100, 100, 750, 850)
        self.setMinimumSize(700, 750)

        # Process handle for background server
        self.server_process: Optional[subprocess.Popen] = None
        self.server_log_file = None
        self.server_log_path: Optional[str] = None

        # Load configuration
        self.config = self.load_config()

        # Analysis signal for thread-safe UI updates
        self.analysis_signal = AnalysisSignal()
        self.analysis_signal.finished.connect(self._update_ui_after_analysis)

        # Initialize variables (store widget references)
        self.init_variables()

        # Build UI
        self.build_ui()

        # Load last used directory and settings
        self.load_last_session()

    def init_variables(self):
        """Initialize all widget references and data stores."""
        # Path variables - store default values, widgets created in build_ui
        self.llama_server_path = self.config.get("llama_server_path", DEFAULT_LLAMA_SERVER_PATH)
        self.gguf_dir = self.config.get("last_gguf_dir", "")

        # Common parameters - default values
        self.host_default = "0.0.0.0"
        self.port_default = "8033"
        self.ngl_default = "99"
        self.ncmoe_default = ""
        self.jinja_default = True
        self.threads_default = "-1"
        self.ctx_size_default = "8192"
        self.temp_default = "0.7"
        self.min_p_default = "0.0"
        self.top_p_default = "0.9"
        self.top_k_default = "40"
        self.presence_penalty_default = "0.0"

        # Additional parameters - default values
        self.mmproj_path_default = ""
        self.batch_size_default = ""
        self.ubatch_size_default = ""
        self.n_predict_default = ""
        self.rope_freq_base_default = ""
        self.rope_freq_scale_default = ""
        self.repeat_penalty_default = ""
        self.frequency_penalty_default = ""
        self.flash_attn_default = False
        self.mlock_default = False
        self.no_mmap_default = False
        self.cont_batching_default = True
        self.metrics_default = False
        self.verbose_default = False
        self.log_disable_default = False
        self.no_mmproj_offload_default = False
        self.no_mmproj_default = False
        self.cpu_moe_default = False
        self.parallel_default = ""
        self.spec_type_enabled_default = False
        self.spec_type_default = "draft-mtp"
        self.spec_draft_n_max_enabled_default = False
        self.spec_draft_n_max_default = "2"
        self.custom_args_default = ""

        # KV Cache settings
        self.cache_type_k_default = ""
        self.cache_type_v_default = ""
        self.cache_type_options = ["", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]

        # OpenAI-compatible API settings
        self.api_key_default = ""
        self.model_alias_default = ""
        self.chat_template_default = ""
        self.chat_template_file_default = ""

        # Run mode
        self.run_in_terminal_default = True

        # Available GGUF files list
        self.gguf_files: list[str] = []

        # Model analysis info
        self.analysis_thread: Optional[threading.Thread] = None
        self.current_analysis_path: Optional[str] = None
        self.last_analysis_result: Optional[Dict[str, Any]] = None

        # Favorites - dict of {model_path: {"note": "...", "added": "timestamp"}}
        self.favorites: Dict[str, Dict[str, str]] = self.config.get("favorites", {})

        # Presets - dict of {preset_name: {settings_dict}}
        self.presets: Dict[str, dict] = self.config.get("presets", {})

        # Tracks which preset each model last used: {model_path: preset_name}
        self.model_preset_map: Dict[str, str] = self.config.get("model_preset_map", {})

        # Clipboard for copy/paste preset between models
        self.clipboard_settings: Optional[dict] = None
        self.clipboard_source: str = ""  # model name the settings were copied from

    def build_ui(self):
        """Build the main user interface."""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Fixed top bar: Run Mode, Control Buttons, Status Indicator ---
        self.build_top_bar(main_layout)

        # --- Scrollable area for main content ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(5)

        # Llama server path section
        self.build_server_path_section(scroll_layout)

        # GGUF file selection section
        self.build_gguf_section(scroll_layout)

        # Tab widget
        self.tab_widget = QTabWidget()

        # Common parameters tab
        common_tab = QWidget()
        self.tab_widget.addTab(common_tab, "Common Parameters")
        self.build_common_params_tab(common_tab)

        # Additional parameters tab
        additional_tab = QWidget()
        self.tab_widget.addTab(additional_tab, "Additional Parameters")
        self.build_additional_params_tab(additional_tab)

        # Presets tab
        presets_tab = QWidget()
        self.tab_widget.addTab(presets_tab, "Presets")
        self.build_presets_tab(presets_tab)

        scroll_layout.addWidget(self.tab_widget)

        # Command preview section
        self.build_command_preview_section(scroll_layout)

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

    def build_top_bar(self, layout: QVBoxLayout):
        """Build the fixed top bar with Run Mode, Control Buttons, and Status Indicator."""
        top_frame = QFrame()
        top_frame.setObjectName("topBar")
        top_frame.setStyleSheet("""
            QFrame#topBar { background-color: #363636; }
        """)
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(10, 8, 10, 8)
        top_layout.setSpacing(8)

        # --- Row 1: Run Mode + Control Buttons ---
        row1 = QHBoxLayout()

        # Run Mode
        mode_groupbox = QGroupBox("Run Mode")
        mode_groupbox.setStyleSheet("""
            QGroupBox { color: #aaaaaa; border: 1px solid #555555; border-radius: 3px; margin-top: 8px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 7px; padding: 0 3px; color: #aaaaaa; }
            QRadioButton { color: #cccccc; background-color: #363636; }
            QRadioButton::indicator:checked { background-color: #4CAF50; }
        """)
        mode_layout = QHBoxLayout(mode_groupbox)
        mode_layout.setContentsMargins(5, 5, 5, 5)

        self.run_terminal_radio = QRadioButton("Run in Terminal")
        self.run_terminal_radio.setChecked(True)
        self.run_terminal_radio.setStyleSheet("color: #cccccc; background-color: #363636;")
        mode_layout.addWidget(self.run_terminal_radio)

        self.run_bg_radio = QRadioButton("Run in Background")
        self.run_bg_radio.setStyleSheet("color: #cccccc; background-color: #363636;")
        mode_layout.addWidget(self.run_bg_radio)

        row1.addWidget(mode_groupbox)
        row1.addStretch()

        # Control Buttons
        btn_frame = QHBoxLayout()

        self.launch_btn = QPushButton("\u25b6 Launch Server")
        self.launch_btn.clicked.connect(self.launch_server)
        self.launch_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border: 1px solid #4CAF50;
                         padding: 3px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:pressed { background-color: #2E7D32; }
        """)
        btn_frame.addWidget(self.launch_btn)

        self.kill_btn = QPushButton("\u25a0 Kill Server")
        self.kill_btn.clicked.connect(self.kill_server)
        self.kill_btn.setStyleSheet("""
            QPushButton { background-color: #F44336; color: white; border: 1px solid #F44336;
                         padding: 3px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #C62828; }
            QPushButton:pressed { background-color: #B71C1C; }
        """)
        btn_frame.addWidget(self.kill_btn)

        self.save_btn = QPushButton("\U0001f4be Save Settings")
        self.save_btn.clicked.connect(self.save_current_settings)
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; border: 1px solid #2196F3;
                         padding: 3px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:pressed { background-color: #0D47A1; }
        """)
        btn_frame.addWidget(self.save_btn)

        row1.addLayout(btn_frame)
        top_layout.addLayout(row1)

        # --- Row 2: Status Indicator ---
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #2a2a2a;")
        status_inner = QVBoxLayout(status_frame)
        status_inner.setContentsMargins(1, 1, 1, 1)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet("""
            QLabel { background-color: #2e2e2e; color: #aaaaaa; padding: 3px 6px; }
        """)
        status_inner.addWidget(self.status_label)

        top_layout.addWidget(status_frame)

        layout.addWidget(top_frame)

        # Thin accent line
        accent = QFrame()
        accent.setFixedHeight(2)
        accent.setStyleSheet("background-color: #4CAF50;")
        layout.addWidget(accent)

    def build_server_path_section(self, layout: QVBoxLayout):
        """Build the llama-server path selection section."""
        groupbox = QGroupBox("Llama Server Executable")
        gb_layout = QVBoxLayout(groupbox)

        path_layout = QHBoxLayout()
        self.server_path_edit = QLineEdit(self.llama_server_path)
        self.server_path_edit.setFixedWidth(400)
        path_layout.addWidget(self.server_path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_llama_server)
        path_layout.addWidget(browse_btn)

        gb_layout.addLayout(path_layout)
        layout.addWidget(groupbox)

    def build_gguf_section(self, layout: QVBoxLayout):
        """Build the GGUF file selection section."""
        groupbox = QGroupBox("Model Selection")
        gb_layout = QVBoxLayout(groupbox)

        # Directory selection
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("GGUF Directory:"))
        self.gguf_dir_edit = QLineEdit(self.gguf_dir)
        dir_layout.addWidget(self.gguf_dir_edit, 1)
        self.browse_gguf_btn = QPushButton("Browse...")
        self.browse_gguf_btn.clicked.connect(self.browse_gguf_dir)
        dir_layout.addWidget(self.browse_gguf_btn)
        gb_layout.addLayout(dir_layout)

        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setFixedWidth(200)
        self.search_edit.textChanged.connect(self.filter_gguf_list)
        search_layout.addWidget(self.search_edit)
        clear_search_btn = QPushButton("Clear")
        clear_search_btn.clicked.connect(lambda: self.search_edit.setText(""))
        search_layout.addWidget(clear_search_btn)
        gb_layout.addLayout(search_layout)

        # GGUF file dropdown
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Select Model:"))
        self.gguf_combo = QComboBox()
        self.gguf_combo.setEditable(False)
        self.gguf_combo.currentTextChanged.connect(self.on_gguf_selected)
        file_layout.addWidget(self.gguf_combo, 1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_gguf_list)
        file_layout.addWidget(self.refresh_btn)
        self.favorite_btn = QPushButton("\u2606")
        self.favorite_btn.setFixedWidth(30)
        self.favorite_btn.clicked.connect(self.toggle_favorite)
        file_layout.addWidget(self.favorite_btn)
        gb_layout.addLayout(file_layout)

        # Model info display
        self.model_info_label = QLabel("No model selected")
        self.model_info_label.setStyleSheet("color: gray;")
        self.model_info_label.setFont(QFont("TkDefaultFont", 9))
        gb_layout.addWidget(self.model_info_label)

        # Favorite note display/edit
        note_layout = QHBoxLayout()
        note_layout.addWidget(QLabel("Note:"))
        self.note_edit = QLineEdit()
        note_layout.addWidget(self.note_edit, 1)
        self.save_note_btn = QPushButton("Save Note")
        self.save_note_btn.clicked.connect(self.save_note)
        self.note_edit.returnPressed.connect(self.save_note)
        note_layout.addWidget(self.save_note_btn)
        gb_layout.addLayout(note_layout)

        layout.addWidget(groupbox)

    def build_common_params_tab(self, parent: QWidget):
        """Build the common parameters tab."""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # Create scrollable area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Network settings
        network_gb = QGroupBox("Network Settings")
        network_layout = QVBoxLayout(network_gb)
        self.host_edit = self.create_param_row(network_layout, "Host (--host):", self.host_default, "IP address to bind (0.0.0.0 for all)")
        self.port_edit = self.create_param_row(network_layout, "Port (--port):", self.port_default, "Port number (default: 8033)")
        scroll_layout.addWidget(network_gb)

        # GPU/Performance settings
        perf_gb = QGroupBox("Performance Settings")
        perf_layout = QVBoxLayout(perf_gb)
        self.ngl_edit = self.create_param_row(perf_layout, "GPU Layers (-ngl):", self.ngl_default, "Number of layers to offload to GPU (99 = all)")
        self.ncmoe_edit = self.create_param_row(perf_layout, "MoE Experts (-ncmoe):", self.ncmoe_default, "Number of MoE experts to use (leave empty for default)")
        self.threads_edit = self.create_param_row(perf_layout, "Threads (--threads):", self.threads_default, "Number of threads (-1 = auto)")
        self.ctx_size_edit = self.create_param_row(perf_layout, "Context Size (--ctx-size):", self.ctx_size_default, "Context window size in tokens")
        scroll_layout.addWidget(perf_gb)

        # Sampling settings
        sampling_gb = QGroupBox("Sampling Settings")
        sampling_layout = QVBoxLayout(sampling_gb)
        self.temp_edit = self.create_param_row(sampling_layout, "Temperature (--temp):", self.temp_default, "Sampling temperature (0.0-2.0)")
        self.min_p_edit = self.create_param_row(sampling_layout, "Min P (--min-p):", self.min_p_default, "Minimum probability threshold (0.0-1.0)")
        self.top_p_edit = self.create_param_row(sampling_layout, "Top P (--top-p):", self.top_p_default, "Top-p sampling probability (0.0-1.0)")
        self.top_k_edit = self.create_param_row(sampling_layout, "Top K (--top-k):", self.top_k_default, "Top-k filtering (0 = disabled)")
        self.presence_penalty_edit = self.create_param_row(sampling_layout, "Presence Penalty:", self.presence_penalty_default, "Presence penalty (-2.0 to 2.0)")
        self.repeat_penalty_edit = self.create_param_row(sampling_layout, "Repeat Penalty:", self.repeat_penalty_default, "Repeat penalty (1.0 = disabled)")
        scroll_layout.addWidget(sampling_gb)

        # Flags
        flags_gb = QGroupBox("Flags")
        flags_layout = QVBoxLayout(flags_gb)
        self.jinja_check = QCheckBox("Enable Jinja templates (--jinja)")
        self.jinja_check.setChecked(self.jinja_default)
        flags_layout.addWidget(self.jinja_check)
        scroll_layout.addWidget(flags_gb)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)

    def build_additional_params_tab(self, parent: QWidget):
        """Build the additional parameters tab."""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # OpenAI-compatible API settings
        api_gb = QGroupBox("OpenAI-Compatible API Settings")
        api_layout = QVBoxLayout(api_gb)
        self.api_key_edit = self.create_param_row(api_layout, "API Key (--api-key):", self.api_key_default, "API key for authentication (optional)")
        self.model_alias_edit = self.create_param_row(api_layout, "Model Alias (--alias):", self.model_alias_default, "Model name returned by /v1/models")
        self.chat_template_edit = self.create_param_row(api_layout, "Chat Template (--chat-template):", self.chat_template_default, "Jinja2 chat template (optional)")

        template_file_layout = QHBoxLayout()
        template_file_layout.addWidget(QLabel("Chat Template File:"))
        template_file_layout.setSpacing(5)
        self.chat_template_file_edit = QLineEdit(self.chat_template_file_default)
        template_file_layout.addWidget(self.chat_template_file_edit, 1)
        self.browse_template_btn = QPushButton("Browse...")
        self.browse_template_btn.clicked.connect(self.browse_chat_template_file)
        template_file_layout.addWidget(self.browse_template_btn)
        api_layout.addLayout(template_file_layout)

        api_hint = QLabel("Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/embeddings")
        api_hint.setStyleSheet("color: gray;")
        api_layout.addWidget(api_hint)
        scroll_layout.addWidget(api_gb)

        # Multimodal settings
        mm_gb = QGroupBox("Multimodal Settings")
        mm_layout = QVBoxLayout(mm_gb)
        mmproj_layout = QHBoxLayout()
        mmproj_layout.addWidget(QLabel("MMProj Path (--mmproj):"))
        self.mmproj_edit = QLineEdit(self.mmproj_path_default)
        mmproj_layout.addWidget(self.mmproj_edit, 1)
        self.browse_mmproj_btn = QPushButton("Browse...")
        self.browse_mmproj_btn.clicked.connect(self.browse_mmproj)
        mmproj_layout.addWidget(self.browse_mmproj_btn)
        mm_layout.addLayout(mmproj_layout)
        scroll_layout.addWidget(mm_gb)

        # Batch settings
        batch_gb = QGroupBox("Batch Settings")
        batch_layout = QVBoxLayout(batch_gb)
        self.batch_size_edit = self.create_param_row(batch_layout, "Batch Size (-b):", self.batch_size_default, "Logical batch size (leave empty for default)")
        self.ubatch_size_edit = self.create_param_row(batch_layout, "Micro Batch (-ub):", self.ubatch_size_default, "Physical batch size (leave empty for default)")
        self.parallel_edit = self.create_param_row(batch_layout, "Parallel Slots (--parallel):", self.parallel_default, "Number of parallel sequences")
        scroll_layout.addWidget(batch_gb)

        # Generation settings
        gen_gb = QGroupBox("Generation Settings")
        gen_layout = QVBoxLayout(gen_gb)
        self.n_predict_edit = self.create_param_row(gen_layout, "Max Predict (-n):", self.n_predict_default, "Max tokens to predict (-1 = infinite)")
        self.frequency_penalty_edit = self.create_param_row(gen_layout, "Frequency Penalty:", self.frequency_penalty_default, "Frequency penalty (0.0-2.0)")
        scroll_layout.addWidget(gen_gb)

        # RoPE settings
        rope_gb = QGroupBox("RoPE Settings")
        rope_layout = QVBoxLayout(rope_gb)
        self.rope_freq_base_edit = self.create_param_row(rope_layout, "RoPE Freq Base:", self.rope_freq_base_default, "RoPE frequency base (leave empty for default)")
        self.rope_freq_scale_edit = self.create_param_row(rope_layout, "RoPE Freq Scale:", self.rope_freq_scale_default, "RoPE frequency scale (leave empty for default)")
        scroll_layout.addWidget(rope_gb)

        # Speculative decoding settings
        spec_gb = QGroupBox("Speculative Decoding")
        spec_layout = QVBoxLayout(spec_gb)

        spec_type_layout = QHBoxLayout()
        self.spec_type_check = QCheckBox("--spec-type")
        self.spec_type_check.setChecked(self.spec_type_enabled_default)
        spec_type_layout.addWidget(self.spec_type_check)
        self.spec_type_combo = QComboBox()
        self.spec_type_combo.addItems(["none", "draft-simple", "draft-eagle3", "draft-mtp", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"])
        self.spec_type_combo.setCurrentText(self.spec_type_default)
        self.spec_type_combo.setFixedWidth(150)
        spec_type_layout.addWidget(self.spec_type_combo)
        spec_type_hint = QLabel("Speculative decoding type")
        spec_type_hint.setStyleSheet("color: gray;")
        spec_type_layout.addWidget(spec_type_hint)
        spec_layout.addLayout(spec_type_layout)

        spec_n_max_layout = QHBoxLayout()
        self.spec_n_max_check = QCheckBox("--spec-draft-n-max")
        self.spec_n_max_check.setChecked(self.spec_draft_n_max_enabled_default)
        spec_n_max_layout.addWidget(self.spec_n_max_check)
        self.spec_n_max_edit = QLineEdit(self.spec_draft_n_max_default)
        self.spec_n_max_edit.setFixedWidth(100)
        spec_n_max_layout.addWidget(self.spec_n_max_edit)
        spec_n_max_hint = QLabel("Number of tokens to draft for speculative decoding")
        spec_n_max_hint.setStyleSheet("color: gray;")
        spec_n_max_layout.addWidget(spec_n_max_hint)
        spec_layout.addLayout(spec_n_max_layout)
        scroll_layout.addWidget(spec_gb)

        # KV Cache settings
        cache_gb = QGroupBox("KV Cache Settings")
        cache_layout = QVBoxLayout(cache_gb)

        cache_k_layout = QHBoxLayout()
        cache_k_layout.addWidget(QLabel("Cache Type K (-ctk):"))
        self.cache_k_combo = QComboBox()
        self.cache_k_combo.addItems(self.cache_type_options)
        self.cache_k_combo.setCurrentText(self.cache_type_k_default)
        self.cache_k_combo.setFixedWidth(100)
        cache_k_layout.addWidget(self.cache_k_combo)
        cache_k_hint = QLabel("KV cache data type for K (empty = default)")
        cache_k_hint.setStyleSheet("color: gray;")
        cache_k_layout.addWidget(cache_k_hint)
        cache_layout.addLayout(cache_k_layout)

        cache_v_layout = QHBoxLayout()
        cache_v_layout.addWidget(QLabel("Cache Type V (-ctv):"))
        self.cache_v_combo = QComboBox()
        self.cache_v_combo.addItems(self.cache_type_options)
        self.cache_v_combo.setCurrentText(self.cache_type_v_default)
        self.cache_v_combo.setFixedWidth(100)
        cache_v_layout.addWidget(self.cache_v_combo)
        cache_v_hint = QLabel("KV cache data type for V (empty = default)")
        cache_v_hint.setStyleSheet("color: gray;")
        cache_v_layout.addWidget(cache_v_hint)
        cache_layout.addLayout(cache_v_layout)
        scroll_layout.addWidget(cache_gb)

        # Additional flags
        flags_gb = QGroupBox("Additional Flags")
        flags_layout = QVBoxLayout(flags_gb)
        self.flash_attn_check = QCheckBox("Flash Attention (-fa)")
        self.flash_attn_check.setChecked(self.flash_attn_default)
        flags_layout.addWidget(self.flash_attn_check)
        self.mlock_check = QCheckBox("Lock memory (--mlock)")
        self.mlock_check.setChecked(self.mlock_default)
        flags_layout.addWidget(self.mlock_check)
        self.no_mmap_check = QCheckBox("Disable mmap (--no-mmap)")
        self.no_mmap_check.setChecked(self.no_mmap_default)
        flags_layout.addWidget(self.no_mmap_check)
        self.cont_batching_check = QCheckBox("Continuous batching (-cb)")
        self.cont_batching_check.setChecked(self.cont_batching_default)
        flags_layout.addWidget(self.cont_batching_check)
        self.metrics_check = QCheckBox("Enable metrics (--metrics)")
        self.metrics_check.setChecked(self.metrics_default)
        flags_layout.addWidget(self.metrics_check)
        self.verbose_check = QCheckBox("Verbose output (--verbose)")
        self.verbose_check.setChecked(self.verbose_default)
        flags_layout.addWidget(self.verbose_check)
        self.log_disable_check = QCheckBox("Disable logging (--log-disable)")
        self.log_disable_check.setChecked(self.log_disable_default)
        flags_layout.addWidget(self.log_disable_check)
        self.no_mmproj_offload_check = QCheckBox("No MMProj Offload (--no-mmproj-offload)")
        self.no_mmproj_offload_check.setChecked(self.no_mmproj_offload_default)
        flags_layout.addWidget(self.no_mmproj_offload_check)
        self.no_mmproj_check = QCheckBox("No MMProj (--no-mmproj)")
        self.no_mmproj_check.setChecked(self.no_mmproj_default)
        flags_layout.addWidget(self.no_mmproj_check)
        self.cpu_moe_check = QCheckBox("CPU MoE (--cpu-moe)")
        self.cpu_moe_check.setChecked(self.cpu_moe_default)
        flags_layout.addWidget(self.cpu_moe_check)
        scroll_layout.addWidget(flags_gb)

        # Custom arguments
        custom_gb = QGroupBox("Custom Arguments")
        custom_layout = QVBoxLayout(custom_gb)
        custom_layout.addWidget(QLabel("Additional arguments (space-separated):"))
        self.custom_args_edit = QLineEdit(self.custom_args_default)
        custom_layout.addWidget(self.custom_args_edit)
        scroll_layout.addWidget(custom_gb)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)

    def create_param_row(self, layout: QVBoxLayout, label: str, default: str, tooltip: str) -> QLineEdit:
        """Create a parameter input row with label and entry. Returns the QLineEdit widget."""
        row_layout = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(200)
        row_layout.addWidget(lbl)

        edit = QLineEdit(default)
        edit.setFixedWidth(120)
        row_layout.addWidget(edit)

        hint = QLabel(tooltip)
        hint.setStyleSheet("color: gray;")
        row_layout.addWidget(hint)

        row_layout.addStretch()
        layout.addLayout(row_layout)

        return edit

    def build_presets_tab(self, parent: QWidget):
        """Build the presets management tab."""
        tab_layout = QVBoxLayout(parent)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # Active preset indicator for current model
        active_gb = QGroupBox("Current Model Preset")
        active_layout = QVBoxLayout(active_gb)
        self.active_preset_label = QLabel("No preset active")
        self.active_preset_label.setFont(QFont("TkDefaultFont", 10, QFont.Weight.Bold))
        active_layout.addWidget(self.active_preset_label)
        tab_layout.addWidget(active_gb)

        # Copy / Paste preset between models
        cp_gb = QGroupBox("Copy / Paste Between Models")
        cp_layout = QVBoxLayout(cp_gb)

        cp_btn_layout = QHBoxLayout()
        self.copy_preset_btn = QPushButton("Copy Current Model Settings")
        self.copy_preset_btn.clicked.connect(self.copy_preset_for_model)
        cp_btn_layout.addWidget(self.copy_preset_btn)
        self.paste_preset_btn = QPushButton("Paste to Current Model")
        self.paste_preset_btn.clicked.connect(self.paste_preset_for_model)
        cp_btn_layout.addWidget(self.paste_preset_btn)
        cp_layout.addLayout(cp_btn_layout)

        self.clipboard_info_label = QLabel("No settings copied")
        self.clipboard_info_label.setStyleSheet("color: gray;")
        cp_layout.addWidget(self.clipboard_info_label)
        tab_layout.addWidget(cp_gb)

        # Save / Load preset section
        manage_gb = QGroupBox("Save / Load Preset")
        manage_layout = QVBoxLayout(manage_gb)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Preset Name:"))
        self.preset_name_edit = QLineEdit()
        name_layout.addWidget(self.preset_name_edit, 1)
        manage_layout.addLayout(name_layout)

        btn_layout = QHBoxLayout()
        self.save_preset_btn = QPushButton("Save Current as Preset")
        self.save_preset_btn.clicked.connect(self.save_preset)
        btn_layout.addWidget(self.save_preset_btn)
        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self.load_preset)
        btn_layout.addWidget(self.load_preset_btn)
        self.delete_preset_btn = QPushButton("Delete Preset")
        self.delete_preset_btn.clicked.connect(self.delete_preset)
        btn_layout.addWidget(self.delete_preset_btn)
        manage_layout.addLayout(btn_layout)

        manage_hint = QLabel(
            "Type a preset name above, then Save to store or Load to apply.\n"
            "Loading a preset also links it to the currently selected model."
        )
        manage_hint.setStyleSheet("color: gray;")
        manage_layout.addWidget(manage_hint)
        tab_layout.addWidget(manage_gb)

        # Saved presets list
        list_gb = QGroupBox("Saved Presets")
        list_layout = QVBoxLayout(list_gb)
        self.preset_list = QListWidget()
        self.preset_list.itemClicked.connect(self.on_preset_list_select)
        self.preset_list.itemDoubleClicked.connect(self.load_preset)
        list_layout.addWidget(self.preset_list)
        tab_layout.addWidget(list_gb, 1)

        # Populate preset list
        self.refresh_preset_list()

    def refresh_preset_list(self):
        """Refresh the preset list with current presets."""
        self.preset_list.clear()
        for name in sorted(self.presets.keys()):
            self.preset_list.addItem(name)

    def on_preset_list_select(self, item):
        """When a preset is selected in the list, fill in the name entry."""
        self.preset_name_edit.setText(item.text())

    def save_preset(self):
        """Save the current parameter configuration as a named preset."""
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Preset Name Required", "Please type a preset name before saving.")
            return

        # Confirm overwrite if preset already exists
        if name in self.presets:
            reply = QMessageBox.question(
                self, "Overwrite Preset",
                f"Preset '{name}' already exists. Overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        settings = self.get_current_settings()
        self.presets[name] = settings
        self.config["presets"] = self.presets

        # Link this preset to the current model
        model_path = self.get_model_full_path()
        if model_path:
            self.model_preset_map[model_path] = name
            self.config["model_preset_map"] = self.model_preset_map
            self.active_preset_label.setText(f"Active preset: {name}")

        self.save_config()
        self.refresh_preset_list()
        self.status_label.setText(f"Preset '{name}' saved")

    def load_preset(self):
        """Load a named preset and apply its settings."""
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Preset Name Required", "Please type or select a preset name to load.")
            return

        if name not in self.presets:
            QMessageBox.critical(self, "Preset Not Found", f"No preset named '{name}' exists.")
            return

        self.apply_settings(self.presets[name])

        # Link this preset to the current model
        model_path = self.get_model_full_path()
        if model_path:
            self.model_preset_map[model_path] = name
            self.config["model_preset_map"] = self.model_preset_map
            self.active_preset_label.setText(f"Active preset: {name}")
            # Also persist as the model's settings so next load remembers
            if "model_settings" not in self.config:
                self.config["model_settings"] = {}
            self.config["model_settings"][model_path] = self.presets[name].copy()
            self.save_config()

        self.update_command_preview()
        self.status_label.setText(f"Preset '{name}' loaded")

    def delete_preset(self):
        """Delete a named preset."""
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Preset Name Required", "Please type or select a preset name to delete.")
            return

        if name not in self.presets:
            QMessageBox.critical(self, "Preset Not Found", f"No preset named '{name}' exists.")
            return

        reply = QMessageBox.question(
            self, "Delete Preset", f"Delete preset '{name}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        del self.presets[name]
        self.config["presets"] = self.presets

        # Remove any model associations pointing to this preset
        models_to_clear = [m for m, p in self.model_preset_map.items() if p == name]
        for m in models_to_clear:
            del self.model_preset_map[m]
        self.config["model_preset_map"] = self.model_preset_map

        # Update active preset label if it was pointing to the deleted preset
        model_path = self.get_model_full_path()
        if model_path and model_path in models_to_clear:
            self.active_preset_label.setText("No preset active")

        self.save_config()
        self.refresh_preset_list()
        self.preset_name_edit.setText("")
        self.status_label.setText(f"Preset '{name}' deleted")

    def update_active_preset_label(self):
        """Update the active preset label for the current model."""
        model_path = self.get_model_full_path()
        if model_path and model_path in self.model_preset_map:
            preset_name = self.model_preset_map[model_path]
            # Verify the preset still exists
            if preset_name in self.presets:
                self.active_preset_label.setText(f"Active preset: {preset_name}")
            else:
                # Preset was deleted, clean up stale reference
                del self.model_preset_map[model_path]
                self.config["model_preset_map"] = self.model_preset_map
                self.active_preset_label.setText("No preset active")
        else:
            self.active_preset_label.setText("No preset active")

    def copy_preset_for_model(self):
        """Copy the current model's settings so they can be pasted to another model."""
        model_path = self.get_model_full_path()
        if not model_path:
            self.status_label.setText("No model selected to copy settings from")
            return

        self.clipboard_settings = self.get_current_settings()
        # Exclude model_alias -- it should stay unique per model
        self.clipboard_settings.pop("model_alias", None)
        self.clipboard_source = os.path.basename(model_path)
        self.clipboard_info_label.setText(f"Copied settings from: {self.clipboard_source}")
        self.status_label.setText(f"Settings copied from {self.clipboard_source}")

    def paste_preset_for_model(self):
        """Paste previously copied settings onto the current model."""
        if not self.clipboard_settings:
            QMessageBox.information(self, "Nothing to Paste", "No settings have been copied yet.\n\n"
                                "Select a model, then click 'Copy Current Model Settings' first.")
            return

        model_path = self.get_model_full_path()
        if not model_path:
            self.status_label.setText("No model selected to paste settings to")
            return

        target_name = os.path.basename(model_path)
        reply = QMessageBox.question(
            self, "Paste Settings",
            f"Apply settings copied from '{self.clipboard_source}' "
            f"to '{target_name}'?\n\n"
            f"This will overwrite the current settings for this model.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Preserve the current model's alias before pasting
        current_alias = self.model_alias_edit.text().strip()

        self.apply_settings(self.clipboard_settings)

        # Restore model alias -- it's excluded from copy/paste
        self.model_alias_edit.setText(current_alias)

        # Persist to model_settings so it remembers next time
        if "model_settings" not in self.config:
            self.config["model_settings"] = {}
        saved = self.clipboard_settings.copy()
        saved["model_alias"] = current_alias
        self.config["model_settings"][model_path] = saved
        self.save_config()

        self.update_command_preview()
        self.status_label.setText(f"Settings pasted from {self.clipboard_source} \u2192 {target_name}")

    def build_command_preview_section(self, layout: QVBoxLayout):
        """Build the command preview section."""
        groupbox = QGroupBox("Command Preview")
        gb_layout = QVBoxLayout(groupbox)

        self.command_text = QPlainTextEdit()
        self.command_text.setReadOnly(True)
        self.command_text.setFont(QFont("Courier", 9))
        self.command_text.setMaximumBlockCount(50)
        gb_layout.addWidget(self.command_text)

        btn_layout = QHBoxLayout()
        update_btn = QPushButton("Update Preview")
        update_btn.clicked.connect(self.update_command_preview)
        btn_layout.addWidget(update_btn)
        copy_btn = QPushButton("Copy Command")
        copy_btn.clicked.connect(self.copy_command)
        btn_layout.addWidget(copy_btn)
        gb_layout.addLayout(btn_layout)

        layout.addWidget(groupbox)

    def browse_llama_server(self):
        """Browse for llama-server executable."""
        path = QFileDialog.getOpenFileName(
            self, "Select llama-server executable",
            os.path.dirname(self.server_path_edit.text()) or os.path.expanduser("~")
        )[0]
        if path:
            self.server_path_edit.setText(path)
            self.save_config()

    def browse_gguf_dir(self):
        """Browse for GGUF files directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select GGUF Files Directory",
            self.gguf_dir_edit.text() or os.path.expanduser("~")
        )
        if path:
            self.gguf_dir_edit.setText(path)
            self.refresh_gguf_list()
            self.save_config()

    def browse_mmproj(self):
        """Browse for mmproj file."""
        path = QFileDialog.getOpenFileName(
            self, "Select MMProj File",
            self.gguf_dir_edit.text() or os.path.expanduser("~"),
            "GGUF files (*.gguf);;All files (*.*)"
        )[0]
        if path:
            self.mmproj_edit.setText(path)

    def browse_chat_template_file(self):
        """Browse for chat template file."""
        path = QFileDialog.getOpenFileName(
            self, "Select Chat Template File",
            os.path.expanduser("~"),
            "Jinja2 templates (*.jinja *.jinja2 *.j2);;Text files (*.txt);;All files (*.*)"
        )[0]
        if path:
            self.chat_template_file_edit.setText(path)

    def refresh_gguf_list(self):
        """Refresh the list of GGUF files in the selected directory and subdirectories."""
        gguf_dir = self.gguf_dir_edit.text().strip()
        if not gguf_dir or not os.path.isdir(gguf_dir):
            self.gguf_files = []
            self.gguf_combo.clear()
            return

        # Recursively find all .gguf files in directory and subdirectories
        gguf_files = []
        for root, dirs, files in os.walk(gguf_dir, followlinks=True):
            for f in files:
                if f.lower().endswith(".gguf"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, gguf_dir)
                    gguf_files.append(rel_path)

        self.gguf_files = sorted(gguf_files)

        # Apply current filter
        self.filter_gguf_list()

        if self.gguf_files:
            self.status_label.setText(f"Found {len(self.gguf_files)} GGUF file(s)")
        else:
            self.status_label.setText("No GGUF files found in directory")

    def filter_gguf_list(self, *args):
        """Filter the GGUF files list based on search text and show favorites first."""
        search_text = self.search_edit.text().strip().lower()
        gguf_dir = self.gguf_dir_edit.text().strip()

        if not search_text:
            filtered = self.gguf_files.copy()
        else:
            search_terms = search_text.split()
            filtered = [
                f for f in self.gguf_files
                if all(term in f.lower() for term in search_terms)
            ]

        # Sort: favorites first, then alphabetically
        def sort_key(filename):
            full_path = os.path.join(gguf_dir, filename) if gguf_dir else filename
            is_favorite = full_path in self.favorites
            return (0 if is_favorite else 1, filename.lower())

        filtered.sort(key=sort_key)

        # Add star marker to favorites in display.
        # Block signals during repopulation: clear()/addItem() would otherwise
        # fire currentTextChanged for whatever lands at the top of the list,
        # causing on_gguf_selected to load settings / start analysis for the
        # wrong model on every search keystroke.
        self.gguf_combo.blockSignals(True)
        self.gguf_combo.clear()
        for f in filtered:
            full_path = os.path.join(gguf_dir, f) if gguf_dir else f
            if full_path in self.favorites:
                self.gguf_combo.addItem(f"\u2605 {f}")
            else:
                self.gguf_combo.addItem(f)
        self.gguf_combo.blockSignals(False)

        # Update status
        if search_text and self.gguf_files:
            self.status_label.setText(f"Showing {len(filtered)} of {len(self.gguf_files)} model(s)")

    def _run_gguf_analysis(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze a GGUF model file to extract metadata.
        Runs in a worker thread to avoid blocking the UI.
        """
        result = {
            "path": model_path,
            "filename": os.path.basename(model_path),
        }

        try:
            file_size_bytes = os.path.getsize(model_path)
            result["file_size_bytes"] = file_size_bytes
            result["file_size_gb"] = round(file_size_bytes / (1024 ** 3), 2)

            if GGUF_READER_AVAILABLE:
                try:
                    result = self._analyze_with_gguf_reader(model_path, result)
                    return result
                except Exception:
                    pass

            if LLAMA_CPP_AVAILABLE:
                try:
                    result = self._analyze_with_llama_cpp(model_path, result)
                    return result
                except Exception as e:
                    result["warning"] = f"Metadata extraction failed: {str(e)}"
                    return result

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
        """Analyze GGUF using the gguf library."""
        reader = GGUFReader(model_path)

        def get_field_value(field_name: str, default=None):
            if field_name not in reader.fields:
                return default
            field = reader.fields[field_name]
            try:
                if hasattr(field, 'parts') and field.parts:
                    data = field.parts[-1]
                    if hasattr(data, 'tolist'):
                        val = data.tolist()
                        if isinstance(val, list):
                            if len(val) == 0:
                                return default
                            if len(val) == 1:
                                return val[0]
                            try:
                                int_vals = [int(x) for x in val]
                                if all(0 <= x < 256 for x in int_vals):
                                    decoded = bytes(int_vals).decode('utf-8', errors='ignore')
                                    if any(c.isalpha() for c in decoded):
                                        return decoded
                            except (ValueError, TypeError, OverflowError):
                                pass
                            return val
                        if isinstance(val, bytes):
                            return val.decode('utf-8', errors='ignore')
                        return val
                    elif isinstance(data, bytes):
                        return data.decode('utf-8', errors='ignore')
                    else:
                        return data
                elif hasattr(field, 'data'):
                    data = field.data
                    if isinstance(data, bytes):
                        return data.decode('utf-8', errors='ignore')
                    return data
            except Exception:
                pass
            return default

        arch = get_field_value("general.architecture", "unknown")
        if isinstance(arch, bytes):
            arch = arch.decode('utf-8', errors='ignore')
        if isinstance(arch, list):
            try:
                arch = bytes(int(x) for x in arch).decode('utf-8', errors='ignore')
            except (ValueError, TypeError):
                arch = str(arch)
        result["architecture"] = str(arch) if arch else "unknown"

        model_name = get_field_value("general.name", result["filename"])
        result["model_name"] = str(model_name) if model_name else result["filename"]

        ctx_length = get_field_value(f"{arch}.context_length")
        if ctx_length is None:
            ctx_length = get_field_value("general.context_length")
        if ctx_length is None:
            for key in reader.fields.keys():
                if 'context_length' in key:
                    ctx_length = get_field_value(key)
                    if ctx_length is not None:
                        break
        result["context_length"] = ctx_length if ctx_length is not None else "unknown"

        layer_count = get_field_value(f"{arch}.block_count")
        if layer_count is None:
            for key in reader.fields.keys():
                if 'block_count' in key:
                    layer_count = get_field_value(key)
                    if layer_count is not None:
                        break
        result["layer_count"] = layer_count if layer_count is not None else "unknown"

        embed_length = get_field_value(f"{arch}.embedding_length")
        if embed_length is None:
            for key in reader.fields.keys():
                if 'embedding_length' in key:
                    embed_length = get_field_value(key)
                    if embed_length is not None:
                        break
        result["embedding_length"] = embed_length if embed_length is not None else "unknown"

        head_count = get_field_value(f"{arch}.attention.head_count")
        if head_count is None:
            for key in reader.fields.keys():
                if 'head_count' in key:
                    head_count = get_field_value(key)
                    if head_count is not None:
                        break
        result["head_count"] = head_count if head_count is not None else "unknown"

        quant = self._guess_quantization(result["filename"])
        if quant == "unknown":
            file_type = get_field_value("general.file_type")
            if file_type is not None:
                quant = f"type_{file_type}"
        result["quantization"] = quant

        return result

    def _analyze_with_llama_cpp(self, model_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GGUF using llama-cpp-python (fallback method)."""
        llm = Llama(
            model_path=model_path,
            vocab_only=True,
            verbose=False,
            n_ctx=0,
            n_gpu_layers=0,
        )

        metadata = llm.metadata if hasattr(llm, 'metadata') else {}

        result["architecture"] = metadata.get("general.architecture", "unknown")
        result["model_name"] = metadata.get("general.name", result["filename"])

        ctx_key = f"{result['architecture']}.context_length"
        result["context_length"] = metadata.get(ctx_key, metadata.get("general.context_length", "unknown"))

        layer_key = f"{result['architecture']}.block_count"
        result["layer_count"] = metadata.get(layer_key, "unknown")

        embed_key = f"{result['architecture']}.embedding_length"
        result["embedding_length"] = metadata.get(embed_key, "unknown")

        head_key = f"{result['architecture']}.attention.head_count"
        result["head_count"] = metadata.get(head_key, "unknown")

        result["quantization"] = metadata.get("general.quantization_version", self._guess_quantization(result["filename"]))

        if hasattr(llm, 'n_vocab'):
            result["vocab_size"] = llm.n_vocab()

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
            info_text = f"\u26a0 {result.get('filename', 'Unknown')}: {result['error']}"
            self.model_info_label.setText(info_text)
            return

        parts = []

        if result.get("architecture") and result["architecture"] != "unknown":
            parts.append(f"Arch: {result['architecture']}")

        if result.get("layer_count") and result["layer_count"] != "unknown":
            parts.append(f"Layers: {result['layer_count']}")

        if result.get("context_length") and result["context_length"] != "unknown":
            ctx = result["context_length"]
            if isinstance(ctx, int) and ctx >= 1024:
                ctx_display = f"{ctx // 1024}K"
            else:
                ctx_display = str(ctx)
            parts.append(f"Ctx: {ctx_display}")

        if result.get("file_size_gb"):
            parts.append(f"Size: {result['file_size_gb']} GB")

        if result.get("quantization") and result["quantization"] != "unknown":
            parts.append(f"Quant: {result['quantization']}")

        if parts:
            info_text = " \u2502 ".join(parts)
        else:
            info_text = f"Size: {result.get('file_size_gb', '?')} GB"

        if result.get("warning"):
            info_text = f"Size: {result.get('file_size_gb', '?')} GB \u2502 \u26a0 {result['warning']}"

        self.model_info_label.setText(info_text)
        self.last_analysis_result = result

    def _start_analysis(self, model_path: str):
        """Start GGUF analysis in a background thread."""
        if self.current_analysis_path != model_path:
            self.current_analysis_path = model_path
            self.model_info_label.setText("Analyzing model...")

            def analysis_worker():
                result = self._run_gguf_analysis(model_path)
                if self.current_analysis_path == model_path:
                    self.analysis_signal.finished.emit(result)

            self.analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
            self.analysis_thread.start()

    def toggle_favorite(self):
        """Toggle favorite status for the selected model."""
        model_path = self.get_model_full_path()
        if not model_path:
            self.status_label.setText("No model selected")
            return

        if model_path in self.favorites:
            del self.favorites[model_path]
            self.status_label.setText(f"Removed from favorites: {os.path.basename(model_path)}")
        else:
            self.favorites[model_path] = {
                "note": "",
                "added": datetime.now().isoformat()
            }
            self.status_label.setText(f"Added to favorites: {os.path.basename(model_path)}")

        self.update_favorite_ui()
        self.save_config()

        # Refresh list to update star markers while preserving selection
        current_selection = os.path.basename(model_path)
        self.refresh_gguf_list()
        # Restore selection with appropriate star prefix
        for i in range(self.gguf_combo.count()):
            display_name = self.gguf_combo.itemText(i)
            actual_name = display_name[2:] if display_name.startswith("\u2605 ") else display_name
            if actual_name == current_selection:
                self.gguf_combo.setCurrentText(display_name)
                break

    def save_note(self, event=None):
        """Save note for the selected model."""
        model_path = self.get_model_full_path()
        if not model_path:
            return

        note = self.note_edit.text().strip()

        if model_path not in self.favorites:
            if note:
                self.favorites[model_path] = {
                    "note": note,
                    "added": datetime.now().isoformat()
                }
                self.update_favorite_ui()
                self.status_label.setText("Note saved and added to favorites")

                current_selection = os.path.basename(model_path)
                self.refresh_gguf_list()
                for i in range(self.gguf_combo.count()):
                    display_name = self.gguf_combo.itemText(i)
                    actual_name = display_name[2:] if display_name.startswith("\u2605 ") else display_name
                    if actual_name == current_selection:
                        self.gguf_combo.setCurrentText(display_name)
                        break
        else:
            self.favorites[model_path]["note"] = note
            self.status_label.setText("Note saved")

        self.save_config()

    def update_favorite_ui(self):
        """Update the favorite button and note display for the current model."""
        model_path = self.get_model_full_path()

        if model_path and model_path in self.favorites:
            self.favorite_btn.setText("\u2605")
            note = self.favorites[model_path].get("note", "")
            self.note_edit.setText(note)
        else:
            self.favorite_btn.setText("\u2606")
            self.note_edit.setText("")

    def on_gguf_selected(self, selected_text: str):
        """Handle GGUF file selection."""
        if selected_text:
            # Remove star prefix if present
            if selected_text.startswith("\u2605 "):
                selected = selected_text[2:]
            else:
                selected = selected_text

            full_path = os.path.join(self.gguf_dir_edit.text().strip(), selected)
            self.load_settings_for_model(full_path)

            # Default model alias to GGUF filename (without extension and path)
            if not self.model_alias_edit.text().strip():
                base_name = os.path.basename(selected)
                model_name = base_name[:-5] if base_name.lower().endswith(".gguf") else base_name
                self.model_alias_edit.setText(model_name)

            # Start GGUF analysis in background
            self._start_analysis(full_path)

            # Update favorite UI
            self.update_favorite_ui()

            # Update active preset label
            self.update_active_preset_label()

            # Update command preview
            self.update_command_preview()
            self.status_label.setText(f"Selected: {selected}")

    def get_model_full_path(self) -> str:
        """Get the full path of the selected model."""
        gguf_dir = self.gguf_dir_edit.text().strip()
        selected = self.gguf_combo.currentText()
        # Remove star prefix if present
        if selected.startswith("\u2605 "):
            selected = selected[2:]
        if gguf_dir and selected:
            return os.path.join(gguf_dir, selected)
        return ""

    def build_command(self) -> list[str]:
        """Build the command arguments list."""
        cmd = [self.server_path_edit.text().strip()]

        model_path = self.get_model_full_path()
        if model_path:
            cmd.extend(["-m", model_path])

        host = self.host_edit.text().strip()
        if host:
            cmd.extend(["--host", host])

        port = self.port_edit.text().strip()
        if port:
            cmd.extend(["--port", port])

        ngl = self.ngl_edit.text().strip()
        if ngl:
            cmd.extend(["-ngl", ngl])

        ncmoe = self.ncmoe_edit.text().strip()
        if ncmoe:
            cmd.extend(["-ncmoe", ncmoe])

        if self.jinja_check.isChecked():
            cmd.append("--jinja")

        threads = self.threads_edit.text().strip()
        if threads:
            cmd.extend(["--threads", threads])

        ctx_size = self.ctx_size_edit.text().strip()
        if ctx_size:
            cmd.extend(["--ctx-size", ctx_size])

        temp = self.temp_edit.text().strip()
        if temp:
            cmd.extend(["--temp", temp])

        min_p = self.min_p_edit.text().strip()
        if min_p:
            cmd.extend(["--min-p", min_p])

        top_p = self.top_p_edit.text().strip()
        if top_p:
            cmd.extend(["--top-p", top_p])

        top_k = self.top_k_edit.text().strip()
        if top_k:
            cmd.extend(["--top-k", top_k])

        presence_penalty = self.presence_penalty_edit.text().strip()
        if presence_penalty:
            cmd.extend(["--presence-penalty", presence_penalty])

        # Additional parameters
        mmproj = self.mmproj_edit.text().strip()
        if mmproj:
            cmd.extend(["--mmproj", mmproj])

        api_key = self.api_key_edit.text().strip()
        if api_key:
            cmd.extend(["--api-key", api_key])

        model_alias = self.model_alias_edit.text().strip()
        if model_alias:
            cmd.extend(["--alias", model_alias])

        chat_template = self.chat_template_edit.text().strip()
        if chat_template:
            cmd.extend(["--chat-template", chat_template])

        chat_template_file = self.chat_template_file_edit.text().strip()
        if chat_template_file:
            cmd.extend(["--chat-template-file", chat_template_file])

        batch_size = self.batch_size_edit.text().strip()
        if batch_size:
            cmd.extend(["-b", batch_size])

        ubatch_size = self.ubatch_size_edit.text().strip()
        if ubatch_size:
            cmd.extend(["-ub", ubatch_size])

        parallel = self.parallel_edit.text().strip()
        if parallel:
            cmd.extend(["--parallel", parallel])

        n_predict = self.n_predict_edit.text().strip()
        if n_predict:
            cmd.extend(["-n", n_predict])

        repeat_penalty = self.repeat_penalty_edit.text().strip()
        if repeat_penalty:
            cmd.extend(["--repeat-penalty", repeat_penalty])

        frequency_penalty = self.frequency_penalty_edit.text().strip()
        if frequency_penalty:
            cmd.extend(["--frequency-penalty", frequency_penalty])

        rope_freq_base = self.rope_freq_base_edit.text().strip()
        if rope_freq_base:
            cmd.extend(["--rope-freq-base", rope_freq_base])

        rope_freq_scale = self.rope_freq_scale_edit.text().strip()
        if rope_freq_scale:
            cmd.extend(["--rope-freq-scale", rope_freq_scale])

        cache_type_k = self.cache_k_combo.currentText().strip()
        if cache_type_k:
            cmd.extend(["--cache-type-k", cache_type_k])

        cache_type_v = self.cache_v_combo.currentText().strip()
        if cache_type_v:
            cmd.extend(["--cache-type-v", cache_type_v])

        if self.spec_type_check.isChecked():
            cmd.extend(["--spec-type", self.spec_type_combo.currentText().strip()])

        if self.spec_n_max_check.isChecked():
            cmd.extend(["--spec-draft-n-max", self.spec_n_max_edit.text().strip()])

        if self.flash_attn_check.isChecked():
            cmd.extend(["-fa", "on"])

        if self.mlock_check.isChecked():
            cmd.append("--mlock")

        if self.no_mmap_check.isChecked():
            cmd.append("--no-mmap")

        if self.cont_batching_check.isChecked():
            cmd.append("-cb")

        if self.metrics_check.isChecked():
            cmd.append("--metrics")

        if self.verbose_check.isChecked():
            cmd.append("--verbose")

        if self.log_disable_check.isChecked():
            cmd.append("--log-disable")

        if self.no_mmproj_offload_check.isChecked():
            cmd.append("--no-mmproj-offload")

        if self.no_mmproj_check.isChecked():
            cmd.append("--no-mmproj")

        if self.cpu_moe_check.isChecked():
            cmd.append("--cpu-moe")

        custom = self.custom_args_edit.text().strip()
        if custom:
            try:
                cmd.extend(shlex.split(custom))
            except ValueError:
                cmd.extend(custom.split())

        return cmd

    def build_command_string(self) -> str:
        """Build a formatted command string for display."""
        cmd = self.build_command()
        if len(cmd) < 2:
            return "# No model selected"

        def quote_if_needed(s: str) -> str:
            if not s:
                return s
            if any(c in s for c in ' \t\n\'"\\$`!*?[]{}();&|<>'):
                return shlex.quote(s)
            return s

        lines = [cmd[0]]
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
        self.command_text.setPlainText(self.build_command_string())

    def copy_command(self):
        """Copy the command to clipboard."""
        cmd = self.build_command_string()
        QApplication.clipboard().setText(cmd)
        self.status_label.setText("Command copied to clipboard")

    def launch_server(self):
        """Launch the llama-server."""
        server_path = self.server_path_edit.text().strip()
        if not server_path or not os.path.isfile(server_path):
            QMessageBox.critical(self, "Error", "Please select a valid llama-server executable")
            return

        if not os.access(server_path, os.X_OK):
            QMessageBox.critical(self, "Error", "llama-server is not executable. Run: chmod +x " + server_path)
            return

        model_path = self.get_model_full_path()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.critical(self, "Error", "Please select a valid GGUF model file")
            return

        self.save_current_settings()

        cmd = self.build_command()

        if self.run_terminal_radio.isChecked():
            try:
                terminals = [
                    ["gnome-terminal", "--", "bash", "-c"],
                    ["konsole", "-e", "bash", "-c"],
                    ["xfce4-terminal", "-x", "bash", "-c"],
                    ["xterm", "-e", "bash", "-c"],
                ]

                cmd_str = " ".join(shlex.quote(c) for c in cmd)
                full_cmd_str = f'{cmd_str}; printf "\\nPress Enter to close..."; read'

                launched = False
                for term_cmd in terminals:
                    try:
                        subprocess.Popen(term_cmd + [full_cmd_str])
                        launched = True
                        break
                    except FileNotFoundError:
                        continue

                if launched:
                    self.status_label.setText("Server launched in terminal")
                    self.show_api_info()
                else:
                    QMessageBox.critical(self, "Error", "No supported terminal emulator found")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to launch server: {e}")
        else:
            # Warn if a background server is already tracked and still alive
            if self.server_process is not None and self.server_process.poll() is None:
                QMessageBox.warning(
                    self, "Server Already Running",
                    f"A background server is already running (PID: {self.server_process.pid}).\n"
                    "Stop it with 'Kill Server' before launching another."
                )
                return
            try:
                # Redirect output to a log file. Using PIPE here would deadlock:
                # llama-server is verbose, and once the OS pipe buffer fills with
                # output that nothing reads, the server blocks on write and hangs.
                log_path = os.path.join(
                    tempfile.gettempdir(),
                    f"llama_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                )
                self.server_log_path = log_path
                self.server_log_file = open(log_path, "w")
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=self.server_log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
                self.status_label.setText(
                    f"Server launched in background (PID: {self.server_process.pid}) "
                    f"\u2502 log: {log_path}"
                )
                self.show_api_info()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to launch server: {e}")

    def show_api_info(self):
        """Show OpenAI-compatible API endpoint information."""
        host = self.host_edit.text().strip() or "localhost"
        port = self.port_edit.text().strip() or "8033"
        api_key = self.api_key_edit.text().strip()
        model_alias = self.model_alias_edit.text().strip() or "default"

        display_host = "localhost" if host == "0.0.0.0" else host
        base_url = f"http://{display_host}:{port}"

        info = f"""Server starting at: {base_url}

═══ OpenAI-Compatible API Endpoints ═══

\u2022 Chat Completions:  {base_url}/v1/chat/completions
\u2022 Text Completions:  {base_url}/v1/completions
\u2022 List Models:       {base_url}/v1/models
\u2022 Embeddings:        {base_url}/v1/embeddings
\u2022 Health Check:      {base_url}/health

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

        self.api_info_window = QWidget()
        self.api_info_window.setWindowTitle("API Endpoint Information")
        self.api_info_window.resize(550, 580)
        self.api_info_window.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowCloseButtonHint)

        frame_layout = QVBoxLayout(self.api_info_window)
        frame_layout.setContentsMargins(10, 10, 10, 10)

        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Courier", 10))
        text.setPlainText(info)
        frame_layout.addWidget(text)

        btn_layout = QHBoxLayout()

        def copy_base_url():
            QApplication.clipboard().setText(f"{base_url}/v1")
            self.status_label.setText("Base URL copied to clipboard")

        copy_url_btn = QPushButton("Copy Base URL")
        copy_url_btn.clicked.connect(copy_base_url)
        btn_layout.addWidget(copy_url_btn)
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.api_info_window.close)
        btn_layout.addWidget(close_btn)
        frame_layout.addLayout(btn_layout)

        self.api_info_window.show()

    def _cleanup_server_handles(self):
        """Close the background server's log file handle, if any."""
        if self.server_log_file is not None:
            try:
                self.server_log_file.close()
            except Exception:
                pass
            self.server_log_file = None
        self.server_process = None

    def kill_server(self):
        """Kill the llama-server process."""
        if self.server_process is not None:
            # If it already exited on its own, just clean up.
            if self.server_process.poll() is not None:
                self._cleanup_server_handles()
                self.status_label.setText("Background server had already exited")
                return
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                self.status_label.setText("Background server stopped")
                self._cleanup_server_handles()
                return
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self._cleanup_server_handles()
                self.status_label.setText("Background server killed")
                return
            except Exception as e:
                self.status_label.setText(f"Error stopping server: {e}")

        result = QMessageBox.question(
            self, "Kill llama-server",
            "No background server found. Do you want to kill ALL running llama-server processes?\n\n"
            "This will use 'pkill llama-server' to terminate any running instances.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if result == QMessageBox.StandardButton.Yes:
            try:
                subprocess.run(["pkill", "-f", "llama-server"], check=False)
                self.status_label.setText("Sent kill signal to all llama-server processes")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to kill processes: {e}")

    def get_current_settings(self) -> dict:
        """Get all current settings as a dictionary."""
        return {
            "host": self.host_edit.text().strip(),
            "port": self.port_edit.text().strip(),
            "ngl": self.ngl_edit.text().strip(),
            "ncmoe": self.ncmoe_edit.text().strip(),
            "jinja": self.jinja_check.isChecked(),
            "threads": self.threads_edit.text().strip(),
            "ctx_size": self.ctx_size_edit.text().strip(),
            "temp": self.temp_edit.text().strip(),
            "min_p": self.min_p_edit.text().strip(),
            "top_p": self.top_p_edit.text().strip(),
            "top_k": self.top_k_edit.text().strip(),
            "presence_penalty": self.presence_penalty_edit.text().strip(),
            "mmproj": self.mmproj_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
            "model_alias": self.model_alias_edit.text().strip(),
            "chat_template": self.chat_template_edit.text().strip(),
            "chat_template_file": self.chat_template_file_edit.text().strip(),
            "batch_size": self.batch_size_edit.text().strip(),
            "ubatch_size": self.ubatch_size_edit.text().strip(),
            "parallel": self.parallel_edit.text().strip(),
            "n_predict": self.n_predict_edit.text().strip(),
            "repeat_penalty": self.repeat_penalty_edit.text().strip(),
            "frequency_penalty": self.frequency_penalty_edit.text().strip(),
            "rope_freq_base": self.rope_freq_base_edit.text().strip(),
            "rope_freq_scale": self.rope_freq_scale_edit.text().strip(),
            "cache_type_k": self.cache_k_combo.currentText().strip(),
            "cache_type_v": self.cache_v_combo.currentText().strip(),
            "flash_attn": self.flash_attn_check.isChecked(),
            "mlock": self.mlock_check.isChecked(),
            "no_mmap": self.no_mmap_check.isChecked(),
            "cont_batching": self.cont_batching_check.isChecked(),
            "metrics": self.metrics_check.isChecked(),
            "verbose": self.verbose_check.isChecked(),
            "log_disable": self.log_disable_check.isChecked(),
            "no_mmproj_offload": self.no_mmproj_offload_check.isChecked(),
            "no_mmproj": self.no_mmproj_check.isChecked(),
            "cpu_moe": self.cpu_moe_check.isChecked(),
            "custom_args": self.custom_args_edit.text().strip(),
            "spec_type_enabled": self.spec_type_check.isChecked(),
            "spec_type": self.spec_type_combo.currentText().strip(),
            "spec_draft_n_max_enabled": self.spec_n_max_check.isChecked(),
            "spec_draft_n_max": self.spec_n_max_edit.text().strip(),
            "run_in_terminal": self.run_terminal_radio.isChecked(),
        }

    def apply_settings(self, settings: dict):
        """Apply settings from a dictionary."""
        self.host_edit.setText(settings.get("host", self.host_default))
        self.port_edit.setText(settings.get("port", self.port_default))
        self.ngl_edit.setText(settings.get("ngl", self.ngl_default))
        self.ncmoe_edit.setText(settings.get("ncmoe", self.ncmoe_default))
        self.jinja_check.setChecked(settings.get("jinja", self.jinja_default))
        self.threads_edit.setText(settings.get("threads", self.threads_default))
        self.ctx_size_edit.setText(settings.get("ctx_size", self.ctx_size_default))
        self.temp_edit.setText(settings.get("temp", self.temp_default))
        self.min_p_edit.setText(settings.get("min_p", self.min_p_default))
        self.top_p_edit.setText(settings.get("top_p", self.top_p_default))
        self.top_k_edit.setText(settings.get("top_k", self.top_k_default))
        self.presence_penalty_edit.setText(settings.get("presence_penalty", self.presence_penalty_default))
        self.mmproj_edit.setText(settings.get("mmproj", self.mmproj_path_default))
        self.api_key_edit.setText(settings.get("api_key", self.api_key_default))
        self.model_alias_edit.setText(settings.get("model_alias", self.model_alias_default))
        self.chat_template_edit.setText(settings.get("chat_template", self.chat_template_default))
        self.chat_template_file_edit.setText(settings.get("chat_template_file", self.chat_template_file_default))
        self.batch_size_edit.setText(settings.get("batch_size", self.batch_size_default))
        self.ubatch_size_edit.setText(settings.get("ubatch_size", self.ubatch_size_default))
        self.parallel_edit.setText(settings.get("parallel", self.parallel_default))
        self.n_predict_edit.setText(settings.get("n_predict", self.n_predict_default))
        self.repeat_penalty_edit.setText(settings.get("repeat_penalty", self.repeat_penalty_default))
        self.frequency_penalty_edit.setText(settings.get("frequency_penalty", self.frequency_penalty_default))
        self.rope_freq_base_edit.setText(settings.get("rope_freq_base", self.rope_freq_base_default))
        self.rope_freq_scale_edit.setText(settings.get("rope_freq_scale", self.rope_freq_scale_default))
        self.cache_k_combo.setCurrentText(settings.get("cache_type_k", self.cache_type_k_default))
        self.cache_v_combo.setCurrentText(settings.get("cache_type_v", self.cache_type_v_default))
        self.flash_attn_check.setChecked(settings.get("flash_attn", self.flash_attn_default))
        self.mlock_check.setChecked(settings.get("mlock", self.mlock_default))
        self.no_mmap_check.setChecked(settings.get("no_mmap", self.no_mmap_default))
        self.cont_batching_check.setChecked(settings.get("cont_batching", self.cont_batching_default))
        self.metrics_check.setChecked(settings.get("metrics", self.metrics_default))
        self.verbose_check.setChecked(settings.get("verbose", self.verbose_default))
        self.log_disable_check.setChecked(settings.get("log_disable", self.log_disable_default))
        self.no_mmproj_offload_check.setChecked(settings.get("no_mmproj_offload", self.no_mmproj_offload_default))
        self.no_mmproj_check.setChecked(settings.get("no_mmproj", self.no_mmproj_default))
        self.cpu_moe_check.setChecked(settings.get("cpu_moe", self.cpu_moe_default))
        self.custom_args_edit.setText(settings.get("custom_args", self.custom_args_default))
        self.spec_type_check.setChecked(settings.get("spec_type_enabled", self.spec_type_enabled_default))
        self.spec_type_combo.setCurrentText(settings.get("spec_type", self.spec_type_default))
        self.spec_n_max_check.setChecked(settings.get("spec_draft_n_max_enabled", self.spec_draft_n_max_enabled_default))
        self.spec_n_max_edit.setText(settings.get("spec_draft_n_max", self.spec_draft_n_max_default))
        if settings.get("run_in_terminal", self.run_in_terminal_default):
            self.run_terminal_radio.setChecked(True)
        else:
            self.run_bg_radio.setChecked(True)

    def save_current_settings(self):
        """Save current settings for the selected model."""
        model_path = self.get_model_full_path()
        if not model_path:
            self.status_label.setText("No model selected to save settings for")
            return

        settings = self.get_current_settings()

        if "model_settings" not in self.config:
            self.config["model_settings"] = {}

        self.config["model_settings"][model_path] = settings

        if model_path in self.model_preset_map:
            preset_name = self.model_preset_map[model_path]
            if preset_name in self.presets and self.presets[preset_name] != settings:
                self.active_preset_label.setText(f"Active preset: {preset_name} (modified)")

        self.save_config()
        self.status_label.setText(f"Settings saved for {os.path.basename(model_path)}")

    def load_settings_for_model(self, model_path: str):
        """Load saved settings for a specific model."""
        if "model_settings" in self.config and model_path in self.config["model_settings"]:
            self.apply_settings(self.config["model_settings"][model_path])
            self.status_label.setText(f"Loaded settings for {os.path.basename(model_path)}")
        else:
            self.apply_settings({})
            self.status_label.setText(f"Using default settings for {os.path.basename(model_path)}")

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
            self.config["llama_server_path"] = self.server_path_edit.text().strip()
            self.config["last_gguf_dir"] = self.gguf_dir_edit.text().strip()
            selected = self.gguf_combo.currentText()
            if selected.startswith("\u2605 "):
                selected = selected[2:]
            self.config["last_selected_gguf"] = selected
            self.config["favorites"] = self.favorites
            self.config["presets"] = self.presets
            self.config["model_preset_map"] = self.model_preset_map

            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def load_last_session(self):
        """Load the last session settings."""
        last_dir = self.config.get("last_gguf_dir", "")
        if last_dir and os.path.isdir(last_dir):
            self.gguf_dir_edit.setText(last_dir)
            self.refresh_gguf_list()

            last_gguf = self.config.get("last_selected_gguf", "")
            if last_gguf:
                if last_gguf in self.gguf_files:
                    for i in range(self.gguf_combo.count()):
                        display_name = self.gguf_combo.itemText(i)
                        actual_name = display_name[2:] if display_name.startswith("\u2605 ") else display_name
                        if actual_name == last_gguf:
                            self.gguf_combo.setCurrentText(display_name)
                            self.on_gguf_selected(display_name)
                            break

    def closeEvent(self, event):
        """Handle application close."""
        self.save_config()

        if self.server_process is not None and self.server_process.poll() is None:
            result = QMessageBox.question(
                self, "Server Running",
                "A background server is still running. Do you want to stop it before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if result == QMessageBox.StandardButton.Yes:
                self.kill_server()

        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Clean, modern style
    window = LlamaServerLauncher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
