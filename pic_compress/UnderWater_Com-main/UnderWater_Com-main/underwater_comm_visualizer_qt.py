from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import numpy as np
from PIL import Image

try:
    from PySide6.QtCore import QObject, Qt, QTimer, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QPalette, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    from PySide6.QtCore import QLibraryInfo

    PYSIDE6_AVAILABLE = True
    PYSIDE6_IMPORT_ERROR = None
except ImportError as exc:
    PYSIDE6_AVAILABLE = False
    PYSIDE6_IMPORT_ERROR = exc

if PYSIDE6_AVAILABLE:
    _qt_plugins_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    _qt_platform_plugins_path = str(Path(_qt_plugins_path) / "platforms")

    def _enforce_pyside_qt_plugin_env():
        os.environ["QT_PLUGIN_PATH"] = _qt_plugins_path
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _qt_platform_plugins_path

    _enforce_pyside_qt_plugin_env()
else:
    def _enforce_pyside_qt_plugin_env():
        return

import matplotlib
from scipy.io import wavfile

if PYSIDE6_AVAILABLE:
    matplotlib.use("QtAgg")
else:
    matplotlib.use("Agg")

from matplotlib import font_manager, rcParams
from matplotlib.figure import Figure

if PYSIDE6_AVAILABLE:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
else:
    FigureCanvasQTAgg = object


_CJK_FONT_CANDIDATES = [
    "Microsoft YaHei",
    "微软雅黑",
    "SimHei",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "AR PL UMing CN",
    "AR PL UKai CN",
]

UI_FONT_FAMILY = "Microsoft YaHei"


def _pick_matplotlib_sans_fonts():
    try:
        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        installed = set()
    selected = [UI_FONT_FAMILY] if UI_FONT_FAMILY in installed else []
    selected.extend(name for name in _CJK_FONT_CANDIDATES if name in installed and name not in selected)
    if "DejaVu Sans" not in selected:
        selected.append("DejaVu Sans")
    return selected


rcParams["font.sans-serif"] = _pick_matplotlib_sans_fonts()
rcParams["axes.unicode_minus"] = False

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
_probe = ROOT_DIR
while _probe != _probe.parent:
    if (_probe / "pic_compress").is_dir():
        PROJECT_ROOT = _probe
        break
    _probe = _probe.parent
if PROJECT_ROOT is None:
    PROJECT_ROOT = ROOT_DIR.parents[3]
PIC_COMPRESS_DIR = PROJECT_ROOT / "pic_compress"
STL10_CACHE_DIR = ROOT_DIR / "savedata" / "stl10_test_images"

DEFAULT_CORE_SCRIPT_PATH = str(ROOT_DIR / "JSCC_TxRx.py")
BUILTIN_METHODS = {
    "基于语义传输": str(ROOT_DIR / "SC_TxRx.py"),
    "基于JPEG传输": str(ROOT_DIR / "JPEG_TxRx.py"),
    "全bit传输": str(ROOT_DIR / "Bit_TxRx.py"),
    "JSCC传输": str(ROOT_DIR / "JSCC_TxRx.py"),
}

PALETTE = {
    "bg": "#f3efe6",
    "panel": "#fbf8f2",
    "panel_alt": "#f6f1e7",
    "panel_deep": "#16313a",
    "card": "#fffdf8",
    "border": "#d4c5a9",
    "grid": "#d8ceb7",
    "text": "#1f2a30",
    "muted": "#5c6a70",
    "accent": "#c46a2b",
    "accent_dark": "#8e4314",
    "accent_soft": "#f3d3bd",
    "teal": "#2f7a78",
    "gold": "#e0b44c",
    "success_bg": "#dff2e7",
    "success_fg": "#1f6b44",
    "active_bg": "#fff0d4",
    "active_fg": "#9a5a10",
    "idle_bg": "#ece5d9",
    "idle_fg": "#6d6458",
    "error_bg": "#f6d9d2",
    "error_fg": "#8c3322",
    "canvas": "#efe7d8",
    "plot_bg": "#f7f1e5",
}


@dataclass
class RunConfig:
    img_path: str = ""
    core_script_path: str = DEFAULT_CORE_SCRIPT_PATH
    rx_bits_path: str = str(ROOT_DIR / "rxbits_deintrlv.txt")
    tx_bits_path: str = str(ROOT_DIR / "tx_bitstream.txt")
    rx_wav_path: str = str(ROOT_DIR / "savedata" / "rx.wav")
    tx_duration: int = 25
    ams22_device_index: int | None = None
    rx_channels: int = 1
    rx_samplerate: int = 64000
    center_frequency_hz: float = 8000.0
    phy_rolloff: float = 0.5
    phy_pilot_amp: float = 0.9
    phy_sps: int = 20
    phy_max_decimation_phases: int = 8
    phy_max_candidate_multiplier: float = 2.0
    phy_max_decode_candidates: int = 96
    min_rx_rms_guard: float = 0.003
    force_offline_loopback: bool = True


if PYSIDE6_AVAILABLE:
    class EventBus(QObject):
        log = Signal(str)
        stage = Signal(str, str)
        preview_left = Signal(str)
        preview_right = Signal(str)
        signal_monitor = Signal(str, object, int)
        bit_stats = Signal(int)
        metrics = Signal(object)
        timing = Signal(object)


    def _status_colors(value: str) -> tuple[str, str]:
        text = str(value or "")
        if "失败" in text:
            return PALETTE["error_bg"], PALETTE["error_fg"]
        if any(key in text for key in ("完成", "已完成", "已选择")):
            return PALETTE["success_bg"], PALETTE["success_fg"]
        if any(key in text for key in ("进行中", "准备中")):
            return PALETTE["active_bg"], PALETTE["active_fg"]
        return PALETTE["idle_bg"], PALETTE["idle_fg"]


    class UnderwaterCommVisualizerQt(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("水下通信系统可视化界面")
            self.resize(1540, 980)
            self.setMinimumSize(1320, 840)

            self.bus = EventBus()
            self.config_data = RunConfig()
            self.core = None
            self.current_image_path = ""
            self._stl10_test_dataset = None
            self.reconstructed_image_path = str(ROOT_DIR / "rx_output.png")
            self.bitstream_cache = None
            self.tx_wave_path = str(ROOT_DIR / "savedata" / "tx.wav")
            self.running = False
            self.last_tx_info = {}
            self.current_bit_len: int | None = None
            self.current_metric_size: tuple[int, int] = (224, 224)
            self.last_rx_diag: dict[str, float | int] = {}
            self.min_rx_rms_guard: float = float(self.config_data.min_rx_rms_guard)
            self.perf_result_dir = ROOT_DIR / "performance_results"
            self.tx_start_time = None
            self.tx_end_time = None
            self.rx_start_time = None
            self.rx_end_time = None
            self.total_start_time = None
            self.total_end_time = None
            self.monitor_cfg = {
                "fft_len": 4096,
                "time_window_sec": 0.1,
                "default_samplerate": int(self.config_data.rx_samplerate),
                "spec_floor_db": -80,
                "carrier_freq": float(self.config_data.center_frequency_hz),
            }
            self.monitor_cfg["freq_xlim"] = self._monitor_freq_xlim(
                self.config_data.center_frequency_hz,
                self.monitor_cfg["default_samplerate"],
            )
            self.monitor_state = {}
            self._dirty_monitor_roles: set[str] = set()
            self._monitor_draw_scheduled = False
            self.stage_badges: dict[str, QLabel] = {}
            self.metric_vars: dict[str, QLabel] = {}
            self.preview_placeholders = {
                "left": ("等待源图像", "选择一张图后会在这里显示。"),
                "right": ("等待重建结果", "接收恢复完成后会在这里显示。"),
            }

            self._connect_bus()
            self._build_ui()
            self._init_monitor_state("tx")
            self._init_monitor_state("rx")
            self._apply_center_frequency_to_monitor(self.config_data.center_frequency_hz)
            self.monitor_redraw_timer = QTimer(self)
            self.monitor_redraw_timer.setInterval(50)
            self.monitor_redraw_timer.timeout.connect(self._flush_monitor_redraws)
            self.monitor_redraw_timer.start()
            self._safe_load_core_module()

        def _connect_bus(self):
            self.bus.log.connect(self._log, Qt.QueuedConnection)
            self.bus.stage.connect(self._set_stage, Qt.QueuedConnection)
            self.bus.preview_left.connect(self._handle_preview_left, Qt.QueuedConnection)
            self.bus.preview_right.connect(self._handle_preview_right, Qt.QueuedConnection)
            self.bus.signal_monitor.connect(self._update_signal_monitor, Qt.QueuedConnection)
            self.bus.bit_stats.connect(self._handle_bit_stats, Qt.QueuedConnection)
            self.bus.metrics.connect(self._apply_metrics, Qt.QueuedConnection)
            self.bus.timing.connect(self._apply_timing, Qt.QueuedConnection)

        def _handle_preview_left(self, path):
            self._update_preview(self.left_img_label, path, "left")

        def _handle_preview_right(self, path):
            self._update_preview(self.right_img_label, path, "right")

        def _handle_bit_stats(self, length):
            self.current_bit_len = int(length)
            self.metric_vars["bit_len"].setText(str(length))

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QVBoxLayout(central)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(10)

            root.addWidget(self._build_hero())

            body = QHBoxLayout()
            body.setSpacing(10)
            root.addLayout(body, 1)

            left = QVBoxLayout()
            left.setSpacing(10)
            right = QVBoxLayout()
            right.setSpacing(10)
            body.addLayout(left, 3)
            body.addLayout(right, 2)

            left.addWidget(self._build_control_panel())
            left.addWidget(self._build_plot_panel(), 1)

            right.addWidget(self._build_status_panel(), 2)
            right.addWidget(self._build_image_panel(), 3)
            right.addWidget(self._build_metrics_panel(), 2)

            self._apply_styles()

        def _build_hero(self):
            frame = QFrame()
            frame.setObjectName("hero")
            layout = QHBoxLayout(frame)
            layout.setContentsMargins(20, 18, 20, 16)

            left = QVBoxLayout()
            title = QLabel("水下通信实验控制台")
            title.setObjectName("heroTitle")
            left.addWidget(title)
            left.addStretch(1)

            right = QVBoxLayout()
            right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.method_chip = QLabel("当前方法: 基于语义传输")
            self.method_chip.setObjectName("methodChip")
            right.addWidget(self.method_chip, 0, Qt.AlignRight)

            layout.addLayout(left, 1)
            layout.addLayout(right)
            return frame

        def _build_control_panel(self):
            box = self._panel_box("实验配置")
            layout = QGridLayout(box)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(8)

            layout.addWidget(QLabel("源图像"), 0, 0)
            self.img_entry = QLineEdit()
            layout.addWidget(self.img_entry, 0, 1, 1, 2)
            browse_img = QPushButton("浏览")
            browse_img.clicked.connect(self.choose_image)
            layout.addWidget(browse_img, 0, 3)
            stl_img = QPushButton("STL测试图")
            stl_img.clicked.connect(self.choose_stl_test_image)
            layout.addWidget(stl_img, 0, 4)

            layout.addWidget(QLabel("方法/脚本"), 1, 0)
            self.method_combo = QComboBox()
            self.method_combo.addItems(BUILTIN_METHODS.keys())
            self.method_combo.setCurrentText("基于语义传输")
            self.method_combo.currentTextChanged.connect(self._preview_selected_method)
            layout.addWidget(self.method_combo, 1, 1)
            use_builtin = QPushButton("使用内置")
            use_builtin.clicked.connect(lambda _checked=False: self.use_selected_method())
            layout.addWidget(use_builtin, 1, 2)

            self.method_hint_label = QLabel("适合快速验证端到端主流程")
            self.method_hint_label.setObjectName("mutedLabel")
            layout.addWidget(self.method_hint_label, 1, 3)

            layout.addWidget(QLabel("当前脚本"), 2, 0)
            self.core_path_label = QLabel(self.config_data.core_script_path)
            self.core_path_label.setObjectName("mutedLabel")
            self.core_path_label.setWordWrap(True)
            layout.addWidget(self.core_path_label, 2, 1, 1, 3)

            acoustic_box = self._panel_box("水声参数设置", alt=True)
            acoustic_layout = QGridLayout(acoustic_box)
            acoustic_layout.setHorizontalSpacing(8)
            acoustic_layout.setVerticalSpacing(6)

            acoustic_layout.addWidget(QLabel("中心频率 (Hz)"), 0, 0)
            self.center_freq_entry = QLineEdit(f"{self.config_data.center_frequency_hz:.1f}")
            self.center_freq_entry.setPlaceholderText("8000")
            self.center_freq_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.center_freq_entry, 0, 1)

            acoustic_layout.addWidget(QLabel("接收采样率 (Hz)"), 0, 2)
            self.rx_samplerate_entry = QLineEdit(str(self.config_data.rx_samplerate))
            self.rx_samplerate_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.rx_samplerate_entry, 0, 3)

            acoustic_layout.addWidget(QLabel("滚降系数"), 1, 0)
            self.rolloff_entry = QLineEdit(f"{self.config_data.phy_rolloff:.2f}")
            self.rolloff_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.rolloff_entry, 1, 1)

            acoustic_layout.addWidget(QLabel("导频幅度"), 1, 2)
            self.pilot_amp_entry = QLineEdit(f"{self.config_data.phy_pilot_amp:.2f}")
            self.pilot_amp_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.pilot_amp_entry, 1, 3)

            acoustic_layout.addWidget(QLabel("每符号采样点"), 2, 0)
            self.sps_entry = QLineEdit(str(self.config_data.phy_sps))
            self.sps_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.sps_entry, 2, 1)

            acoustic_layout.addWidget(QLabel("相位搜索数"), 2, 2)
            self.decimation_phases_entry = QLineEdit(str(self.config_data.phy_max_decimation_phases))
            self.decimation_phases_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.decimation_phases_entry, 2, 3)

            acoustic_layout.addWidget(QLabel("候选倍率"), 3, 0)
            self.candidate_multiplier_entry = QLineEdit(f"{self.config_data.phy_max_candidate_multiplier:.2f}")
            self.candidate_multiplier_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.candidate_multiplier_entry, 3, 1)

            acoustic_layout.addWidget(QLabel("最大解码候选"), 3, 2)
            self.max_decode_candidates_entry = QLineEdit(str(self.config_data.phy_max_decode_candidates))
            self.max_decode_candidates_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.max_decode_candidates_entry, 3, 3)

            acoustic_layout.addWidget(QLabel("弱信号门限 (RMS)"), 4, 0)
            self.rx_rms_guard_entry = QLineEdit(f"{self.config_data.min_rx_rms_guard:.4f}")
            self.rx_rms_guard_entry.editingFinished.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.rx_rms_guard_entry, 4, 1)

            freq_hint = QLabel("建议先改中心频率与采样率，再按链路质量微调其余参数")
            freq_hint.setObjectName("mutedLabel")
            acoustic_layout.addWidget(freq_hint, 4, 2, 1, 2)

            self.offline_loopback_checkbox = QCheckBox("启用离线回环（不走实际水池）")
            self.offline_loopback_checkbox.setChecked(bool(self.config_data.force_offline_loopback))
            self.offline_loopback_checkbox.toggled.connect(self._on_center_frequency_changed)
            acoustic_layout.addWidget(self.offline_loopback_checkbox, 5, 0, 1, 4)

            layout.addWidget(acoustic_box, 3, 0, 1, 4)

            btn_row = QHBoxLayout()
            btn_specs = [
                ("1. 发射端分析", self.run_tx_analysis),
                ("2. 发射与传输", self.run_full_tx),
                ("3. 接收与重建", self.run_rx),
                ("4. 性能评估", self.calc_metrics),
                ("5. 性能分析", self.show_performance_analysis),
            ]
            for text, handler in btn_specs:
                btn = QPushButton(text)
                btn.clicked.connect(handler)
                btn_row.addWidget(btn)
            layout.addLayout(btn_row, 4, 0, 1, 4)
            return box

        def _build_plot_panel(self):
            box = self._panel_box("传输信号监测", alt=True)
            layout = QVBoxLayout(box)

            self.fig = Figure(figsize=(9.3, 7.8), dpi=100)
            self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07, hspace=0.42, wspace=0.22)
            self.fig.patch.set_facecolor(PALETTE["panel_alt"])
            self.ax_tx_wave = self.fig.add_subplot(221)
            self.ax_tx_spec = self.fig.add_subplot(222)
            self.ax_rx_wave = self.fig.add_subplot(223)
            self.ax_rx_spec = self.fig.add_subplot(224)
            self._configure_plot_axes()

            self.canvas_plot = FigureCanvasQTAgg(self.fig)
            layout.addWidget(self.canvas_plot, 1)

            self.power_label = QLabel("发射功率: - dB    接收功率: - dB")
            self.power_label.setObjectName("mutedLabelAlt")
            layout.addWidget(self.power_label)
            return box

        def _build_status_panel(self):
            box = self._panel_box("处理流程状态")
            layout = QVBoxLayout(box)

            grid = QGridLayout()
            stage_names = [
                "算法模块加载",
                "源图像加载",
                "语义表征编码",
                "比特映射与量化",
                "发射信号生成",
                "物理链路传输",
                "接收信号采集",
                "同步检测与解调",
                "图像语义重建",
                "性能评估",
            ]
            self.stage_values: dict[str, str] = {}
            for row, name in enumerate(stage_names):
                grid.addWidget(QLabel(name), row, 0)
                badge = QLabel("等待")
                badge.setAlignment(Qt.AlignCenter)
                badge.setMinimumWidth(88)
                badge.setAutoFillBackground(True)
                badge.setStyleSheet("border-radius:12px; padding:5px 10px; font-weight:700;")
                self.stage_badges[name] = badge
                self.stage_values[name] = "等待"
                self._style_badge(badge, "等待")
                grid.addWidget(badge, row, 1)
            layout.addLayout(grid)

            self.progress = QProgressBar()
            self.progress.setRange(0, len(stage_names))
            layout.addWidget(self.progress)

            log_box = self._panel_box("运行日志", alt=True)
            log_layout = QHBoxLayout(log_box)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            clear_btn = QPushButton("清空日志")
            clear_btn.clicked.connect(self.clear_log)
            clear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            log_layout.addWidget(self.log_text, 1)
            log_layout.addWidget(clear_btn)
            layout.addWidget(log_box, 1)
            return box

        def _build_image_panel(self):
            box = self._panel_box("图像重建对比")
            layout = QGridLayout(box)

            layout.addWidget(QLabel("源图像"), 0, 0, alignment=Qt.AlignCenter)
            layout.addWidget(QLabel("重建图像"), 0, 1, alignment=Qt.AlignCenter)

            self.left_img_label = self._create_image_label(*self.preview_placeholders["left"])
            self.right_img_label = self._create_image_label(*self.preview_placeholders["right"])
            layout.addWidget(self.left_img_label, 1, 0)
            layout.addWidget(self.right_img_label, 1, 1)
            return box

        def _build_metrics_panel(self):
            box = self._panel_box("性能指标", alt=True)
            layout = QGridLayout(box)
            items = [
                ("传输比特数", "bit_len"),
                ("PSNR", "psnr"),
                ("SSIM", "ssim"),
                ("压缩率", "compression"),
                ("BER", "ber"),
                ("传输时长", "transfer_time"),
                ("传输速率", "bit_rate"),
            ]
            for idx, (label, key) in enumerate(items):
                row, col = divmod(idx, 2)
                card = QFrame()
                card.setObjectName("metricCard")
                card_layout = QVBoxLayout(card)
                title = QLabel(label)
                title.setObjectName("metricTitle")
                value = QLabel("-")
                value.setObjectName("metricValue")
                card_layout.addWidget(title)
                card_layout.addWidget(value)
                layout.addWidget(card, row, col)
                self.metric_vars[key] = value
            return box

        def _panel_box(self, title: str, alt: bool = False):
            box = QGroupBox(title)
            box.setProperty("alt", alt)
            return box

        def _create_image_label(self, title: str, subtitle: str):
            label = QLabel(f"{title}\n{subtitle}")
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            label.setMinimumHeight(240)
            label.setScaledContents(False)
            label.setObjectName("imagePlaceholder")
            return label

        def _apply_styles(self):
            self.setStyleSheet(
                f"""
                QWidget {{
                    background: {PALETTE['bg']};
                    color: {PALETTE['text']};
                    font-family: "{UI_FONT_FAMILY}";
                    font-size: 13px;
                }}
                QMainWindow {{
                    background: {PALETTE['bg']};
                }}
                QFrame#hero {{
                    background: {PALETTE['panel_deep']};
                    border-radius: 16px;
                }}
                QLabel#heroTitle {{
                    color: #fff9f1;
                    font-size: 30px;
                    font-weight: 700;
                    background: transparent;
                }}
                QLabel#heroSubtitle {{
                    color: #d5e3e3;
                    font-size: 14px;
                    background: transparent;
                }}
                QLabel#heroMeta {{
                    color: #f4d6bf;
                    font-size: 12px;
                    background: transparent;
                }}
                QLabel#methodChip {{
                    color: #ebfbf7;
                    background: #204c55;
                    border-radius: 14px;
                    padding: 8px 14px;
                    font-weight: 700;
                }}
                QGroupBox {{
                    background: {PALETTE['panel']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 14px;
                    margin-top: 12px;
                    padding-top: 12px;
                    font-weight: 700;
                }}
                QGroupBox[alt="true"] {{
                    background: {PALETTE['panel_alt']};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 12px;
                    padding: 0 6px;
                    color: {PALETTE['accent_dark']};
                }}
                QLineEdit, QComboBox, QTextEdit {{
                    background: {PALETTE['card']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 10px;
                    padding: 8px;
                }}
                QPushButton {{
                    background: {PALETTE['accent']};
                    color: #fffaf4;
                    border: none;
                    border-radius: 10px;
                    padding: 10px 14px;
                    font-weight: 700;
                }}
                QPushButton:hover {{
                    background: {PALETTE['accent_dark']};
                }}
                QProgressBar {{
                    border: none;
                    background: {PALETTE['idle_bg']};
                    border-radius: 8px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background: {PALETTE['accent']};
                    border-radius: 8px;
                }}
                QLabel#mutedLabel, QLabel#mutedLabelAlt, QLabel#metricTitle {{
                    color: {PALETTE['muted']};
                }}
                QLabel#metricValue {{
                    color: {PALETTE['accent_dark']};
                    font-size: 24px;
                    font-weight: 700;
                }}
                QFrame#metricCard {{
                    background: {PALETTE['card']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 12px;
                }}
                QLabel#imagePlaceholder {{
                    background: {PALETTE['canvas']};
                    border: 1px solid {PALETTE['border']};
                    border-radius: 14px;
                    color: {PALETTE['muted']};
                    padding: 12px;
                }}
                """
            )

        def _configure_plot_axes(self):
            axes = [self.ax_tx_wave, self.ax_tx_spec, self.ax_rx_wave, self.ax_rx_spec]
            for ax in axes:
                ax.set_facecolor(PALETTE["plot_bg"])
                for spine in ax.spines.values():
                    spine.set_color(PALETTE["border"])
                ax.tick_params(colors=PALETTE["muted"])
                ax.grid(color=PALETTE["grid"], linestyle="--", linewidth=0.8, alpha=0.7)
                ax.title.set_color(PALETTE["accent_dark"])
                ax.xaxis.label.set_color(PALETTE["muted"])
                ax.yaxis.label.set_color(PALETTE["muted"])

        def _style_badge(self, label: QLabel, value: str):
            bg, fg = _status_colors(value)
            palette = label.palette()
            palette.setColor(QPalette.Window, QColor(bg))
            palette.setColor(QPalette.WindowText, QColor(fg))
            label.setPalette(palette)
            label.setText(value)

        def _method_name(self):
            if self.core is not None and hasattr(self.core, "METHOD_NAME"):
                try:
                    return str(getattr(self.core, "METHOD_NAME"))
                except Exception:
                    pass
            path = self._get_core_script_path()
            return Path(path).stem if path else "未命名方法"

        def _get_core_script_path(self):
            path = self.config_data.core_script_path or DEFAULT_CORE_SCRIPT_PATH
            return str(path).strip() or DEFAULT_CORE_SCRIPT_PATH

        def _set_core_script_path(self, path):
            resolved = str(path).strip() or DEFAULT_CORE_SCRIPT_PATH
            self.config_data.core_script_path = resolved
            if hasattr(self, "core_path_label"):
                self.core_path_label.setText(resolved)

        @staticmethod
        def _monitor_freq_xlim(center_frequency_hz, samplerate, span_hz=4000.0):
            nyq = max(1000.0, float(samplerate) / 2.0)
            fc = float(center_frequency_hz)
            fmin = max(0.0, fc - float(span_hz))
            fmax = min(nyq, fc + float(span_hz))
            if fmax - fmin < 500.0:
                return (0.0, nyq)
            return (fmin, fmax)

        def _apply_center_frequency_to_monitor(self, center_frequency_hz):
            fc = float(center_frequency_hz)
            self.monitor_cfg["carrier_freq"] = fc
            self.monitor_cfg["freq_xlim"] = self._monitor_freq_xlim(fc, self.monitor_cfg["default_samplerate"])

            for role in ("tx", "rx"):
                state = self.monitor_state.get(role)
                if state is None:
                    continue
                sr = float(state.get("samplerate", self.monitor_cfg["default_samplerate"]))
                fmin, fmax = self._monitor_freq_xlim(fc, sr)
                spec_ax = self.ax_tx_spec if role == "tx" else self.ax_rx_spec
                spec_ax.set_xlim(fmin, fmax)
                line_carrier = state.get("line_carrier")
                if line_carrier is not None:
                    line_carrier.set_xdata([fc, fc])
                self._dirty_monitor_roles.add(role)

        def _runtime_phy_params(self):
            return {
                "rolloff": float(self.config_data.phy_rolloff),
                "pilot_amp": float(self.config_data.phy_pilot_amp),
                "sps": int(self.config_data.phy_sps),
                "max_decimation_phases": int(self.config_data.phy_max_decimation_phases),
                "max_candidate_multiplier": float(self.config_data.phy_max_candidate_multiplier),
                "max_decode_candidates": int(self.config_data.phy_max_decode_candidates),
            }

        def _sync_center_frequency_from_ui(self, announce=False):
            if not hasattr(self, "center_freq_entry"):
                return True

            def _read_float(widget, label, min_value=None, max_value=None):
                text = widget.text().strip()
                if not text:
                    raise ValueError(f"{label}不能为空")
                value = float(text)
                if not np.isfinite(value):
                    raise ValueError(f"{label}必须是有限数值")
                if min_value is not None and value < min_value:
                    raise ValueError(f"{label}不能小于{min_value}")
                if max_value is not None and value > max_value:
                    raise ValueError(f"{label}不能大于{max_value}")
                return value

            def _read_int(widget, label, min_value=None, max_value=None):
                value = int(round(_read_float(widget, label, min_value=min_value, max_value=max_value)))
                if min_value is not None and value < min_value:
                    raise ValueError(f"{label}不能小于{min_value}")
                if max_value is not None and value > max_value:
                    raise ValueError(f"{label}不能大于{max_value}")
                return value

            try:
                fc = _read_float(self.center_freq_entry, "中心频率", min_value=500.0)
                rx_samplerate = _read_int(self.rx_samplerate_entry, "接收采样率", min_value=16000, max_value=192000)
                rolloff = _read_float(self.rolloff_entry, "滚降系数", min_value=0.05, max_value=1.0)
                pilot_amp = _read_float(self.pilot_amp_entry, "导频幅度", min_value=0.1, max_value=4.0)
                sps = _read_int(self.sps_entry, "每符号采样点", min_value=4, max_value=64)
                dec_phases = _read_int(self.decimation_phases_entry, "相位搜索数", min_value=1, max_value=64)
                cand_multiplier = _read_float(self.candidate_multiplier_entry, "候选倍率", min_value=1.0, max_value=8.0)
                max_decode = _read_int(self.max_decode_candidates_entry, "最大解码候选", min_value=1, max_value=512)
                rx_guard = _read_float(self.rx_rms_guard_entry, "弱信号门限", min_value=0.0, max_value=0.2)
            except Exception as e:
                QMessageBox.warning(self, "参数无效", str(e))
                return False

            dec_phases = max(1, min(dec_phases, sps))

            self.config_data.center_frequency_hz = fc
            self.config_data.rx_samplerate = int(rx_samplerate)
            self.config_data.phy_rolloff = float(rolloff)
            self.config_data.phy_pilot_amp = float(pilot_amp)
            self.config_data.phy_sps = int(sps)
            self.config_data.phy_max_decimation_phases = int(dec_phases)
            self.config_data.phy_max_candidate_multiplier = float(cand_multiplier)
            self.config_data.phy_max_decode_candidates = int(max_decode)
            self.config_data.min_rx_rms_guard = float(rx_guard)
            if hasattr(self, "offline_loopback_checkbox"):
                self.config_data.force_offline_loopback = bool(self.offline_loopback_checkbox.isChecked())
            self.min_rx_rms_guard = float(rx_guard)

            self.center_freq_entry.setText(f"{fc:.1f}")
            self.rx_samplerate_entry.setText(str(self.config_data.rx_samplerate))
            self.rolloff_entry.setText(f"{self.config_data.phy_rolloff:.2f}")
            self.pilot_amp_entry.setText(f"{self.config_data.phy_pilot_amp:.2f}")
            self.sps_entry.setText(str(self.config_data.phy_sps))
            self.decimation_phases_entry.setText(str(self.config_data.phy_max_decimation_phases))
            self.candidate_multiplier_entry.setText(f"{self.config_data.phy_max_candidate_multiplier:.2f}")
            self.max_decode_candidates_entry.setText(str(self.config_data.phy_max_decode_candidates))
            self.rx_rms_guard_entry.setText(f"{self.config_data.min_rx_rms_guard:.4f}")
            if hasattr(self, "offline_loopback_checkbox"):
                self.offline_loopback_checkbox.setChecked(bool(self.config_data.force_offline_loopback))

            self.monitor_cfg["default_samplerate"] = int(self.config_data.rx_samplerate)
            self._apply_center_frequency_to_monitor(fc)
            if announce:
                self._log(
                    "水声参数已更新: "
                    f"fc={fc:.1f}Hz, fs={self.config_data.rx_samplerate}, rolloff={self.config_data.phy_rolloff:.2f}, "
                    f"pilot_amp={self.config_data.phy_pilot_amp:.2f}, sps={self.config_data.phy_sps}, "
                    f"phases={self.config_data.phy_max_decimation_phases}, cand_mul={self.config_data.phy_max_candidate_multiplier:.2f}, "
                    f"max_decode={self.config_data.phy_max_decode_candidates}, rx_guard={self.config_data.min_rx_rms_guard:.4f}, "
                    f"offline_loopback={self.config_data.force_offline_loopback}"
                )
            self.canvas_plot.draw_idle()
            return True

        def _on_center_frequency_changed(self):
            self._sync_center_frequency_from_ui(announce=True)

        def _preview_selected_method(self, method_name):
            name = str(method_name or "").strip()
            if not name:
                return
            self._update_method_presentation(name)

        def _update_method_presentation(self, method_name):
            name = str(method_name or "未命名方法")
            hint_map = {
                "基于语义传输": "压缩最强，适合展示端到端语义恢复效果",
                "基于JPEG传输": "在压缩率和图像质量之间保持平衡",
                "全bit传输": "链路最直观，适合定位物理层与误码问题",
                "JSCC传输": "使用 JSCC Swin 编解码与 4bit tanh 量化进行端到端传输",
            }
            self.method_chip.setText(f"当前方法: {name}")
            self.method_hint_label.setText(hint_map.get(name, "自定义方法已载入，可直接运行现有流程"))

        def _safe_load_core_module(self):
            try:
                self._set_stage("算法模块加载", "进行中")
                _enforce_pyside_qt_plugin_env()
                self.config_data.core_script_path = self._get_core_script_path()
                self.core = self._load_core_module(self.config_data.core_script_path)
                _enforce_pyside_qt_plugin_env()
                self._sync_method_combo_with_path(self.config_data.core_script_path)
                self._update_method_presentation(self._method_name())
                if hasattr(self.core, "init_system") and callable(self.core.init_system):
                    self.core.init_system()
                    _enforce_pyside_qt_plugin_env()
                    self._log(f"[{self._method_name()}] 已调用传输方法 init_system()")
                self._set_stage("算法模块加载", "已完成")
                self._log(f"已加载传输方法: {self.config_data.core_script_path}")
            except Exception as e:
                self._set_stage("算法模块加载", f"失败: {e}")
                self._log(f"传输方法加载失败: {e}")
                QMessageBox.warning(self, "传输方法加载失败", "没有成功导入传输方法脚本。")

        @staticmethod
        def _load_core_module(path):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            module_name = f"uw_comm_qt_core_{int(time.time() * 1000)}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

        def _sync_method_combo_with_path(self, path):
            norm_path = str(Path(path).resolve())
            for name, candidate in BUILTIN_METHODS.items():
                try:
                    if str(Path(candidate).resolve()) == norm_path:
                        self.method_combo.blockSignals(True)
                        self.method_combo.setCurrentText(name)
                        self.method_combo.blockSignals(False)
                        self._update_method_presentation(name)
                        return
                except Exception:
                    continue
            self._update_method_presentation(Path(path).stem if path else "未命名方法")

        def choose_image(self):
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择待传输图像",
                "",
                "图像文件 (*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP);;"
                "PNG 图像 (*.png *.PNG);;"
                "JPEG 图像 (*.jpg *.JPG *.jpeg *.JPEG);;"
                "BMP 图像 (*.bmp *.BMP);;"
                "所有文件 (*.*)",
            )
            if not path:
                return
            self.current_image_path = path
            self.img_entry.setText(path)
            self._update_preview(self.left_img_label, path, "left")
            self._set_stage("源图像加载", "已选择")
            self._log(f"已选择图像: {path}")

        def _get_stl10_test_dataset(self):
            if self._stl10_test_dataset is not None:
                return self._stl10_test_dataset
            try:
                from torchvision import datasets
            except Exception as e:
                raise RuntimeError(f"无法导入 torchvision: {e}") from e
            data_root = str(PIC_COMPRESS_DIR / "data")
            try:
                self._stl10_test_dataset = datasets.STL10(root=data_root, split="test", download=False)
            except Exception as e:
                raise RuntimeError(
                    f"未找到 STL10 测试集，请先准备数据目录: {data_root}，错误: {e}"
                ) from e
            return self._stl10_test_dataset

        def choose_stl_test_image(self):
            try:
                ds = self._get_stl10_test_dataset()
                max_idx = max(0, len(ds) - 1)
                idx, ok = QInputDialog.getInt(
                    self,
                    "选择 STL10 测试图",
                    f"请输入测试集索引 (0 - {max_idx})",
                    0,
                    0,
                    max_idx,
                    1,
                )
                if not ok:
                    return
                img, _ = ds[int(idx)]
                STL10_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                save_path = STL10_CACHE_DIR / f"stl10_test_{int(idx):05d}.png"
                img.save(save_path)
                self.current_image_path = str(save_path)
                self.img_entry.setText(str(save_path))
                self._update_preview(self.left_img_label, str(save_path), "left")
                self._set_stage("源图像加载", "已选择")
                self._log(f"已加载 STL10 测试图[{idx}]: {save_path}")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载 STL10 测试图:\n{e}")

        def use_selected_method(self, reload_after=True):
            method_name = self.method_combo.currentText().strip()
            path = BUILTIN_METHODS.get(method_name)
            if not path:
                return
            self._set_core_script_path(path)
            self._update_method_presentation(method_name)
            self._log(f"已切换内置传输方法: {method_name}")
            if reload_after and not self.running:
                self._safe_load_core_module()

        def run_tx_analysis(self):
            if self._prepare_before_run(require_image=True, reset_state=True):
                threading.Thread(target=self._run_tx_analysis_worker, daemon=True).start()

        def run_full_tx(self):
            if self._prepare_before_run(require_image=True, reset_state=True):
                threading.Thread(target=self._run_full_tx_worker, daemon=True).start()

        def run_rx(self):
            if self._prepare_before_run(require_image=True, reset_state=False):
                threading.Thread(target=self._run_rx_worker, daemon=True).start()

        def calc_metrics(self):
            if self._prepare_before_run(require_image=True, reset_state=False):
                threading.Thread(target=self._calc_metrics_worker, daemon=True).start()

        def _prepare_before_run(self, require_image=True, reset_state=False):
            if self.running:
                return False
            if self.core is None:
                QMessageBox.warning(self, "提示", "传输方法未成功加载。")
                return False
            if not self._sync_center_frequency_from_ui(announce=False):
                return False
            img_path = self.img_entry.text().strip()
            if require_image and (not img_path or not os.path.exists(img_path)):
                QMessageBox.warning(self, "提示", "请先选择有效图像。")
                return False
            self.config_data.img_path = img_path
            self.config_data.core_script_path = self._get_core_script_path()
            if reset_state:
                self._reset_run_state(False)
            return True

        def _reset_run_state(self, reset_images=False):
            self.bitstream_cache = None
            self.last_tx_info = {}
            self.current_bit_len = None
            self.current_metric_size = (224, 224)
            self.last_rx_diag = {}
            self.tx_start_time = self.tx_end_time = None
            self.rx_start_time = self.rx_end_time = None
            self.total_start_time = self.total_end_time = None
            for label in self.metric_vars.values():
                label.setText("-")
            self._reset_monitor("tx")
            self._reset_monitor("rx")
            for name in [
                "源图像加载",
                "语义表征编码",
                "比特映射与量化",
                "发射信号生成",
                "物理链路传输",
                "接收信号采集",
                "同步检测与解调",
                "图像语义重建",
                "性能评估",
            ]:
                self._set_stage(name, "等待")
            if reset_images:
                self.left_img_label.setText("\n".join(self.preview_placeholders["left"]))
                self.right_img_label.setText("\n".join(self.preview_placeholders["right"]))
                self.left_img_label.setPixmap(QPixmap())
                self.right_img_label.setPixmap(QPixmap())

        def _read_effective_bits_from_file(self, path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return sum(1 for ch in f.read() if ch in "01")
            except Exception:
                return None

        @staticmethod
        def _safe_float(value):
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        def _extract_phy_diag(self):
            if self.core is None:
                return {}
            stats = getattr(self.core, "LAST_PHY_STATS", None)
            return dict(stats) if isinstance(stats, dict) else {}

        def _infer_metric_size(self):
            default_size = (224, 224)
            if self.core is None:
                return default_size
            for key in ("IMAGE_SIZE", "image_shape", "img_shape"):
                value = getattr(self.core, key, None)
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    try:
                        w = int(value[0])
                        h = int(value[1])
                    except Exception:
                        continue
                    if w > 0 and h > 0:
                        return (w, h)
            return default_size

        def _probe_wav_quality(self, wav_path):
            try:
                fs, data = wavfile.read(wav_path)
            except Exception:
                return None

            arr = np.asarray(data)
            if arr.ndim > 1:
                arr = arr[:, 0]
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float64) / max(1, np.iinfo(arr.dtype).max)
            else:
                arr = arr.astype(np.float64)

            if arr.size == 0:
                return None
            rms = float(np.sqrt(np.mean(arr ** 2) + 1e-12))
            peak = float(np.max(np.abs(arr)))
            return {
                "samplerate": int(fs),
                "duration_s": float(arr.size / max(1, int(fs))),
                "rms": rms,
                "peak": peak,
            }

        def _infer_tx_bits_fallback(self, img_path):
            if self.core is None:
                return None
            method_name = self._method_name().lower()
            if "jpeg" in method_name:
                try:
                    if hasattr(self.core, "img_to_rgb_pil") and hasattr(self.core, "pil_to_bytes_jpeg"):
                        quality = getattr(self.core, "JPEG_QUALITY", 90)
                        image_size = getattr(self.core, "IMAGE_SIZE", None)
                        img = self.core.img_to_rgb_pil(img_path, size=image_size)
                        comp_bytes = self.core.pil_to_bytes_jpeg(img, quality=quality)
                        return int(len(comp_bytes) * 8)
                except Exception:
                    pass
            if "全bit" in method_name or "bit传输" in method_name or "rawbit" in method_name:
                try:
                    import cv2

                    shape = getattr(self.core, "image_shape", None) or (96, 96)
                    raw = cv2.imread(img_path)
                    if raw is None:
                        return None
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    raw = cv2.resize(raw, tuple(shape))
                    return int(raw.size * 8)
                except Exception:
                    pass
            if "jscc" in method_name:
                try:
                    if hasattr(self.core, "estimate_tx_bits") and callable(self.core.estimate_tx_bits):
                        return int(self.core.estimate_tx_bits(img_path))
                except Exception:
                    pass
            try:
                if hasattr(self.core, "SC_TX") and getattr(self.core, "SC_TX") is not None and hasattr(self.core, "device"):
                    import torch
                    import cv2

                    raw = cv2.imread(img_path)
                    if raw is None:
                        return None
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(rgb, (224, 224))
                    image = resized.astype(np.float32) / 127.5 - 1.0
                    image = image.transpose(2, 0, 1)[None, :, :, :]
                    tensor = torch.from_numpy(image).to(self.core.device)
                    se = self.core.SC_TX(tensor)
                    return int((se > 0.5).int().reshape(-1).numel())
            except Exception:
                pass
            return None

        def _ensure_perf_result_dir(self):
            self.perf_result_dir.mkdir(parents=True, exist_ok=True)

        @staticmethod
        def _safe_method_filename(method_name):
            name = str(method_name).strip() or "未命名方法"
            name = re.sub(r'[\\/:*?"<>|]+', "_", name)
            return name[:80] or "未命名方法"

        @staticmethod
        def _metric_sort_key(metric_name):
            order = {
                "psnr": 0,
                "ssim": 1,
                "compression": 2,
                "ber": 3,
                "transfer_time": 4,
                "bit_rate": 5,
                "bit_len": 6,
            }
            return order.get(metric_name, 999)

        @staticmethod
        def _analysis_metric_excludes():
            return {
                "sync_peak",
                "rx_rms",
                "rx_peak",
                "rx_samplerate",
                "center_frequency_hz",
            }

        def _save_performance_result(self, metrics):
            try:
                method_name = self._method_name()
                payload = {
                    "method_name": method_name,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_image": self.config_data.img_path,
                    "core_script_path": self._get_core_script_path(),
                    "acoustic_params": {
                        "center_frequency_hz": float(self.config_data.center_frequency_hz),
                        "rx_samplerate": int(self.config_data.rx_samplerate),
                        "rolloff": float(self.config_data.phy_rolloff),
                        "pilot_amp": float(self.config_data.phy_pilot_amp),
                        "sps": int(self.config_data.phy_sps),
                        "max_decimation_phases": int(self.config_data.phy_max_decimation_phases),
                        "max_candidate_multiplier": float(self.config_data.phy_max_candidate_multiplier),
                        "max_decode_candidates": int(self.config_data.phy_max_decode_candidates),
                        "min_rx_rms_guard": float(self.config_data.min_rx_rms_guard),
                    },
                    "metrics": metrics,
                }
                self._ensure_perf_result_dir()
                save_path = self.perf_result_dir / f"{self._safe_method_filename(method_name)}.json"
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                self.bus.log.emit(f"[{method_name}] 性能结果已保存到本地: {save_path}")
            except Exception as e:
                self.bus.log.emit(f"[{self._method_name()}] 性能结果保存失败: {e}")

        def _load_saved_performance_results(self):
            self._ensure_perf_result_dir()
            results = []
            for file in sorted(self.perf_result_dir.glob("*.json")):
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    method_name = data.get("method_name") or file.stem
                    metrics = data.get("metrics") or {}
                    if not isinstance(metrics, dict):
                        continue
                    normalized = {}
                    for k, v in metrics.items():
                        try:
                            normalized[k] = None if v is None else float(v)
                        except Exception:
                            continue
                    if normalized:
                        results.append({"method_name": method_name, "metrics": normalized})
                except Exception as e:
                    self._log(f"读取性能结果失败 {file.name}: {e}")
            return results

        def show_performance_analysis(self):
            results = self._load_saved_performance_results()
            if not results:
                QMessageBox.information(self, "提示", "本地还没有已保存的性能评估结果，请先执行“性能评估”。")
                return
            excludes = self._analysis_metric_excludes()
            metric_names = sorted(
                {
                    k
                    for item in results
                    for k, v in item["metrics"].items()
                    if v is not None and k not in excludes
                },
                key=self._metric_sort_key,
            )
            if not metric_names:
                QMessageBox.information(self, "提示", "未读取到可用于绘图的指标数据。")
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("性能分析")
            dialog.resize(1280, 940)
            layout = QVBoxLayout(dialog)

            header = QLabel("性能结果对比")
            header.setStyleSheet(f"font-size:24px; font-weight:700; color:{PALETTE['accent_dark']};")
            layout.addWidget(header)

            method_summary = QLabel("、".join(item["method_name"] for item in results))
            method_summary.setStyleSheet(f"color:{PALETTE['muted']};")
            layout.addWidget(method_summary)

            fig = self._build_performance_figure(results, metric_names)
            canvas = FigureCanvasQTAgg(fig)
            layout.addWidget(canvas, 1)

            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn, 0, Qt.AlignRight)
            dialog.exec()

        def _build_performance_figure(self, results, metric_names):
            cols = 2
            rows = math.ceil(len(metric_names) / cols)
            fig = Figure(figsize=(13.4, max(6.8, rows * 3.75)), dpi=100)
            fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.07, hspace=0.92, wspace=0.30)
            fig.patch.set_facecolor(PALETTE["panel"])
            method_names = [item["method_name"] for item in results]
            colors = [PALETTE["accent"], PALETTE["teal"], PALETTE["gold"], "#7a8b5b", "#ab5b4d", "#5f6fb3"]

            title_map = {
                "psnr": "PSNR 对比 (dB)",
                "ssim": "SSIM 对比",
                "compression": "压缩率对比 (%)",
                "ber": "BER 对比",
                "transfer_time": "传输时长对比 (s)",
                "bit_rate": "传输速率对比 (kbps)",
                "bit_len": "传输比特数对比",
            }
            y_label_map = {
                "psnr": "PSNR (dB)",
                "ssim": "SSIM",
                "compression": "压缩率 (%)",
                "ber": "BER",
                "transfer_time": "时间 (s)",
                "bit_rate": "速率 (kbps)",
                "bit_len": "比特数",
            }

            for idx, metric_name in enumerate(metric_names, start=1):
                ax = fig.add_subplot(rows, cols, idx)
                values = []
                display_values = []
                for item in results:
                    value = item["metrics"].get(metric_name)
                    if metric_name == "compression" and value is not None:
                        display = value * 100.0
                    elif metric_name == "bit_rate" and value is not None:
                        display = value / 1000.0
                    else:
                        display = value
                    display_values.append(display)
                    values.append(0.0 if display is None else display)

                bars = ax.bar(method_names, values, color=colors[: len(method_names)], width=0.62, edgecolor="#555555", linewidth=0.6)
                ax.set_title(title_map.get(metric_name, metric_name), fontsize=11, pad=10)
                ax.set_ylabel(y_label_map.get(metric_name, metric_name))
                ax.set_facecolor(PALETTE["plot_bg"])
                for spine in ax.spines.values():
                    spine.set_color(PALETTE["border"])
                ax.grid(axis="y", linestyle="--", alpha=0.28, color=PALETTE["grid"])
                ax.set_axisbelow(True)
                ax.tick_params(axis="x", rotation=24, labelsize=9, pad=6, colors=PALETTE["muted"])
                ax.tick_params(axis="y", colors=PALETTE["muted"])
                ax.title.set_color(PALETTE["accent_dark"])
                ax.yaxis.label.set_color(PALETTE["muted"])
                max_val = max(values) if values else 0
                ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)

                for bar, raw_value in zip(bars, display_values):
                    if raw_value is None:
                        label = "-"
                    elif metric_name == "bit_len":
                        label = f"{int(round(raw_value))}"
                    elif metric_name == "ssim":
                        label = f"{raw_value:.4f}"
                    elif metric_name == "ber":
                        label = f"{raw_value:.6f}"
                    else:
                        label = f"{raw_value:.2f}"
                    y = bar.get_height() + (0.02 if max_val == 0 else max_val * 0.02)
                    ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va="bottom", fontsize=8.5, color=PALETTE["text"])

            total_axes = rows * cols
            for idx in range(len(metric_names) + 1, total_axes + 1):
                ax = fig.add_subplot(rows, cols, idx)
                ax.axis("off")
            return fig

        def _record_tx_result(self, tx_result):
            if not isinstance(tx_result, dict):
                tx_result = {}
            self.last_tx_info = dict(tx_result)
            tx_bits = tx_result.get("tx_bits")
            if tx_bits is None:
                tx_bits_path = tx_result.get("tx_bitstream_path") or self.config_data.tx_bits_path
                tx_bits = self._read_effective_bits_from_file(tx_bits_path)
            if tx_bits is not None:
                self.current_bit_len = int(tx_bits)
                self.bus.bit_stats.emit(int(tx_bits))
            if tx_result.get("tx_wav_path"):
                self.tx_wave_path = tx_result["tx_wav_path"]
            if tx_result.get("rx_wav_path"):
                self.config_data.rx_wav_path = tx_result["rx_wav_path"]

        def _emit_signal_monitor(self, role, data, samplerate):
            arr = np.asarray(data, dtype=np.float32)
            self.bus.signal_monitor.emit(str(role), arr.copy(), int(samplerate))

        def _run_tx_analysis_worker(self):
            self.running = True
            self.total_start_time = self.tx_start_time = time.perf_counter()
            try:
                self.bus.stage.emit("源图像加载", "完成")
                self.bus.preview_left.emit(self.config_data.img_path)
                self.bus.stage.emit("语义表征编码", "进行中")
                self.bus.stage.emit("比特映射与量化", "进行中")
                bit_len = self._infer_tx_bits_fallback(self.config_data.img_path)
                if bit_len is None:
                    raise RuntimeError("无法推断当前传输方法的发送比特数，请执行完整传输以获取精确值。")
                self.bus.bit_stats.emit(int(bit_len))
                self.bus.stage.emit("语义表征编码", "完成")
                self.bus.stage.emit("比特映射与量化", "完成")
                self.bus.stage.emit("发射信号生成", "仅完整传输时执行")
                self.tx_end_time = self.total_end_time = time.perf_counter()
                self.bus.timing.emit(self.total_end_time - self.total_start_time)
                self.bus.log.emit(f"[{self._method_name()}] 发射编码分析完成，推断发送比特数={bit_len}")
            except Exception as e:
                self.bus.log.emit(f"[{self._method_name()}] 发送端分析失败: {e}")
            finally:
                self.running = False

        def _run_full_tx_worker(self):
            self.running = True
            self.total_start_time = self.tx_start_time = time.perf_counter()
            try:
                self.bus.preview_left.emit(self.config_data.img_path)
                self.bus.stage.emit("源图像加载", "完成")
                self.bus.stage.emit("物理链路传输", "进行中")
                self.bus.stage.emit("接收信号采集", "准备中")
                self.bus.log.emit(f"[{self._method_name()}] 开始执行完整发送流程...")

                tx_kwargs = {
                    "rx_wav_path": self.config_data.rx_wav_path,
                    "rx_channels": self.config_data.rx_channels,
                    "rx_samplerate": self.config_data.rx_samplerate,
                }
                if self.config_data.ams22_device_index is not None:
                    tx_kwargs["ams22_device_index"] = self.config_data.ams22_device_index
                try:
                    tx_sig = inspect.signature(self.core.Tx)
                    tx_accepts_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in tx_sig.parameters.values()
                    )
                    force_offline_loopback = bool(self.config_data.force_offline_loopback)
                    if "force_offline_loopback" in tx_sig.parameters or tx_accepts_varkw:
                        tx_kwargs["force_offline_loopback"] = force_offline_loopback
                    if "center_frequency_hz" in tx_sig.parameters or tx_accepts_varkw:
                        tx_kwargs["center_frequency_hz"] = self.config_data.center_frequency_hz
                    if "phy_params" in tx_sig.parameters or tx_accepts_varkw:
                        tx_kwargs["phy_params"] = self._runtime_phy_params()
                    if "monitor_callback" in tx_sig.parameters:
                        tx_kwargs["monitor_callback"] = self._emit_signal_monitor
                    if "log_callback" in tx_sig.parameters:
                        tx_kwargs["log_callback"] = self.bus.log.emit
                except Exception:
                    pass

                tx_result = self.core.Tx(self.config_data.img_path, **tx_kwargs)
                self._record_tx_result(tx_result)
                if self.current_bit_len is None:
                    tx_bits_path = tx_result.get("tx_bitstream_path") or self.config_data.tx_bits_path
                    bit_len = self._read_effective_bits_from_file(tx_bits_path)
                    if bit_len is None:
                        bit_len = self._infer_tx_bits_fallback(self.config_data.img_path)
                    if bit_len is not None:
                        self.current_bit_len = int(bit_len)
                        self.bus.bit_stats.emit(int(bit_len))

                self.tx_end_time = self.total_end_time = time.perf_counter()
                self.bus.stage.emit("物理链路传输", "完成")
                self.bus.stage.emit("接收信号采集", "完成")
                self.bus.timing.emit(self.total_end_time - self.total_start_time)
                self.bus.log.emit(f"[{self._method_name()}] 完整发送流程完成")
            except Exception as e:
                self.bus.log.emit(f"[{self._method_name()}] 完整发送失败: {e}")
            finally:
                self.running = False

        def _run_rx_worker(self):
            self.running = True
            if self.total_start_time is None:
                self.total_start_time = time.perf_counter()
            self.rx_start_time = time.perf_counter()
            try:
                self.bus.stage.emit("同步检测与解调", "进行中")
                if not os.path.exists(self.config_data.rx_wav_path):
                    raise RuntimeError(f"接收音频文件不存在: {self.config_data.rx_wav_path}")

                wav_diag = self._probe_wav_quality(self.config_data.rx_wav_path)
                if wav_diag is not None:
                    self.last_rx_diag.update(wav_diag)
                    self.bus.log.emit(
                        f"[{self._method_name()}] 接收波形体检: fs={wav_diag['samplerate']} Hz, "
                        f"duration={wav_diag['duration_s']:.2f}s, rms={wav_diag['rms']:.6f}, peak={wav_diag['peak']:.6f}"
                    )
                    tx_mode = str(self.last_tx_info.get("transmission_mode", ""))
                    if tx_mode != "offline_loopback" and wav_diag["rms"] < self.min_rx_rms_guard:
                        raise RuntimeError(
                            "接收信号过弱，已中止解调。请检查输入设备是否为 AMS-22、增益是否足够、声学链路是否畅通。"
                        )

                rx_kwargs = {}
                try:
                    rx_sig = inspect.signature(self.core.Rx)
                    rx_accepts_varkw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in rx_sig.parameters.values()
                    )
                    if "center_frequency_hz" in rx_sig.parameters or rx_accepts_varkw:
                        rx_kwargs["center_frequency_hz"] = self.config_data.center_frequency_hz
                    if "phy_params" in rx_sig.parameters or rx_accepts_varkw:
                        rx_kwargs["phy_params"] = self._runtime_phy_params()
                    if "log_callback" in rx_sig.parameters:
                        rx_kwargs["log_callback"] = self.bus.log.emit
                    if "save_img_path" in rx_sig.parameters:
                        rx_kwargs["save_img_path"] = self.reconstructed_image_path
                    if "rx_wav_path" in rx_sig.parameters:
                        rx_kwargs["rx_wav_path"] = self.config_data.rx_wav_path
                except Exception:
                    pass
                result = self.core.Rx(self.config_data.rx_bits_path, **rx_kwargs)
                phy_diag = self._extract_phy_diag()
                if phy_diag:
                    self.last_rx_diag.update(phy_diag)
                    self.bus.log.emit(
                        f"[{self._method_name()}] 解调诊断: sync_peak={phy_diag.get('sync_peak', '-')}, "
                        f"ber={phy_diag.get('ber', '-')}, rx_rms={phy_diag.get('rx_passband_rms', '-')}"
                    )
                if isinstance(result, str) and result:
                    self.reconstructed_image_path = result
                self.bus.stage.emit("同步检测与解调", "完成")
                self.bus.stage.emit("图像语义重建", "完成")
                if os.path.exists(self.reconstructed_image_path):
                    self.bus.preview_right.emit(self.reconstructed_image_path)
                self.rx_end_time = self.total_end_time = time.perf_counter()
                self.bus.timing.emit(self.total_end_time - self.total_start_time)
                self.bus.log.emit(f"[{self._method_name()}] 接收恢复完成")
            except Exception as e:
                self.bus.log.emit(f"[{self._method_name()}] 接收恢复失败: {e}")
            finally:
                self.running = False

        def _calc_metrics_worker(self):
            self.running = True
            try:
                img_path = self.config_data.img_path
                rec_path = self.reconstructed_image_path
                if not os.path.exists(img_path) or not os.path.exists(rec_path):
                    raise RuntimeError("源图像或重建图像不存在")
                self.bus.stage.emit("性能评估", "进行中")
                metric_size = self._infer_metric_size()
                self.current_metric_size = metric_size
                metric_kwargs = {}
                try:
                    metric_sig = inspect.signature(self.core.calc_metrics_and_show)
                    if "log_callback" in metric_sig.parameters:
                        metric_kwargs["log_callback"] = self.bus.log.emit
                    if "size" in metric_sig.parameters:
                        metric_kwargs["size"] = metric_size
                    if "bitstream_path" in metric_sig.parameters:
                        metric_kwargs["bitstream_path"] = self.last_tx_info.get("tx_bitstream_path", self.config_data.tx_bits_path)
                except Exception:
                    pass
                psnr, ssim, cr = self.core.calc_metrics_and_show(img_path, rec_path, **metric_kwargs)
                ber = self._compute_ber()
                phy_diag = self._extract_phy_diag()
                if phy_diag:
                    self.last_rx_diag.update(phy_diag)
                self.bus.metrics.emit({"psnr": psnr, "ssim": ssim, "compression": cr, "ber": ber})
                self.bus.stage.emit("性能评估", "完成")
                bit_len = self._current_bit_length()
                transfer_time = (self.total_end_time - self.total_start_time) if (self.total_end_time and self.total_start_time) else None
                recon_bits = int(np.prod(metric_size) * 3 * 8)
                bitrate = (recon_bits / transfer_time) if transfer_time not in (None, 0) else None
                metrics_snapshot = {
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                    "compression": float(cr),
                    "ber": None if ber is None else float(ber),
                    "transfer_time": None if transfer_time is None else float(transfer_time),
                    "bit_rate": None if bitrate is None else float(bitrate),
                    "bit_len": None if bit_len is None else int(bit_len),
                }
                self._save_performance_result(metrics_snapshot)
                self.bus.log.emit(
                    f"[{self._method_name()}] 性能评估完成: PSNR={psnr:.2f}, SSIM={ssim:.4f}, "
                    f"BER={'-' if ber is None else f'{ber:.6f}'}, 压缩率={cr * 100:.2f}%, "
                    f"传输速率={'-' if bitrate is None else f'{bitrate / 1000:.2f} kbps'}"
                )
            except Exception as e:
                self.bus.log.emit(f"[{self._method_name()}] 性能评估失败: {e}")
            finally:
                self.running = False

        def _apply_metrics(self, payload):
            self.metric_vars["psnr"].setText(f"{payload['psnr']:.2f} dB")
            self.metric_vars["ssim"].setText(f"{payload['ssim']:.4f}")
            self.metric_vars["compression"].setText(f"{payload['compression'] * 100:.2f}%")
            ber = payload.get("ber")
            self.metric_vars["ber"].setText("-" if ber is None else f"{ber:.6f}")
            transfer_time = None
            if self.total_start_time is not None and self.total_end_time is not None:
                transfer_time = self.total_end_time - self.total_start_time
            self.metric_vars["transfer_time"].setText("-" if transfer_time is None else f"{transfer_time:.3f} s")
            self._refresh_rate_metrics()

        def _apply_timing(self, transfer_time):
            self.metric_vars["transfer_time"].setText("-" if transfer_time is None else f"{transfer_time:.3f} s")
            self._refresh_rate_metrics()

        def _init_monitor_state(self, role):
            sr = self.monitor_cfg["default_samplerate"]
            time_len = int(sr * self.monitor_cfg["time_window_sec"])
            fft_len = self.monitor_cfg["fft_len"]
            state = {
                "samplerate": sr,
                "time_buffer": np.zeros(time_len, dtype=np.float32),
                "latest_chunk": np.zeros(min(2048, fft_len), dtype=np.float32),
                "power_db": None,
                "spectrum_freqs": None,
                "spectrum_mag_db": None,
                "line_carrier": None,
            }
            self.monitor_state[role] = state
            wave_ax = self.ax_tx_wave if role == "tx" else self.ax_rx_wave
            spec_ax = self.ax_tx_spec if role == "tx" else self.ax_rx_spec
            wave_title = "发射端时域波形" if role == "tx" else "接收端时域波形"
            spec_title = "发射端频谱特性" if role == "tx" else "接收端频谱特性"
            x_time = np.arange(time_len) / sr
            freqs = np.fft.rfftfreq(fft_len, d=1 / sr)
            wave_color = PALETTE["accent"] if role == "tx" else PALETTE["teal"]
            spec_color = PALETTE["accent_dark"] if role == "tx" else PALETTE["teal"]
            state["line_wave"], = wave_ax.plot(x_time, state["time_buffer"], color=wave_color, linewidth=1.6)
            state["line_spec"], = spec_ax.plot(freqs, np.full_like(freqs, -120.0), color=spec_color, linewidth=1.6)
            wave_ax.set_title(wave_title)
            wave_ax.set_xlabel("Time (s)")
            wave_ax.set_ylabel("Amplitude")
            spec_ax.set_title(spec_title)
            spec_ax.set_xlabel("Frequency (Hz)")
            spec_ax.set_ylabel("幅度 (dB, 相对峰值)")
            fmin, fmax = self._monitor_freq_xlim(self.monitor_cfg.get("carrier_freq", 8000.0), sr)
            spec_ax.set_xlim(fmin, fmax)
            spec_ax.set_ylim(self.monitor_cfg.get("spec_floor_db", -80), 3)
            state["line_carrier"] = spec_ax.axvline(
                self.monitor_cfg.get("carrier_freq", 8000.0),
                color=PALETTE["gold"],
                linestyle="--",
                linewidth=1,
            )
            self._dirty_monitor_roles.add(role)

        def _reset_monitor(self, role):
            if role not in self.monitor_state:
                return
            sr = self.monitor_state[role]["samplerate"]
            time_len = int(sr * self.monitor_cfg["time_window_sec"])
            self.monitor_state[role]["time_buffer"] = np.zeros(time_len, dtype=np.float32)
            self.monitor_state[role]["latest_chunk"] = np.zeros(min(2048, self.monitor_cfg["fft_len"]), dtype=np.float32)
            self.monitor_state[role]["power_db"] = None
            self.monitor_state[role]["spectrum_freqs"] = None
            self.monitor_state[role]["spectrum_mag_db"] = None
            self._dirty_monitor_roles.add(role)

        def _update_signal_monitor(self, role, data, samplerate):
            base_role = "tx" if str(role).startswith("tx") else "rx"
            state = self.monitor_state.get(base_role)
            if state is None:
                return
            arr = np.asarray(data, dtype=np.float32)
            state["samplerate"] = samplerate
            if str(role).endswith("_spectrum") and arr.ndim == 2 and arr.shape[1] >= 2:
                state["spectrum_freqs"] = arr[:, 0].copy()
                state["spectrum_mag_db"] = arr[:, 1].copy()
                self._dirty_monitor_roles.add(base_role)
                return
            if arr.ndim > 1:
                arr = arr[:, 0]
            arr = arr.reshape(-1)
            if arr.size == 0:
                return
            desired_len = int(max(1, samplerate * self.monitor_cfg["time_window_sec"]))
            if len(state["time_buffer"]) != desired_len:
                state["time_buffer"] = np.zeros(desired_len, dtype=np.float32)
            shift = len(arr)
            if shift >= desired_len:
                state["time_buffer"] = arr[-desired_len:]
            else:
                state["time_buffer"][:-shift] = state["time_buffer"][shift:]
                state["time_buffer"][-shift:] = arr
            state["latest_chunk"] = arr[-min(len(arr), self.monitor_cfg["fft_len"]):]
            state["power_db"] = 10 * np.log10(np.mean(np.square(arr, dtype=np.float32)) + 1e-12)
            self._dirty_monitor_roles.add(base_role)

        def _flush_monitor_redraws(self):
            if not self._dirty_monitor_roles:
                return
            dirty_roles = tuple(sorted(self._dirty_monitor_roles))
            self._dirty_monitor_roles.clear()
            for role in dirty_roles:
                self._redraw_monitor(role)
            if not self._monitor_draw_scheduled:
                self._monitor_draw_scheduled = True
                self.canvas_plot.draw_idle()
                QTimer.singleShot(0, self._clear_monitor_draw_scheduled)

        def _clear_monitor_draw_scheduled(self):
            self._monitor_draw_scheduled = False

        def _redraw_monitor(self, role):
            state = self.monitor_state[role]
            sr = state["samplerate"]
            fft_len = self.monitor_cfg["fft_len"]
            x_time = np.arange(len(state["time_buffer"])) / sr
            wave_data = state["time_buffer"]
            peak = np.percentile(np.abs(wave_data), 99.5) if wave_data.size else 1.0
            peak = max(float(peak), 1e-6)
            wave_plot = np.clip(wave_data / peak, -1.0, 1.0)
            state["line_wave"].set_data(x_time, wave_plot)
            state["line_wave"].axes.set_xlim(0, x_time[-1] if len(x_time) else self.monitor_cfg["time_window_sec"])
            state["line_wave"].axes.set_ylim(-1, 1)
            if state["spectrum_freqs"] is not None and state["spectrum_mag_db"] is not None:
                freqs = state["spectrum_freqs"]
                mag_db = state["spectrum_mag_db"]
            else:
                freqs = np.fft.rfftfreq(fft_len, d=1 / sr)
                chunk = state["latest_chunk"]
                if len(chunk) < fft_len:
                    padded = np.zeros(fft_len, dtype=np.float32)
                    padded[: len(chunk)] = chunk
                    chunk = padded
                else:
                    chunk = chunk[:fft_len]
                mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
                ref = max(float(np.max(mag)), 1e-8)
                mag_db = 20 * np.log10(mag / ref + 1e-8)
            state["line_spec"].set_data(freqs, mag_db)
            fmin, fmax = self._monitor_freq_xlim(self.monitor_cfg.get("carrier_freq", 8000.0), sr)
            state["line_spec"].axes.set_xlim(fmin, fmax)
            state["line_spec"].axes.set_ylim(self.monitor_cfg.get("spec_floor_db", -80), 3)
            line_carrier = state.get("line_carrier")
            if line_carrier is not None:
                carrier = float(self.monitor_cfg.get("carrier_freq", 8000.0))
                line_carrier.set_xdata([carrier, carrier])
            tx_p = self.monitor_state.get("tx", {}).get("power_db")
            rx_p = self.monitor_state.get("rx", {}).get("power_db")
            self.power_label.setText(
                f"发射功率: {'-' if tx_p is None else f'{tx_p:.1f}'} dB    "
                f"接收功率: {'-' if rx_p is None else f'{rx_p:.1f}'} dB"
            )

        def _update_preview(self, label: QLabel, image_path: str, side: str):
            try:
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    raise ValueError("无法加载图像")
                scaled = pixmap.scaled(
                    label.size() if label.width() > 10 else label.minimumSizeHint(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                label.setPixmap(scaled)
                label.setText("")
                label.setProperty("source_path", image_path)
            except Exception as e:
                label.setPixmap(QPixmap())
                label.setText(f"图像预览失败\n{e}")

        def _compute_ber(self):
            tx_path = self.last_tx_info.get("tx_bitstream_path", self.config_data.tx_bits_path)
            rx_path = self.config_data.rx_bits_path
            if not tx_path or not rx_path or not os.path.exists(tx_path) or not os.path.exists(rx_path):
                return None
            try:
                with open(tx_path, "r", encoding="utf-8") as f:
                    tx_bits = "".join(ch for ch in f.read() if ch in "01")
                with open(rx_path, "r", encoding="utf-8") as f:
                    rx_bits = "".join(ch for ch in f.read() if ch in "01")
                n = min(len(tx_bits), len(rx_bits))
                if n <= 0:
                    return None
                return sum(1 for a, b in zip(tx_bits[:n], rx_bits[:n]) if a != b) / n
            except Exception as e:
                self._log(f"[{self._method_name()}] BER 计算失败: {e}")
                return None

        def _current_bit_length(self):
            return self.current_bit_len

        def _refresh_rate_metrics(self):
            try:
                transfer_time = float(self.metric_vars["transfer_time"].text().split()[0])
                recon_bits = int(np.prod(self.current_metric_size) * 3 * 8)
                self.metric_vars["bit_rate"].setText(f"{recon_bits / transfer_time / 1000:.2f} kbps" if transfer_time > 0 else "-")
            except Exception:
                self.metric_vars["bit_rate"].setText("-")

        def _set_stage(self, name, value):
            if name in self.stage_badges:
                if self.stage_values.get(name) == value:
                    return
                self.stage_values[name] = value
                self._style_badge(self.stage_badges[name], value)
            done = sum(1 for v in self.stage_values.values() if ("完成" in v or "已完成" in v or "已选择" in v))
            self.progress.setValue(done)

        def _log(self, text):
            ts = time.strftime("%H:%M:%S")
            self.log_text.append(f"[{ts}] {text}")

        def clear_log(self):
            self.log_text.clear()


else:
    class UnderwaterCommVisualizerQt:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PySide6 未安装，无法启动 Qt 版界面。"
            )


def _missing_qt_message() -> str:
    return (
        "当前环境未安装 PySide6，Qt 版界面暂时无法启动。\n\n"
        "建议在项目使用的环境里安装后再运行：\n"
        "  conda run -n SemCom python -m pip install PySide6\n\n"
        f"原始错误：{PYSIDE6_IMPORT_ERROR}"
    )


def main():
    if not PYSIDE6_AVAILABLE:
        print(_missing_qt_message(), file=sys.stderr)
        raise SystemExit(1)

    _enforce_pyside_qt_plugin_env()
    app = QApplication(sys.argv)
    app.setApplicationName("Underwater Comm Visualizer Qt")
    app.setFont(QFont(UI_FONT_FAMILY, 10))
    window = UnderwaterCommVisualizerQt()
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
