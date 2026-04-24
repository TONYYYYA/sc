import os
import sys
import time
import json
import math
import queue
import threading
import importlib.util
import inspect
import re
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import rcParams
from matplotlib import font_manager

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
    "teal_soft": "#d8eeea",
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
    force_offline_loopback: bool = True

class StageBus:
    def __init__(self):
        self.q = queue.Queue()
    def emit(self, stage, payload=None):
        self.q.put({"stage": stage, "payload": payload or {}, "ts": time.time()})

class UnderwaterCommVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("水下通信系统可视化界面")
        self.configure(bg=PALETTE["bg"])
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(1560, max(1360, screen_w - 120))
        win_h = min(980, max(860, screen_h - 120))
        self.geometry(f"{win_w}x{win_h}+40+20")
        self.minsize(1320, 840)

        self.ui_font_family = None
        self._set_default_fonts()
        self._setup_theme()
        self.bus = StageBus()
        self.config_data = RunConfig()
        self.core = None
        self.current_image_path = ""
        self._stl10_test_dataset = None
        self.reconstructed_image_path = str(ROOT_DIR / "rx_output.png")
        self.bitstream_cache = None
        self.tx_wave_path = str(ROOT_DIR / "savedata" / "tx.wav")
        self.current_metric_size = (224, 224)
        self.running = False
        self.last_tx_info = {}
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
            "default_samplerate": 64000,
            "wave_display_mode": "normalized",
            "spec_display_mode": "relative_db",
            "spec_floor_db": -80,
            "freq_xlim": (4000, 12000),
            "carrier_freq": 8000,
        }
        self.monitor_state = {}
        self.stage_badges = {}
        self.metric_value_labels = {}
        self.preview_placeholders = {}

        self._build_ui()
        self._init_monitor_state("tx")
        self._init_monitor_state("rx")
        self._poll_bus()
        self.after(300, self._bind_resize_events)
        self._safe_load_core_module()

    def _set_default_fonts(self):
        try:
            import tkinter.font as tkfont
            self.ui_font_family = UI_FONT_FAMILY
            default_font = tkfont.nametofont("TkDefaultFont")
            text_font = tkfont.nametofont("TkTextFont")
            heading_font = tkfont.nametofont("TkHeadingFont")
            default_font.configure(family=self.ui_font_family, size=10)
            text_font.configure(family=self.ui_font_family, size=10)
            heading_font.configure(family=self.ui_font_family, size=10)
        except Exception:
            self.ui_font_family = UI_FONT_FAMILY

    def _setup_theme(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        base_font = (self.ui_font_family, 10) if self.ui_font_family else ("TkDefaultFont", 10)
        heading_font = (self.ui_font_family, 11, "bold") if self.ui_font_family else ("TkHeadingFont", 11, "bold")
        hero_font = (self.ui_font_family, 22, "bold") if self.ui_font_family else ("TkHeadingFont", 22, "bold")
        small_font = (self.ui_font_family, 9) if self.ui_font_family else ("TkDefaultFont", 9)

        self.style.configure(".", background=PALETTE["bg"], foreground=PALETTE["text"], font=base_font)
        self.style.configure("Shell.TFrame", background=PALETTE["bg"])
        self.style.configure("Card.TFrame", background=PALETTE["panel"], relief="flat")
        self.style.configure("AltCard.TFrame", background=PALETTE["panel_alt"], relief="flat")
        self.style.configure("Hero.TFrame", background=PALETTE["panel_deep"])
        self.style.configure("TLabel", background=PALETTE["bg"], foreground=PALETTE["text"])
        self.style.configure("Panel.TLabel", background=PALETTE["panel"], foreground=PALETTE["text"])
        self.style.configure("Muted.TLabel", background=PALETTE["panel"], foreground=PALETTE["muted"])
        self.style.configure("AltMuted.TLabel", background=PALETTE["panel_alt"], foreground=PALETTE["muted"])
        self.style.configure("CardMuted.TLabel", background=PALETTE["card"], foreground=PALETTE["muted"])
        self.style.configure("HeroTitle.TLabel", background=PALETTE["panel_deep"], foreground="#fff9f1", font=hero_font)
        self.style.configure("HeroSubtitle.TLabel", background=PALETTE["panel_deep"], foreground="#d5e3e3", font=base_font)
        self.style.configure("HeroMeta.TLabel", background=PALETTE["panel_deep"], foreground="#f4d6bf", font=small_font)
        self.style.configure("Section.TLabelframe", background=PALETTE["panel"], borderwidth=1, relief="solid")
        self.style.configure("Section.TLabelframe.Label", background=PALETTE["panel"], foreground=PALETTE["accent_dark"], font=heading_font)
        self.style.configure("AltSection.TLabelframe", background=PALETTE["panel_alt"], borderwidth=1, relief="solid")
        self.style.configure("AltSection.TLabelframe.Label", background=PALETTE["panel_alt"], foreground=PALETTE["teal"], font=heading_font)
        self.style.configure("TEntry", fieldbackground=PALETTE["card"], foreground=PALETTE["text"], bordercolor=PALETTE["border"], lightcolor=PALETTE["border"], darkcolor=PALETTE["border"], insertcolor=PALETTE["text"])
        self.style.configure("TSpinbox", fieldbackground=PALETTE["card"], foreground=PALETTE["text"], arrowsize=14)
        self.style.configure("TCombobox", fieldbackground=PALETTE["card"], foreground=PALETTE["text"], arrowsize=14)
        self.style.map("TCombobox", fieldbackground=[("readonly", PALETTE["card"])], selectbackground=[("readonly", PALETTE["accent_soft"])])
        self.style.configure("TButton", background=PALETTE["accent"], foreground="#fffaf4", borderwidth=0, focusthickness=0, padding=(12, 8), font=base_font)
        self.style.map("TButton", background=[("active", PALETTE["accent_dark"]), ("pressed", PALETTE["accent_dark"])], foreground=[("disabled", "#f2dbcc")])
        self.style.configure("Secondary.TButton", background=PALETTE["teal"], foreground="#f8fffe")
        self.style.map("Secondary.TButton", background=[("active", "#245f61"), ("pressed", "#245f61")])
        self.style.configure("Ghost.TButton", background=PALETTE["panel_alt"], foreground=PALETTE["text"], borderwidth=1, relief="solid", padding=(10, 8))
        self.style.map("Ghost.TButton", background=[("active", PALETTE["accent_soft"]), ("pressed", PALETTE["accent_soft"])])
        self.style.configure("TProgressbar", troughcolor=PALETTE["idle_bg"], background=PALETTE["accent"], bordercolor=PALETTE["idle_bg"], lightcolor=PALETTE["accent"], darkcolor=PALETTE["accent"], thickness=12)

    def _status_colors(self, value):
        text = str(value or "")
        if "失败" in text:
            return PALETTE["error_bg"], PALETTE["error_fg"]
        if any(key in text for key in ("完成", "已完成", "已选择")):
            return PALETTE["success_bg"], PALETTE["success_fg"]
        if any(key in text for key in ("进行中", "准备中")):
            return PALETTE["active_bg"], PALETTE["active_fg"]
        return PALETTE["idle_bg"], PALETTE["idle_fg"]

    def _make_badge(self, parent, text="", width=None):
        bg, fg = self._status_colors(text)
        kwargs = {
            "master": parent,
            "text": text,
            "bg": bg,
            "fg": fg,
            "padx": 10,
            "pady": 4,
            "bd": 0,
            "relief": "flat",
            "anchor": "center",
        }
        if width is not None:
            kwargs["width"] = width
        if self.ui_font_family:
            kwargs["font"] = (self.ui_font_family, 9, "bold")
        return tk.Label(**kwargs)

    def _style_text_widget(self, widget):
        widget.configure(
            bg=PALETTE["card"],
            fg=PALETTE["text"],
            insertbackground=PALETTE["text"],
            selectbackground=PALETTE["accent_soft"],
            selectforeground=PALETTE["text"],
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=PALETTE["border"],
            highlightcolor=PALETTE["accent"],
            padx=10,
            pady=10,
        )
        if self.ui_font_family:
            widget.configure(font=(self.ui_font_family, 10))

    def _draw_preview_placeholder(self, canvas, title, subtitle):
        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), fill=PALETTE["canvas"], outline="")
        canvas.create_oval(28, 28, 92, 92, fill=PALETTE["accent_soft"], outline="")
        canvas.create_text(60, 60, text="~", fill=PALETTE["accent_dark"], font=(self.ui_font_family or "TkDefaultFont", 22, "bold"))
        canvas.create_text(
            max(120, canvas.winfo_width() // 2),
            max(62, canvas.winfo_height() // 2 - 10),
            text=title,
            fill=PALETTE["accent_dark"],
            font=(self.ui_font_family or "TkDefaultFont", 14, "bold"),
        )
        canvas.create_text(
            max(120, canvas.winfo_width() // 2),
            max(92, canvas.winfo_height() // 2 + 18),
            text=subtitle,
            fill=PALETTE["muted"],
            font=(self.ui_font_family or "TkDefaultFont", 10),
        )

    def _build_ui(self):
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self._build_hero_banner()

        left = ttk.Frame(self, padding=(14, 10, 8, 14), style="Shell.TFrame")
        right = ttk.Frame(self, padding=(8, 10, 14, 14), style="Shell.TFrame")
        left.grid(row=1, column=0, sticky="nsew")
        right.grid(row=1, column=1, sticky="nsew")

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=3)
        right.rowconfigure(3, weight=2)
        # Keep image comparison panel visible when upper controls grow.
        right.grid_rowconfigure(2, minsize=260)
        right.grid_rowconfigure(3, minsize=170)
        right.columnconfigure(0, weight=1)

        self._build_control_panel(left)
        self._build_plot_panel(left)
        self._build_status_panel(right)
        self._build_image_panel(right)
        self._build_metrics_panel(right)

    def _build_hero_banner(self):
        hero = ttk.Frame(self, padding=(20, 18, 20, 16), style="Hero.TFrame")
        hero.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 0))
        hero.columnconfigure(0, weight=1)
        hero.columnconfigure(1, weight=0)

        left = ttk.Frame(hero, style="Hero.TFrame")
        left.grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="水下通信实验控制台", style="HeroTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            left,
            text="把发送、接收、重建和评估放进同一块更清晰的观测面板里。",
            style="HeroSubtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        right = ttk.Frame(hero, style="Hero.TFrame")
        right.grid(row=0, column=1, sticky="e")
        self.method_chip_var = tk.StringVar(value="当前方法: 基于语义传输")
        self.method_chip = self._make_badge(right, self.method_chip_var.get())
        self.method_chip.configure(bg="#204c55", fg="#ebfbf7", padx=14, pady=6)
        self.method_chip.grid(row=0, column=0, sticky="e")
        ttk.Label(
            right,
            text="信道监测 · 图像重建 · 性能分析",
            style="HeroMeta.TLabel",
        ).grid(row=1, column=0, sticky="e", pady=(8, 0))

    def _build_control_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="实验配置", padding=14, style="Section.TLabelframe")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        for col in range(4):
            frame.columnconfigure(col, weight=1 if col in (1, 3) else 0)

        ttk.Label(frame, text="源图像", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self.img_entry = ttk.Entry(frame)
        self.img_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="浏览", command=self.choose_image, style="Secondary.TButton").grid(row=0, column=2, padx=6, pady=4)
        ttk.Button(frame, text="STL测试图", command=self.choose_stl_test_image, style="Ghost.TButton").grid(row=0, column=3, padx=6, pady=4)

        ttk.Label(frame, text="传输方法", style="Panel.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self.method_combo = ttk.Combobox(frame, values=list(BUILTIN_METHODS.keys()), state="readonly")
        self.method_combo.set("基于语义传输")
        self.method_combo.grid(row=1, column=1, sticky="ew", pady=4)
        self.method_combo.bind("<<ComboboxSelected>>", self.on_method_selected)
        ttk.Button(frame, text="使用内置", command=self.use_selected_method, style="Ghost.TButton").grid(row=1, column=2, padx=6, pady=4)
        self.method_hint_var = tk.StringVar(value="适合快速验证端到端主流程")
        ttk.Label(frame, textvariable=self.method_hint_var, style="Muted.TLabel").grid(row=1, column=3, sticky="w", padx=(6, 0), pady=4)

        ttk.Label(frame, text="脚本路径", style="Panel.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.core_entry = ttk.Entry(frame)
        self.core_entry.insert(0, self.config_data.core_script_path)
        self.core_entry.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="浏览", command=self.choose_core_script, style="Ghost.TButton").grid(row=2, column=2, padx=6, pady=4)
        ttk.Button(frame, text="重新载入", command=self.reload_core_script, style="Ghost.TButton").grid(row=2, column=3, padx=6, pady=4)

        ttk.Label(
            frame,
            text="JSCC链路模式",
            style="Panel.TLabel",
        ).grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.force_offline_loopback_var = tk.BooleanVar(value=bool(self.config_data.force_offline_loopback))
        ttk.Checkbutton(
            frame,
            text="启用离线回环（不走实际水池）",
            variable=self.force_offline_loopback_var,
        ).grid(row=3, column=1, columnspan=3, sticky="w", pady=4)

        ttk.Label(
            frame,
            text="录音设备由后台自动选择；关闭离线回环后将尝试真实收发。",
            style="Muted.TLabel",
        ).grid(row=4, column=0, columnspan=4, sticky="w", pady=6)

        btns = ttk.Frame(frame, style="Card.TFrame")
        btns.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        ttk.Button(btns, text="1. 发射端分析", command=self.run_tx_analysis).pack(side="left", padx=4)
        ttk.Button(btns, text="2. 发射与传输", command=self.run_full_tx, style="Secondary.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="3. 接收与重建", command=self.run_rx, style="Ghost.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="4. 性能评估", command=self.calc_metrics, style="Ghost.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="5. 性能分析", command=self.show_performance_analysis, style="Ghost.TButton").pack(side="left", padx=4)

    def on_method_selected(self, event=None):
        self.use_selected_method(reload_after=False)

    def use_selected_method(self, reload_after=True):
        method_name = self.method_combo.get().strip()
        path = BUILTIN_METHODS.get(method_name)
        if not path:
            return
        self.core_entry.delete(0, tk.END)
        self.core_entry.insert(0, path)
        self._log(f"已切换内置传输方法: {method_name}")
        if reload_after and not self.running:
            self._safe_load_core_module()

    def _update_method_presentation(self, method_name):
        name = str(method_name or "未命名方法")
        hint_map = {
            "基于语义传输": "压缩最强，适合展示端到端语义恢复效果",
            "基于JPEG传输": "在压缩率和图像质量之间保持平衡",
            "全bit传输": "链路最直观，适合定位物理层与误码问题",
            "JSCC传输": "使用 JSCC Swin 编解码与 4bit tanh 量化进行端到端传输",
        }
        if hasattr(self, "method_chip_var"):
            self.method_chip_var.set(f"当前方法: {name}")
        if hasattr(self, "method_chip"):
            self.method_chip.configure(text=self.method_chip_var.get())
        if hasattr(self, "method_hint_var"):
            self.method_hint_var.set(hint_map.get(name, "自定义方法已载入，可直接运行现有流程"))

    def _configure_plot_axes(self):
        self.fig.patch.set_facecolor(PALETTE["panel_alt"])
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

    def _build_plot_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="传输信号监测", padding=12, style="AltSection.TLabelframe")
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(9.3, 7.8), dpi=100)
        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.07, hspace=0.42, wspace=0.22)

        self.ax_tx_wave = self.fig.add_subplot(221)
        self.ax_tx_spec = self.fig.add_subplot(222)
        self.ax_rx_wave = self.fig.add_subplot(223)
        self.ax_rx_spec = self.fig.add_subplot(224)
        self._configure_plot_axes()

        self.power_label_var = tk.StringVar(value="发射功率: - dB    接收功率: - dB")
        ttk.Label(frame, textvariable=self.power_label_var, style="AltMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 0))

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_status_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="处理流程状态", padding=12, style="Section.TLabelframe")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        self.stage_vars = {}
        stage_names = [
            "算法模块加载", "源图像加载", "语义表征编码", "比特映射与量化", "发射信号生成",
            "物理链路传输", "接收信号采集", "同步检测与解调", "图像语义重建", "性能评估",
        ]
        for i, name in enumerate(stage_names):
            ttk.Label(frame, text=name, style="Panel.TLabel").grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value="等待")
            self.stage_vars[name] = var
            badge = self._make_badge(frame, var.get(), width=10)
            badge.grid(row=i, column=1, sticky="e", pady=4)
            self.stage_badges[name] = badge

        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=10)
        self.progress.grid(row=len(stage_names), column=0, columnspan=2, sticky="ew", pady=(12, 2))

        log_frame = ttk.LabelFrame(parent, text="运行日志", padding=10, style="AltSection.TLabelframe")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=6, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self._style_text_widget(self.log_text)

        log_btn_frame = ttk.Frame(log_frame, width=82, style="AltCard.TFrame")
        log_btn_frame.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        log_btn_frame.grid_propagate(False)
        clear_btn_kwargs = {
            "master": log_btn_frame,
            "text": "清空\n日志",
            "command": self.clear_log,
            "relief": "flat",
            "bd": 0,
            "width": 5,
            "justify": "center",
            "cursor": "hand2",
            "bg": PALETTE["accent"],
            "fg": "#fff9f2",
            "activebackground": PALETTE["accent_dark"],
            "activeforeground": "#fff9f2",
        }
        if self.ui_font_family:
            clear_btn_kwargs["font"] = (self.ui_font_family, 10)
        clear_btn = tk.Button(**clear_btn_kwargs)
        clear_btn.pack(fill="both", expand=True)

    def _build_image_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="图像重建对比", padding=12, style="Section.TLabelframe")
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="源图像", style="Panel.TLabel").grid(row=0, column=0, pady=(0, 4))
        ttk.Label(frame, text="重建图像", style="Panel.TLabel").grid(row=0, column=1, pady=(0, 4))

        self.left_img_container = ttk.Frame(frame, style="AltCard.TFrame")
        self.right_img_container = ttk.Frame(frame, style="AltCard.TFrame")
        self.left_img_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.right_img_container.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.left_img_container.rowconfigure(0, weight=1)
        self.left_img_container.columnconfigure(0, weight=1)
        self.right_img_container.rowconfigure(0, weight=1)
        self.right_img_container.columnconfigure(0, weight=1)

        self.left_img_canvas = tk.Canvas(self.left_img_container, highlightthickness=1, highlightbackground=PALETTE["border"], bg=PALETTE["canvas"])
        self.right_img_canvas = tk.Canvas(self.right_img_container, highlightthickness=1, highlightbackground=PALETTE["border"], bg=PALETTE["canvas"])
        self.left_img_canvas.grid(row=0, column=0, sticky="nsew")
        self.right_img_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_placeholders[self.left_img_canvas] = ("等待源图像", "选择一张图后会在这里显示。")
        self.preview_placeholders[self.right_img_canvas] = ("等待重建结果", "接收恢复完成后会在这里显示。")
        self._draw_preview_placeholder(self.left_img_canvas, *self.preview_placeholders[self.left_img_canvas])
        self._draw_preview_placeholder(self.right_img_canvas, *self.preview_placeholders[self.right_img_canvas])

    def _build_metrics_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="性能指标", padding=12, style="AltSection.TLabelframe")
        frame.grid(row=3, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        for r in range(4):
            frame.rowconfigure(r, weight=1)

        self.metric_vars = {
            "bit_len": tk.StringVar(value="-"),
            "psnr": tk.StringVar(value="-"),
            "ssim": tk.StringVar(value="-"),
            "compression": tk.StringVar(value="-"),
            "ber": tk.StringVar(value="-"),
            "transfer_time": tk.StringVar(value="-"),
            "bit_rate": tk.StringVar(value="-"),
        }

        items = [
            ("传输比特数", "bit_len"), ("PSNR", "psnr"), ("SSIM", "ssim"), ("压缩率", "compression"),
            ("BER", "ber"), ("传输时长", "transfer_time"), ("传输速率", "bit_rate"),
        ]
        for idx, (label, key) in enumerate(items):
            row = idx // 2
            col = idx % 2
            card = ttk.Frame(frame, style="Card.TFrame", padding=(12, 10))
            card.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=label, style="CardMuted.TLabel").grid(row=0, column=0, sticky="w")
            value_label = tk.Label(
                card,
                textvariable=self.metric_vars[key],
                bg=PALETTE["card"],
                fg=PALETTE["accent_dark"] if key in ("psnr", "ssim", "compression") else PALETTE["text"],
                anchor="w",
                padx=0,
                pady=6,
            )
            if self.ui_font_family:
                value_label.configure(font=(self.ui_font_family, 15, "bold"))
            value_label.grid(row=1, column=0, sticky="w")
            self.metric_value_labels[key] = value_label

    def _method_name(self):
        if self.core is not None and hasattr(self.core, "METHOD_NAME"):
            try:
                return str(getattr(self.core, "METHOD_NAME"))
            except Exception:
                pass
        path = self.core_entry.get().strip() or self.config_data.core_script_path
        return Path(path).stem if path else "未命名方法"

    def _safe_load_core_module(self):
        try:
            self._set_stage("算法模块加载", "进行中")
            self.config_data.core_script_path = self.core_entry.get().strip() or DEFAULT_CORE_SCRIPT_PATH
            self.core = self._load_core_module(self.config_data.core_script_path)
            self._sync_method_combo_with_path(self.config_data.core_script_path)
            self._update_method_presentation(self._method_name())
            if hasattr(self.core, "init_system") and callable(self.core.init_system):
                self.core.init_system()
                self._log(f"[{self._method_name()}] 已调用传输方法 init_system()")
            self._set_stage("算法模块加载", "已完成")
            self._log(f"已加载传输方法: {self.config_data.core_script_path}")
        except Exception as e:
            self._set_stage("算法模块加载", f"失败: {e}")
            self._log(f"传输方法加载失败: {e}")
            messagebox.showwarning("传输方法加载失败", "没有成功导入传输方法脚本。")

    @staticmethod
    def _load_core_module(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        module_name = f"uw_comm_core_{int(time.time() * 1000)}"
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
                    self.method_combo.set(name)
                    self._update_method_presentation(name)
                    return
            except Exception:
                continue
        self._update_method_presentation(Path(path).stem if path else "未命名方法")

    def choose_core_script(self):
        path = filedialog.askopenfilename(title="选择传输方法脚本", filetypes=[("Python Files", "*.py")])
        if not path:
            return
        self.core_entry.delete(0, tk.END)
        self.core_entry.insert(0, path)
        self._sync_method_combo_with_path(path)
        self._log(f"已选择传输方法: {path}")

    def reload_core_script(self):
        if self.running:
            messagebox.showinfo("提示", "当前任务执行中，暂时不能重新加载传输方法")
            return
        self._safe_load_core_module()

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择待传输图像",
            filetypes=[
                ("图像文件", "*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG *.bmp *.BMP"),
                ("PNG 图像", "*.png *.PNG"),
                ("JPEG 图像", "*.jpg *.JPG *.jpeg *.JPEG"),
                ("BMP 图像", "*.bmp *.BMP"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.current_image_path = path
        self.img_entry.delete(0, tk.END)
        self.img_entry.insert(0, path)
        self._update_preview(self.left_img_canvas, path)
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
            idx = simpledialog.askinteger(
                "选择 STL10 测试图",
                f"请输入测试集索引 (0 - {max_idx})",
                parent=self,
                minvalue=0,
                maxvalue=max_idx,
            )
            if idx is None:
                return
            img, _ = ds[int(idx)]
            STL10_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = STL10_CACHE_DIR / f"stl10_test_{int(idx):05d}.png"
            img.save(save_path)
            self.current_image_path = str(save_path)
            self.img_entry.delete(0, tk.END)
            self.img_entry.insert(0, str(save_path))
            self._update_preview(self.left_img_canvas, str(save_path))
            self._set_stage("源图像加载", "已选择")
            self._log(f"已加载 STL10 测试图[{idx}]: {save_path}")
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载 STL10 测试图:\n{e}")

    def run_tx_analysis(self):
        if not self.running:
            threading.Thread(target=self._run_tx_analysis_worker, daemon=True).start()

    def run_full_tx(self):
        if not self.running:
            threading.Thread(target=self._run_full_tx_worker, daemon=True).start()

    def run_rx(self):
        if not self.running:
            threading.Thread(target=self._run_rx_worker, daemon=True).start()

    def calc_metrics(self):
        if not self.running:
            threading.Thread(target=self._calc_metrics_worker, daemon=True).start()

    def _validate_before_run(self):
        if self.core is None:
            raise RuntimeError("传输方法未成功加载")
        img_path = self.img_entry.get().strip()
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError("请先选择有效图像")
        self.config_data.img_path = img_path
        self.config_data.core_script_path = self.core_entry.get().strip() or DEFAULT_CORE_SCRIPT_PATH

    def _reset_run_state(self, reset_images=False):
        self.bitstream_cache = None
        self.last_tx_info = {}
        self.tx_start_time = self.tx_end_time = None
        self.rx_start_time = self.rx_end_time = None
        self.total_start_time = self.total_end_time = None
        for key in self.metric_vars:
            self.metric_vars[key].set("-")
        self._reset_monitor("tx")
        self._reset_monitor("rx")
        for name in ["源图像加载","语义表征编码","比特映射与量化","发射信号生成","物理链路传输","接收信号采集","同步检测与解调","图像语义重建","性能评估"]:
            self._set_stage(name, "等待")
        if reset_images:
            for canvas in (self.left_img_canvas, self.right_img_canvas):
                canvas.image = None
                title, subtitle = self.preview_placeholders.get(canvas, ("等待预览", ""))
                self._draw_preview_placeholder(canvas, title, subtitle)

    def _read_effective_bits_from_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return sum(1 for ch in f.read() if ch in "01")
        except Exception:
            return None

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
        if "全bit" in method_name or "bit传输" in method_name:
            try:
                shape = getattr(self.core, "image_shape", None) or (96, 96)
                raw = cv2.imread(img_path)
                if raw is None:
                    return None
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                raw = cv2.resize(raw, tuple(shape))
                return int(raw.size * 8)
            except Exception:
                pass
        if "rawbit" in method_name or "raw_bit" in method_name:
            try:
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
                raw = cv2.imread(img_path)
                if raw is None:
                    return None
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224))
                image = resized.astype(np.float32) / 127.5 - 1.0
                image = image.transpose(2, 0, 1)[None, :, :, :]
                import torch
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

    def _save_performance_result(self, metrics):
        try:
            method_name = self._method_name()
            payload = {
                "method_name": method_name,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source_image": self.img_entry.get().strip(),
                "core_script_path": self.core_entry.get().strip(),
                "metrics": metrics,
            }
            self._ensure_perf_result_dir()
            save_path = self.perf_result_dir / f"{self._safe_method_filename(method_name)}.json"
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.bus.emit("log", {"text": f"[{method_name}] 性能结果已保存到本地: {save_path}"})
        except Exception as e:
            self.bus.emit("log", {"text": f"[{self._method_name()}] 性能结果保存失败: {e}"})

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
                    results.append({
                        "method_name": method_name,
                        "metrics": normalized,
                        "saved_at": data.get("saved_at", ""),
                        "path": str(file),
                    })
            except Exception as e:
                self._log(f"读取性能结果失败 {file.name}: {e}")
        return results

    def show_performance_analysis(self):
        results = self._load_saved_performance_results()
        if not results:
            messagebox.showinfo("提示", "本地还没有已保存的性能评估结果，请先执行“性能评估”。")
            return

        metric_names = sorted({k for item in results for k, v in item["metrics"].items() if v is not None}, key=self._metric_sort_key)
        if not metric_names:
            messagebox.showinfo("提示", "未读取到可用于绘图的指标数据。")
            return

        top = tk.Toplevel(self)
        top.title("性能分析")
        top.configure(bg=PALETTE["bg"])
        top.geometry("1280x940")
        top.minsize(980, 820)
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)

        header = ttk.Frame(top, padding=(16, 16, 16, 8), style="Hero.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 0))
        method_summary = "、".join(item["method_name"] for item in results)
        ttk.Label(header, text="性能结果对比", style="HeroTitle.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"共读取 {len(results)} 个方法：{method_summary}", style="HeroSubtitle.TLabel").pack(anchor="w", pady=(6, 0))

        container = ttk.Frame(top, padding=(12, 10, 12, 6), style="Shell.TFrame")
        container.grid(row=1, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        fig = self._build_performance_figure(results, metric_names)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(top, padding=(12, 0, 18, 18), style="Shell.TFrame")
        footer.grid(row=2, column=0, sticky="e")
        ttk.Button(footer, text="关闭", command=top.destroy, style="Ghost.TButton").pack(anchor="e")

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

            bars = ax.bar(method_names, values, color=colors[:len(method_names)], width=0.62, edgecolor="#555555", linewidth=0.6)
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
            min_positive = min([v for v in values if v > 0], default=None)
            if metric_name in ("ber", "ssim"):
                upper = max_val * 1.18 if max_val > 0 else 1.0
                ax.set_ylim(0, upper)
            elif metric_name == "bit_len":
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
                ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)
            else:
                ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 1.0)

            for bar, raw_value in zip(bars, display_values):
                if raw_value is None:
                    label = "-"
                    y = bar.get_height() + (0.02 if max_val == 0 else max_val * 0.02)
                else:
                    if metric_name == "bit_len":
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
            self.bus.emit("bit_stats", {"length": int(tx_bits)})
        if tx_result.get("tx_wav_path"):
            self.tx_wave_path = tx_result["tx_wav_path"]
        if tx_result.get("rx_wav_path"):
            self.config_data.rx_wav_path = tx_result["rx_wav_path"]

    def _emit_signal_monitor(self, role, data, samplerate):
        arr = np.asarray(data, dtype=np.float32)
        self.bus.emit("signal_monitor", {"role": role, "data": arr.copy(), "samplerate": int(samplerate)})

    def _run_tx_analysis_worker(self):
        self.running = True
        self._reset_run_state(False)
        self.total_start_time = self.tx_start_time = time.perf_counter()
        try:
            self._validate_before_run()
            self.bus.emit("stage", {"name": "源图像加载", "value": "完成"})
            self.bus.emit("preview_left", {"path": self.config_data.img_path})
            self.bus.emit("stage", {"name": "语义表征编码", "value": "进行中"})
            self.bus.emit("stage", {"name": "比特映射与量化", "value": "进行中"})
            bit_len = self._infer_tx_bits_fallback(self.config_data.img_path)
            if bit_len is None:
                raise RuntimeError("无法推断当前传输方法的发送比特数，请执行完整传输以获取精确值。")
            self.bus.emit("bit_stats", {"length": int(bit_len)})
            self.bus.emit("stage", {"name": "语义表征编码", "value": "完成"})
            self.bus.emit("stage", {"name": "比特映射与量化", "value": "完成"})
            self.bus.emit("stage", {"name": "发射信号生成", "value": "仅完整传输时执行"})
            self.tx_end_time = self.total_end_time = time.perf_counter()
            self.bus.emit("timing", {"transfer_time": self.total_end_time - self.total_start_time})
            self.bus.emit("log", {"text": f"[{self._method_name()}] 发射编码分析完成，推断发送比特数={bit_len}"})
        except Exception as e:
            self.bus.emit("log", {"text": f"[{self._method_name()}] 发送端分析失败: {e}"})
        finally:
            self.running = False

    def _run_full_tx_worker(self):
        self.running = True
        self._reset_run_state(False)
        self.total_start_time = self.tx_start_time = time.perf_counter()
        try:
            self._validate_before_run()
            self.bus.emit("preview_left", {"path": self.config_data.img_path})
            self.bus.emit("stage", {"name": "源图像加载", "value": "完成"})
            self.bus.emit("stage", {"name": "物理链路传输", "value": "进行中"})
            self.bus.emit("stage", {"name": "接收信号采集", "value": "准备中"})
            self.bus.emit("log", {"text": f"[{self._method_name()}] 开始执行完整发送流程..."})

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
                force_offline_loopback = bool(self.force_offline_loopback_var.get())
                self.config_data.force_offline_loopback = force_offline_loopback
                if "force_offline_loopback" in tx_sig.parameters or tx_accepts_varkw:
                    tx_kwargs["force_offline_loopback"] = force_offline_loopback
                if "monitor_callback" in tx_sig.parameters:
                    tx_kwargs["monitor_callback"] = self._emit_signal_monitor
                if "log_callback" in tx_sig.parameters:
                    tx_kwargs["log_callback"] = lambda text: self.bus.emit("log", {"text": text})
            except Exception:
                pass

            tx_result = self.core.Tx(self.config_data.img_path, **tx_kwargs)
            self._record_tx_result(tx_result)

            if self._current_bit_length() is None:
                bit_len = self._infer_tx_bits_fallback(self.config_data.img_path)
                if bit_len is not None:
                    self.bus.emit("bit_stats", {"length": int(bit_len)})

            self.tx_end_time = self.total_end_time = time.perf_counter()
            self.bus.emit("stage", {"name": "物理链路传输", "value": "完成"})
            self.bus.emit("stage", {"name": "接收信号采集", "value": "完成"})
            self.bus.emit("timing", {"transfer_time": self.total_end_time - self.total_start_time})
            self.bus.emit("log", {"text": f"[{self._method_name()}] 完整发送流程完成"})
        except Exception as e:
            self.bus.emit("log", {"text": f"[{self._method_name()}] 完整发送失败: {e}"})
        finally:
            self.running = False

    def _run_rx_worker(self):
        self.running = True
        if self.total_start_time is None:
            self.total_start_time = time.perf_counter()
        self.rx_start_time = time.perf_counter()
        try:
            self.bus.emit("stage", {"name": "同步检测与解调", "value": "进行中"})
            rx_kwargs = {}
            try:
                rx_sig = inspect.signature(self.core.Rx)
                if "log_callback" in rx_sig.parameters:
                    rx_kwargs["log_callback"] = lambda text: self.bus.emit("log", {"text": text})
                if "save_img_path" in rx_sig.parameters:
                    rx_kwargs["save_img_path"] = self.reconstructed_image_path
                if "rx_wav_path" in rx_sig.parameters:
                    rx_kwargs["rx_wav_path"] = self.config_data.rx_wav_path
            except Exception:
                pass
            result = self.core.Rx(self.config_data.rx_bits_path, **rx_kwargs)
            if isinstance(result, str) and result:
                self.reconstructed_image_path = result
            self.bus.emit("stage", {"name": "同步检测与解调", "value": "完成"})
            self.bus.emit("stage", {"name": "图像语义重建", "value": "完成"})
            if os.path.exists(self.reconstructed_image_path):
                self.bus.emit("preview_right", {"path": self.reconstructed_image_path})
            self.rx_end_time = self.total_end_time = time.perf_counter()
            self.bus.emit("timing", {"transfer_time": self.total_end_time - self.total_start_time})
            self.bus.emit("log", {"text": f"[{self._method_name()}] 接收恢复完成"})
        except Exception as e:
            self.bus.emit("log", {"text": f"[{self._method_name()}] 接收恢复失败: {e}"})
        finally:
            self.running = False

    def _calc_metrics_worker(self):
        self.running = True
        try:
            img_path = self.img_entry.get().strip()
            rec_path = self.reconstructed_image_path
            if not os.path.exists(img_path) or not os.path.exists(rec_path):
                raise RuntimeError("源图像或重建图像不存在")
            self.bus.emit("stage", {"name": "性能评估", "value": "进行中"})
            metric_size = self._infer_metric_size()
            self.current_metric_size = metric_size
            metric_kwargs = {}
            try:
                metric_sig = inspect.signature(self.core.calc_metrics_and_show)
                if "log_callback" in metric_sig.parameters:
                    metric_kwargs["log_callback"] = lambda text: self.bus.emit("log", {"text": text})
                if "size" in metric_sig.parameters:
                    metric_kwargs["size"] = metric_size
                if "bitstream_path" in metric_sig.parameters:
                    metric_kwargs["bitstream_path"] = self.last_tx_info.get("tx_bitstream_path", self.config_data.tx_bits_path)
            except Exception:
                pass
            psnr, ssim, cr = self.core.calc_metrics_and_show(img_path, rec_path, **metric_kwargs)
            ber = self._compute_ber()
            self.bus.emit("metrics", {"psnr": psnr, "ssim": ssim, "compression": cr, "ber": ber})
            self.bus.emit("stage", {"name": "性能评估", "value": "完成"})
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
            self.bus.emit("log", {"text": f"[{self._method_name()}] 性能评估完成: PSNR={psnr:.2f}, SSIM={ssim:.4f}, BER={'-' if ber is None else f'{ber:.6f}'}, 压缩率={cr * 100:.2f}%, 传输速率={'-' if bitrate is None else f'{bitrate/1000:.2f} kbps'}"})
        except Exception as e:
            self.bus.emit("log", {"text": f"[{self._method_name()}] 性能评估失败: {e}"})
        finally:
            self.running = False

    def _poll_bus(self):
        try:
            while True:
                self._handle_bus_event(self.bus.q.get_nowait())
        except queue.Empty:
            pass
        self.after(25, self._poll_bus)

    def _handle_bus_event(self, item):
        stage = item["stage"]
        payload = item["payload"]
        if stage == "log":
            self._log(payload.get("text", ""))
        elif stage == "stage":
            self._set_stage(payload["name"], payload["value"])
        elif stage == "preview_left":
            self._update_preview(self.left_img_canvas, payload["path"])
        elif stage == "preview_right":
            self._update_preview(self.right_img_canvas, payload["path"])
        elif stage == "signal_monitor":
            self._update_signal_monitor(payload["role"], payload["data"], payload["samplerate"])
        elif stage == "bit_stats":
            self.metric_vars["bit_len"].set(str(payload["length"]))
        elif stage == "metrics":
            self.metric_vars["psnr"].set(f"{payload['psnr']:.2f} dB")
            self.metric_vars["ssim"].set(f"{payload['ssim']:.4f}")
            self.metric_vars["compression"].set(f"{payload['compression'] * 100:.2f}%")
            ber = payload.get("ber")
            self.metric_vars["ber"].set("-" if ber is None else f"{ber:.6f}")
            transfer_time = None
            if self.total_start_time is not None and self.total_end_time is not None:
                transfer_time = self.total_end_time - self.total_start_time
            self.metric_vars["transfer_time"].set("-" if transfer_time is None else f"{transfer_time:.3f} s")
            self._refresh_rate_metrics()
        elif stage == "timing":
            transfer_time = payload.get("transfer_time")
            self.metric_vars["transfer_time"].set("-" if transfer_time is None else f"{transfer_time:.3f} s")

    def _init_monitor_state(self, role):
        sr = self.monitor_cfg["default_samplerate"]
        time_len = int(sr * self.monitor_cfg["time_window_sec"])
        fft_len = self.monitor_cfg["fft_len"]
        state = {"samplerate": sr, "time_buffer": np.zeros(time_len, dtype=np.float32), "latest_chunk": np.zeros(min(2048, fft_len), dtype=np.float32), "power_db": None, "spectrum_freqs": None, "spectrum_mag_db": None}
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
        wave_ax.set_title(wave_title); wave_ax.set_xlabel("Time (s)"); wave_ax.set_ylabel("Amplitude"); wave_ax.set_xlim(0, x_time[-1] if len(x_time) else self.monitor_cfg["time_window_sec"]); wave_ax.set_ylim(-1, 1)
        spec_ax.set_title(spec_title); spec_ax.set_xlabel("Frequency (Hz)"); spec_ax.set_ylabel("幅度 (dB, 相对峰值)")
        fmin, fmax = self.monitor_cfg.get("freq_xlim", (0, sr / 2))
        spec_ax.set_xlim(fmin, fmax); spec_ax.set_ylim(self.monitor_cfg.get("spec_floor_db", -80), 3)
        spec_ax.axvline(self.monitor_cfg.get("carrier_freq", 8000), color=PALETTE["gold"], linestyle="--", linewidth=1)
        self.canvas_plot.draw_idle()

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
        self._redraw_monitor(role)

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
            self._redraw_monitor(base_role)
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
        self._redraw_monitor(base_role)

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
            freqs = state["spectrum_freqs"]; mag_db = state["spectrum_mag_db"]
        else:
            freqs = np.fft.rfftfreq(fft_len, d=1 / sr)
            chunk = state["latest_chunk"]
            if len(chunk) < fft_len:
                padded = np.zeros(fft_len, dtype=np.float32); padded[:len(chunk)] = chunk; chunk = padded
            else:
                chunk = chunk[:fft_len]
            mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
            ref = max(float(np.max(mag)), 1e-8)
            mag_db = 20 * np.log10(mag / ref + 1e-8)
        state["line_spec"].set_data(freqs, mag_db)
        fmin, fmax = self.monitor_cfg.get("freq_xlim", (0, sr / 2))
        state["line_spec"].axes.set_xlim(fmin, min(fmax, sr / 2))
        state["line_spec"].axes.set_ylim(self.monitor_cfg.get("spec_floor_db", -80), 3)
        tx_p = self.monitor_state.get("tx", {}).get("power_db")
        rx_p = self.monitor_state.get("rx", {}).get("power_db")
        self.power_label_var.set(f"发射功率: {'-' if tx_p is None else f'{tx_p:.1f}'} dB    接收功率: {'-' if rx_p is None else f'{rx_p:.1f}'} dB")
        self.canvas_plot.draw_idle()

    def _fit_image_to_canvas(self, pil_img, canvas):
        canvas.update_idletasks()
        w = canvas.winfo_width(); h = canvas.winfo_height()
        if w <= 2 or h <= 2:
            return None, None, None
        pil_img = pil_img.resize((224, 224), Image.LANCZOS)
        src_w, src_h = pil_img.size
        scale = min(w / src_w, h / src_h)
        new_w = max(1, int(src_w * scale)); new_h = max(1, int(src_h * scale))
        return pil_img.resize((new_w, new_h), Image.LANCZOS), w, h

    def _update_preview(self, canvas, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            show_img, canvas_w, canvas_h = self._fit_image_to_canvas(img, canvas)
            if show_img is None:
                self.after(80, lambda: self._update_preview(canvas, image_path)); return
            tk_img = ImageTk.PhotoImage(show_img)
            canvas.delete("all")
            canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=PALETTE["canvas"], outline="")
            shadow_dx = 8
            shadow_dy = 8
            img_x = canvas_w // 2
            img_y = canvas_h // 2
            half_w = show_img.width // 2
            half_h = show_img.height // 2
            canvas.create_rectangle(
                img_x - half_w + shadow_dx,
                img_y - half_h + shadow_dy,
                img_x + half_w + shadow_dx,
                img_y + half_h + shadow_dy,
                fill="#cbb89d",
                outline="",
            )
            canvas.create_image(canvas_w // 2, canvas_h // 2, image=tk_img, anchor="center")
            canvas.image = tk_img
            canvas._source_path = image_path
        except Exception as e:
            self._log(f"图像预览失败: {e}")

    def _bind_resize_events(self):
        self.left_img_canvas.bind("<Configure>", lambda e: self._refresh_preview_widget(self.left_img_canvas))
        self.right_img_canvas.bind("<Configure>", lambda e: self._refresh_preview_widget(self.right_img_canvas))

    def _refresh_preview_widget(self, canvas):
        path = getattr(canvas, "_source_path", None)
        if path and os.path.exists(path):
            self.after(20, lambda: self._update_preview(canvas, path))
        else:
            title, subtitle = self.preview_placeholders.get(canvas, ("等待预览", ""))
            self.after(20, lambda: self._draw_preview_placeholder(canvas, title, subtitle))

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
        try:
            return int(self.metric_vars["bit_len"].get())
        except Exception:
            return None

    def _refresh_rate_metrics(self):
        try:
            transfer_time = float(self.metric_vars["transfer_time"].get().split()[0])
            recon_bits = 224 * 224 * 3 * 8
            self.metric_vars["bit_rate"].set(f"{recon_bits / transfer_time / 1000:.2f} kbps" if transfer_time > 0 else "-")
        except Exception:
            self.metric_vars["bit_rate"].set("-")

    def _set_stage(self, name, value):
        if name in self.stage_vars:
            self.stage_vars[name].set(value)
        badge = self.stage_badges.get(name)
        if badge is not None:
            bg, fg = self._status_colors(value)
            badge.configure(text=value, bg=bg, fg=fg)
        done = sum(1 for v in self.stage_vars.values() if ("完成" in v.get() or "已完成" in v.get() or "已选择" in v.get()))
        self.progress["value"] = done

    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {text}\n")
        self.log_text.see("end")

    def clear_log(self):
        self.log_text.delete("1.0", "end")

if __name__ == "__main__":
    app = UnderwaterCommVisualizer()
    app.mainloop()
