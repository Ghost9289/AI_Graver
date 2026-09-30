from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from ai_retouch import AIRetouchError, ensure_template, find_codex, load_template, retouch_with_openai
from config_store import AppSettings, load_settings, save_settings
from graver_bridge import launch_graver_with_image
from processor import TARGET_DPI, EngravingOptions, ProcessingError, process_portrait, render_preview, set_learned_model
from report_uploader import ReportUploader
from scanner import PhotoScanner
from training import MIN_PAIRS, TrainingStore
from stones import all_stones, get_stone, load_catalog, simulate_on_stone
from updater import UpdateError, UpdateInfo, check_for_update, check_remote_access, download_installer, start_silent_update
from version import APP_VERSION


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "AI Graver"
SETTINGS_PATH = DATA_DIR / "settings.ini"
OUTPUT_DIR = DATA_DIR / "output"
AI_TEMPLATE_PATH = DATA_DIR / "ai_template.txt"
AI_CACHE_DIR = DATA_DIR / "ai_cache"
AUTOLEARN_STATE_PATH = DATA_DIR / "autolearn.json"
AUTO_RETRAIN_NEW_PAIRS = 3
WISHES_HINT = "Например: убрать очки; убрать человека справа; тёмный костюм; волосы не трогать"


def _with_wishes(template: str, wishes: str) -> str:
    """Operator's wishes for this photo go after the template; keeping the person's identity still wins."""
    if not wishes:
        return template
    return (
        f"{template}\n\nOPERATOR WISHES FOR THIS PHOTO (written in Russian; follow them exactly, "
        f"but never change the person's face or identity):\n{wishes}"
    )


STONES_PATH = DATA_DIR / "stones.json"
load_catalog(STONES_PATH)
STONE_LABELS = {f"{stone.group}: {stone.name}": stone.key for stone in all_stones()}
STONE_NAMES = {value: label for label, value in STONE_LABELS.items()}
DEFAULT_STONE_LABEL = STONE_NAMES["black_granite"]
RESTORATION_LABELS = {
    "Естественно": "natural",
    "Сильное восстановление": "strong",
    "Старое / очень плохое фото": "old_photo",
}
RESTORATION_NAMES = {value: label for label, value in RESTORATION_LABELS.items()}


def _desktop_dir() -> Path:
    """Real desktop folder (it is often redirected to OneDrive\Рабочий стол)."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(260)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0, None, 0, buffer) == 0 and buffer.value:
        return Path(buffer.value)
    return Path.home() / "Desktop"


def _find_graver_executable() -> str:
    candidates = [
        Path.home() / "OneDrive" / "Рабочий стол" / "Программы" / "Гравер 5.22.8 — копия" / "graver5_x64.exe",
        Path.home() / "OneDrive" / "Рабочий стол" / "Банана" / "Милосердие" / "Гравер 5.22.8 — копия" / "graver5_x64.exe",
        Path.home() / "Desktop" / "Гравер 5.22.8 — копия" / "graver5_x64.exe",
        APP_DIR / "graver5_x64.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


class AIGraverApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        remote_status = check_remote_access()
        if remote_status is not None and not remote_status.enabled:
            messagebox.showerror("AI Graver отключён", remote_status.message)
            self.destroy()
            raise SystemExit(0)
        self.deiconify()
        self.title(f"AI Graver v{APP_VERSION}")
        self.minsize(980, 700)  # below this the wishes button at the bottom of the sidebar gets cut off
        self.geometry("1280x800")
        self.colors = {
            "ink": "#172033",
            "muted": "#667085",
            "page": "#F4F7FB",
            "card": "#FFFFFF",
            "soft": "#F8FAFC",
            "border": "#E5EAF2",
            "accent": "#4263EB",
            "accent_hover": "#334FC8",
            "accent_soft": "#EDF1FF",
            "header": "#111A2D",
            "header_muted": "#AAB7D1",
            "preview": "#EEF2F7",
        }
        self.configure(bg=self.colors["page"])

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.settings = load_settings(SETTINGS_PATH)
        detected_graver = _find_graver_executable()
        if detected_graver:
            self.settings.graver_exe = detected_graver
        elif not Path(self.settings.graver_exe).is_file():
            self.settings.graver_exe = ""
        self.source_path: Path | None = None
        self.ai_source_path: Path | None = None
        self.training = TrainingStore(DATA_DIR)
        self.reports = ReportUploader(self.training)
        self.scanner = PhotoScanner(DATA_DIR)
        self.training.extra_pairs = self.scanner.pairs
        self.scan_running = False
        self.training_auto = False
        self.ai_failed = False
        self.learned_model = self.training.load_active_model()
        set_learned_model(self.learned_model)
        self.use_learned = tk.BooleanVar(value=True)
        self.training_running = False
        self.restored_path: Path | None = None
        self.result_path: Path | None = None
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.original_preview: ImageTk.PhotoImage | None = None
        self.result_preview: ImageTk.PhotoImage | None = None
        self.preview_timer: str | None = None
        self.preview_revision = 0
        self.preview_running = False
        self.available_update: UpdateInfo | None = None
        self.update_check_in_progress = False

        self.contrast = tk.IntVar(value=self.settings.contrast)
        self.detail = tk.IntVar(value=self.settings.detail)
        self.shadows = tk.IntVar(value=self.settings.shadows)
        self.black_background = tk.BooleanVar(value=self.settings.black_background)
        self.export_bmp = tk.BooleanVar(value=self.settings.export_bmp)
        self.portrait_mode = tk.StringVar(value=self.settings.portrait_mode if self.settings.portrait_mode in {"chest", "full"} else "chest")
        self.graver_exe = tk.StringVar(value=self.settings.graver_exe)
        self.auto_mode = tk.BooleanVar(value=True)
        self.stone_profile = tk.StringVar(value=STONE_NAMES.get(self.settings.stone_profile, DEFAULT_STONE_LABEL))
        self.enhance_4k = tk.BooleanVar(value=self.settings.enhance_4k)
        self.restoration_mode = tk.StringVar(value=RESTORATION_NAMES.get(self.settings.restoration_mode, "Естественно"))
        self.stone_view = tk.BooleanVar(value=True)
        self.last_tone_report = ""
        self.openai_api_key = tk.StringVar(value=self.settings.openai_api_key)
        self.openai_model = tk.StringVar(value=self.settings.openai_model)
        self.status = tk.StringVar(value="Выберите фотографию для подготовки.")

        self._build_compact_ui()
        self.after(150, self._poll_queue)
        self.after(1200, lambda: self._check_updates(announce=False))
        self.after(2500, self._start_reports)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        font = "Segoe UI"
        style.configure("TButton", font=(font, 10, "bold"), borderwidth=0, padding=(15, 10))
        style.configure("Primary.TButton", foreground="#FFFFFF", background=self.colors["accent"])
        style.map("Primary.TButton", background=[("active", self.colors["accent_hover"]), ("disabled", "#A8B4D8")])
        style.configure("Secondary.TButton", foreground=self.colors["ink"], background="#FFFFFF", bordercolor=self.colors["border"])
        style.map("Secondary.TButton", background=[("active", self.colors["soft"])])
        style.configure("Dark.TButton", foreground="#FFFFFF", background="#273653")
        style.map("Dark.TButton", background=[("active", "#344565")])
        style.configure("Modern.TEntry", fieldbackground="#FFFFFF", foreground=self.colors["ink"], bordercolor=self.colors["border"], padding=(10, 9))
        style.configure("Modern.TCheckbutton", background=self.colors["card"], foreground=self.colors["ink"], font=(font, 10))
        style.configure("Modern.TRadiobutton", background=self.colors["card"], foreground=self.colors["ink"], font=(font, 10, "bold"))
        style.configure("Horizontal.TScale", background=self.colors["soft"], troughcolor="#DCE3EF")

        wrapper = tk.Frame(self, bg=self.colors["page"], padx=22, pady=20)
        wrapper.pack(fill="both", expand=True)

        header = tk.Frame(wrapper, bg=self.colors["header"], padx=24, pady=20)
        header.pack(fill="x")
        title_block = tk.Frame(header, bg=self.colors["header"])
        title_block.pack(side="left")
        tk.Label(title_block, text="AI GRAVER", bg=self.colors["header"], fg="#FFFFFF", font=(font, 20, "bold")).pack(anchor="w")
        tk.Label(
            title_block,
            text="Подготовка портрета к гравировке на камне",
            bg=self.colors["header"], fg=self.colors["header_muted"], font=(font, 10),
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            header, text=f"v{APP_VERSION}  •  Локальная обработка", bg="#273653", fg="#DCE7FF", font=(font, 9, "bold"), padx=12, pady=7
        ).pack(side="right", anchor="n")
        ttk.Button(header, text="Проверить обновления", style="Dark.TButton", command=self._check_updates).pack(side="right", padx=(0, 10), anchor="n")

        self.update_banner = tk.Frame(wrapper, bg="#E7F0FF", highlightthickness=1, highlightbackground="#B8CCFF", padx=16, pady=12)
        self.update_message = tk.StringVar()
        tk.Label(self.update_banner, textvariable=self.update_message, bg="#E7F0FF", fg=self.colors["ink"], font=(font, 10, "bold")).pack(side="left")
        ttk.Button(self.update_banner, text="Скачать и установить", style="Primary.TButton", command=self._install_available_update).pack(side="right")

        actions = self._card(wrapper, pady=15)
        actions.pack(fill="x", pady=(16, 12))
        self.actions_card = actions
        left_actions = tk.Frame(actions, bg=self.colors["card"])
        left_actions.pack(side="left")
        ttk.Button(left_actions, text="＋  Выбрать портрет", style="Secondary.TButton", command=self._select_image).pack(side="left")
        self.process_button = ttk.Button(left_actions, text="Сделать для Graver", style="Primary.TButton", command=self._process)
        self.process_button.pack(side="left", padx=(10, 0))
        tk.Label(actions, text="PNG / BMP • 8-bit gray • 83 DPI", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 9)).pack(
            side="right", pady=8
        )

        profile = self._card(wrapper, pady=18)
        profile.pack(fill="x", pady=(0, 12))
        self._section_title(profile, "Профиль гравировки", "Настройте тон и детализацию до обработки")
        controls = tk.Frame(profile, bg=self.colors["card"])
        controls.pack(fill="x", pady=(16, 0))
        for column in range(3):
            controls.columnconfigure(column, weight=1)
        self._add_slider(controls, "Контраст", self.contrast, 0, 80, 0)
        self._add_slider(controls, "Детализация", self.detail, 0, 80, 1)
        self._add_slider(controls, "Тени", self.shadows, 0, 80, 2)
        composition = tk.Frame(profile, bg=self.colors["card"])
        composition.pack(fill="x", pady=(16, 0))
        tk.Label(composition, text="Кадрирование", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 9, "bold")).pack(side="left")
        ttk.Radiobutton(
            composition,
            text="До груди",
            value="chest",
            variable=self.portrait_mode,
            style="Modern.TRadiobutton",
            command=self._schedule_preview,
        ).pack(side="left", padx=(20, 0))
        ttk.Radiobutton(
            composition,
            text="Полный рост",
            value="full",
            variable=self.portrait_mode,
            style="Modern.TRadiobutton",
            command=self._schedule_preview,
        ).pack(side="left", padx=(16, 0))

        options = tk.Frame(profile, bg=self.colors["card"])
        options.pack(fill="x", pady=(12, 0))
        ttk.Checkbutton(
            options,
            text="Автоудаление фона (чёрный фон)",
            variable=self.black_background,
            style="Modern.TCheckbutton",
            command=self._schedule_preview,
        ).pack(side="left")
        ttk.Checkbutton(
            options,
            text="Сохранять BMP вместе с PNG",
            variable=self.export_bmp,
            style="Modern.TCheckbutton",
            command=self._schedule_preview,
        ).pack(side="left", padx=(26, 0))
        tk.Label(options, text="3084 × 5526 px", bg=self.colors["accent_soft"], fg=self.colors["accent"], font=(font, 9, "bold"), padx=10, pady=5).pack(
            side="right"
        )

        graver = self._card(wrapper, pady=16)
        graver.pack(fill="x", pady=(0, 12))
        graver_top = tk.Frame(graver, bg=self.colors["card"])
        graver_top.pack(fill="x")
        self._section_title(graver_top, "Запуск Graver 5", "Путь нужен только кнопке запуска; обработка фото работает и без него", side="left")
        ttk.Button(graver_top, text="Проверить в Graver", style="Dark.TButton", command=self._launch_graver).pack(side="right", anchor="n")
        path_row = tk.Frame(graver, bg=self.colors["card"])
        path_row.pack(fill="x", pady=(14, 0))
        ttk.Entry(path_row, textvariable=self.graver_exe, style="Modern.TEntry").pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="Найти Graver", style="Secondary.TButton", command=self._select_graver).pack(side="left", padx=(10, 0))

        preview = self._card(wrapper, pady=18)
        preview.pack(fill="both", expand=True)
        self._section_title(preview, "Предпросмотр", "Сравните оригинал и подготовленный файл перед импортом")
        preview_grid = tk.Frame(preview, bg=self.colors["card"])
        preview_grid.pack(fill="both", expand=True, pady=(16, 0))
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)
        preview_grid.rowconfigure(0, weight=1)
        self.source_label = self._preview_panel(preview_grid, "Исходник", "Выберите портрет")
        self.source_label.master.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.result_label = self._preview_panel(preview_grid, "Результат", "Здесь появится файл для Graver")
        self.result_label.master.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        footer = tk.Frame(wrapper, bg=self.colors["page"])
        footer.pack(fill="x", pady=(12, 0))
        tk.Label(footer, textvariable=self.status, bg=self.colors["accent_soft"], fg=self.colors["accent"], font=(font, 9, "bold"), padx=12, pady=8).pack(side="left")
        ttk.Button(footer, text="Открыть результаты", style="Secondary.TButton", command=self._open_output_folder).pack(side="right")

    def _build_compact_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        font = "Segoe UI"
        style.configure("TButton", font=(font, 9, "bold"), borderwidth=0, padding=(12, 8))
        style.configure("Primary.TButton", foreground="#FFFFFF", background=self.colors["accent"])
        style.map("Primary.TButton", background=[("active", self.colors["accent_hover"]), ("disabled", "#A8B4D8")])
        style.configure("Secondary.TButton", foreground=self.colors["ink"], background="#FFFFFF", bordercolor=self.colors["border"])
        style.map("Secondary.TButton", background=[("active", self.colors["soft"])])
        style.configure("Dark.TButton", foreground="#FFFFFF", background="#273653")
        style.map("Dark.TButton", background=[("active", "#344565")])
        style.configure("Modern.TEntry", fieldbackground="#FFFFFF", foreground=self.colors["ink"], bordercolor=self.colors["border"], padding=(8, 7))
        style.configure("Modern.TCheckbutton", background=self.colors["card"], foreground=self.colors["ink"], font=(font, 9))
        style.configure("Modern.TRadiobutton", background=self.colors["card"], foreground=self.colors["ink"], font=(font, 9, "bold"))
        style.configure("Horizontal.TScale", background=self.colors["card"], troughcolor="#DCE3EF")

        root = tk.Frame(self, bg=self.colors["page"])
        root.pack(fill="both", expand=True)

        header = tk.Frame(root, bg=self.colors["header"], padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="AI GRAVER", bg=self.colors["header"], fg="#FFFFFF", font=(font, 16, "bold")).pack(side="left")
        tk.Label(header, text="Автоматическая подготовка портрета", bg=self.colors["header"], fg=self.colors["header_muted"], font=(font, 9)).pack(side="left", padx=(14, 0))
        tk.Label(header, text=f"v{APP_VERSION}", bg="#273653", fg="#DCE7FF", font=(font, 9, "bold"), padx=10, pady=5).pack(side="right")
        ttk.Button(header, text="Обновления", style="Dark.TButton", command=self._check_updates).pack(side="right", padx=(0, 8))
        self.header_menu = tk.Menu(self, tearoff=False, font=(font, 9))
        self.header_menu.add_command(label="Обучение…", command=self._open_training)
        self.header_menu.add_command(label="Открыть результаты", command=self._open_output_folder)
        self.header_menu.add_command(label="Путь к Graver…", command=self._select_graver)
        self.header_menu.add_command(label="AI-ретушь: ключ и шаблон…", command=self._open_ai_settings)
        menu_button = ttk.Button(header, text="Меню ▾", style="Dark.TButton")
        menu_button.configure(
            command=lambda: self.header_menu.tk_popup(menu_button.winfo_rootx(), menu_button.winfo_rooty() + menu_button.winfo_height())
        )
        menu_button.pack(side="right", padx=(0, 8))

        body = tk.Frame(root, bg=self.colors["page"], padx=12, pady=12)
        body.pack(fill="both", expand=True)

        sidebar = tk.Frame(body, width=292, bg=self.colors["card"], highlightthickness=1, highlightbackground=self.colors["border"], padx=14, pady=14)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self.update_banner = tk.Frame(sidebar, bg="#E7F0FF", highlightthickness=1, highlightbackground="#B8CCFF", padx=10, pady=8)
        self.update_message = tk.StringVar()
        tk.Label(self.update_banner, textvariable=self.update_message, bg="#E7F0FF", fg=self.colors["ink"], wraplength=230, justify="left", font=(font, 9, "bold")).pack(anchor="w")
        ttk.Button(self.update_banner, text="Установить", style="Primary.TButton", command=self._install_available_update).pack(fill="x", pady=(7, 0))

        self.actions_card = tk.Frame(sidebar, bg=self.colors["card"])
        self.actions_card.pack(fill="x")
        tk.Label(self.actions_card, text="НОВЫЙ ПОРТРЕТ", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8, "bold")).pack(anchor="w")
        ttk.Button(self.actions_card, text="＋  Выбрать фотографию", style="Primary.TButton", command=self._select_image).pack(fill="x", pady=(7, 6))
        self.process_button = ttk.Button(self.actions_card, text="Обработать повторно", style="Secondary.TButton", command=self._process)
        self.process_button.pack(fill="x")
        self.result_actions = tk.Frame(self.actions_card, bg=self.colors["card"])
        self.result_actions.pack(fill="x", pady=(6, 0))
        self.save_button = ttk.Button(self.result_actions, text="💾 Сохранить как…", style="Secondary.TButton", command=self._save_result_as)
        self.save_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.open_in_graver_button = ttk.Button(self.result_actions, text="🖨 Открыть в Graver", style="Primary.TButton", command=self._launch_graver)
        self.open_in_graver_button.pack(side="left", fill="x", expand=True, padx=(3, 0))
        self.save_button.state(["disabled"])
        self.open_in_graver_button.state(["disabled"])
        tk.Label(self.actions_card, text="AI-ретушь → обработка → сохранение → Graver", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8)).pack(anchor="w", pady=(6, 0))

        ttk.Separator(sidebar).pack(fill="x", pady=8)
        tk.Label(sidebar, text="РУЧНАЯ КОРРЕКЦИЯ", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8, "bold")).pack(anchor="w")
        stone_row = tk.Frame(sidebar, bg=self.colors["card"])
        stone_row.pack(fill="x", pady=(6, 1))
        tk.Label(stone_row, text="Камень памятника", bg=self.colors["card"], fg=self.colors["ink"], font=(font, 9)).pack(side="left")
        ttk.Checkbutton(stone_row, text="вид на камне", variable=self.stone_view, style="Modern.TCheckbutton", command=self._refresh_result_view).pack(side="right")
        stone_picker = ttk.Combobox(sidebar, textvariable=self.stone_profile, values=list(STONE_LABELS), state="readonly", height=20, font=(font, 8))
        stone_picker.pack(fill="x", pady=(2, 0))
        stone_picker.bind("<<ComboboxSelected>>", lambda _event: self._schedule_preview(delay=50))
        ttk.Checkbutton(sidebar, text="AI-ретушь (ChatGPT) + лицо + 4K", variable=self.enhance_4k, style="Modern.TCheckbutton", command=self._schedule_preview).pack(anchor="w", pady=(4, 0))
        restoration_picker = ttk.Combobox(sidebar, textvariable=self.restoration_mode, values=list(RESTORATION_LABELS), state="readonly", width=26, font=(font, 8))
        restoration_picker.pack(fill="x", pady=(3, 0))
        restoration_picker.bind("<<ComboboxSelected>>", lambda _event: self._schedule_preview(delay=50))
        self._add_compact_slider(sidebar, "Контраст", self.contrast, 0, 80)
        self._add_compact_slider(sidebar, "Детализация", self.detail, 0, 80)
        self._add_compact_slider(sidebar, "Тени", self.shadows, 0, 80)

        crop = tk.Frame(sidebar, bg=self.colors["card"])
        crop.pack(fill="x", pady=(9, 0))
        tk.Label(crop, text="Кадр", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8, "bold")).pack(anchor="w")
        ttk.Radiobutton(crop, text="До груди", value="chest", variable=self.portrait_mode, style="Modern.TRadiobutton", command=self._schedule_preview).pack(side="left", pady=(5, 0))
        ttk.Radiobutton(crop, text="Полный рост", value="full", variable=self.portrait_mode, style="Modern.TRadiobutton", command=self._schedule_preview).pack(side="left", padx=(10, 0), pady=(5, 0))
        output_options = tk.Frame(sidebar, bg=self.colors["card"])
        output_options.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(output_options, text="Удалять фон", variable=self.black_background, style="Modern.TCheckbutton", command=self._schedule_preview).pack(side="left")
        ttk.Checkbutton(output_options, text="BMP", variable=self.export_bmp, style="Modern.TCheckbutton").pack(side="right")

        ttk.Separator(sidebar).pack(fill="x", pady=8)
        tk.Label(sidebar, text="ПОЖЕЛАНИЯ К РЕТУШИ ЭТОГО ФОТО", bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8, "bold")).pack(anchor="w")
        self.wishes_text = tk.Text(
            sidebar, height=3, wrap="word", font=(font, 9), relief="flat", padx=6, pady=4,
            bg=self.colors["soft"], fg=self.colors["ink"], insertbackground=self.colors["ink"],
            highlightthickness=1, highlightbackground=self.colors["border"], highlightcolor=self.colors["accent"],
        )
        self.wishes_text.pack(fill="x", pady=(4, 4))
        self._wishes_placeholder(show=True)
        self.wishes_text.bind("<FocusIn>", lambda _event: self._wishes_placeholder(show=False))
        self.wishes_text.bind("<FocusOut>", lambda _event: self._wishes_placeholder(show=not self._wishes()))
        ttk.Button(sidebar, text="Переделать с пожеланиями", style="Primary.TButton", command=self._process).pack(fill="x")

        workspace = tk.Frame(body, bg=self.colors["card"], highlightthickness=1, highlightbackground=self.colors["border"], padx=14, pady=12)
        workspace.pack(side="left", fill="both", expand=True, padx=(12, 0))
        workspace_header = tk.Frame(workspace, bg=self.colors["card"])
        workspace_header.pack(fill="x")
        tk.Label(workspace_header, text="Рабочая область", bg=self.colors["card"], fg=self.colors["ink"], font=(font, 11, "bold")).pack(side="left")
        tk.Label(workspace_header, text="PNG / BMP  •  3084 × 5526  •  83 DPI", bg=self.colors["accent_soft"], fg=self.colors["accent"], font=(font, 8, "bold"), padx=9, pady=4).pack(side="right")

        preview_grid = tk.Frame(workspace, bg=self.colors["card"])
        preview_grid.pack(fill="both", expand=True, pady=(10, 8))
        preview_grid.columnconfigure(0, weight=1)
        preview_grid.columnconfigure(1, weight=1)
        preview_grid.rowconfigure(0, weight=1)
        self.source_label = self._preview_panel(preview_grid, "Исходник / улучшено 4K", "Выберите фотографию слева")
        self.source_label.master.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.result_label = self._preview_panel(preview_grid, "Готово для Graver", "Результат появится автоматически")
        self.result_label.master.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.source_label.configure(cursor="hand2")
        self.result_label.configure(cursor="hand2")
        self.source_label.bind("<Double-Button-1>", lambda _event: self._open_full_size("original"))
        self.result_label.bind("<Double-Button-1>", lambda _event: self._open_full_size("result"))
        tk.Label(workspace, textvariable=self.status, bg=self.colors["accent_soft"], fg=self.colors["accent"], anchor="w", font=(font, 9, "bold"), padx=10, pady=7).pack(fill="x")

    def _add_compact_slider(self, parent: tk.Misc, label: str, variable: tk.IntVar, minimum: int, maximum: int) -> None:
        row = tk.Frame(parent, bg=self.colors["card"])
        row.pack(fill="x", pady=(5, 0))
        tk.Label(row, text=label, bg=self.colors["card"], fg=self.colors["ink"], font=("Segoe UI", 9)).pack(side="left")
        tk.Label(row, textvariable=variable, bg=self.colors["accent_soft"], fg=self.colors["accent"], font=("Segoe UI", 8, "bold"), padx=5, pady=1).pack(side="right")
        scale = ttk.Scale(parent, from_=minimum, to=maximum, variable=variable, orient="horizontal", command=lambda value, target=variable: self._on_slider_change(target, value))
        scale.pack(fill="x")
        scale.bind("<ButtonRelease-1>", lambda _event: self._schedule_preview(delay=50))

    def _card(self, parent: tk.Misc, padx: int = 20, pady: int = 18) -> tk.Frame:
        return tk.Frame(parent, bg=self.colors["card"], highlightthickness=1, highlightbackground=self.colors["border"], padx=padx, pady=pady)

    def _section_title(self, parent: tk.Misc, title: str, subtitle: str, side: str = "top") -> None:
        block = tk.Frame(parent, bg=self.colors["card"])
        block.pack(side=side, anchor="nw")
        tk.Label(block, text=title, bg=self.colors["card"], fg=self.colors["ink"], font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(block, text=subtitle, bg=self.colors["card"], fg=self.colors["muted"], font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

    def _add_slider(self, parent: tk.Misc, label: str, variable: tk.IntVar, minimum: int, maximum: int, column: int) -> None:
        container = tk.Frame(parent, bg=self.colors["soft"], padx=12, pady=10)
        container.grid(row=0, column=column, sticky="ew", padx=5)
        heading = tk.Frame(container, bg=self.colors["soft"])
        heading.pack(fill="x")
        tk.Label(heading, text=label, bg=self.colors["soft"], fg=self.colors["ink"], font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(heading, textvariable=variable, bg=self.colors["accent_soft"], fg=self.colors["accent"], font=("Segoe UI", 9, "bold"), padx=6, pady=2).pack(side="right")
        scale = ttk.Scale(
            container,
            from_=minimum,
            to=maximum,
            variable=variable,
            orient="horizontal",
            command=lambda value, target=variable: self._on_slider_change(target, value),
        )
        scale.pack(fill="x", pady=(10, 1))
        scale.bind("<ButtonRelease-1>", lambda _event: self._schedule_preview(delay=50))

    def _preview_panel(self, parent: tk.Misc, title: str, placeholder: str) -> tk.Label:
        panel = tk.Frame(parent, bg=self.colors["soft"], padx=12, pady=12)
        tk.Label(panel, text=title.upper(), bg=self.colors["soft"], fg=self.colors["muted"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        image_label = tk.Label(
            panel,
            text=placeholder,
            bg=self.colors["preview"],
            fg=self.colors["muted"],
            font=("Segoe UI", 10),
            justify="center",
            anchor="center",
        )
        image_label.pack(fill="both", expand=True, pady=(10, 0))
        return image_label

    def _wishes(self) -> str:
        """Operator's wishes for this photo (empty while the grey example hint is shown)."""
        if getattr(self, "_wishes_hint_shown", False):
            return ""
        return self.wishes_text.get("1.0", "end").strip()

    def _wishes_placeholder(self, show: bool) -> None:
        if show and not getattr(self, "_wishes_hint_shown", False):
            self.wishes_text.delete("1.0", "end")
            self.wishes_text.insert("1.0", WISHES_HINT)
            self.wishes_text.configure(fg=self.colors["muted"])
            self._wishes_hint_shown = True
        elif not show and getattr(self, "_wishes_hint_shown", False):
            self.wishes_text.delete("1.0", "end")
            self.wishes_text.configure(fg=self.colors["ink"])
            self._wishes_hint_shown = False

    def _select_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите портрет",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Все файлы", "*.*")],
        )
        if not selected:
            return
        self.source_path = Path(selected)
        self.ai_source_path = None
        self.restored_path = None
        # Wishes belong to one photo: never carry "remove the glasses" over to the next client.
        self._wishes_placeholder(show=False)
        self._wishes_placeholder(show=True)
        self._show_preview(self.source_path, self.source_label, "original")
        self.result_path = None
        self._update_result_actions_state()
        if self.auto_mode.get():
            self.status.set("Автоматически удаляю фон и готовлю файл для Graver…")
            self.after(40, self._process)
        else:
            self.status.set("Создаю быстрый предпросмотр с текущими настройками…")
            self._schedule_preview(delay=20)

    def _select_graver(self) -> None:
        selected = filedialog.askopenfilename(
            title="Выберите graver5_x64.exe",
            filetypes=[("Приложение Windows", "*.exe"), ("Все файлы", "*.*")],
        )
        if selected:
            self.graver_exe.set(selected)

    def _process(self) -> None:
        if not self.source_path:
            messagebox.showwarning("Нет фотографии", "Сначала выберите исходную фотографию.")
            return
        self.process_button.state(["disabled"])
        if self.enhance_4k.get():
            self.status.set("AI-ретушь через ChatGPT — обычно 1–3 минуты, затем подготовка 4K…")
        else:
            self.status.set("Обработка изображения…")
        options = EngravingOptions(
            contrast=self.contrast.get(),
            detail=self.detail.get(),
            shadows=self.shadows.get(),
            black_background=self.black_background.get(),
            export_bmp=self.export_bmp.get(),
            portrait_mode=self.portrait_mode.get(),
            stone_profile=self._stone_key(),
            enhance_4k=self.enhance_4k.get(),
            restoration_mode=RESTORATION_LABELS.get(self.restoration_mode.get(), "natural"),
            use_learned=self.use_learned.get(),
        )
        ai_config = self._ai_config() if options.enhance_4k else None
        wishes = self._wishes()
        self.ai_failed = False
        if ai_config:
            ai_config["wishes"] = wishes
        else:
            self.ai_failed = True
        thread = threading.Thread(target=self._run_processing, args=(self.source_path, options, ai_config), daemon=True)
        thread.start()

    def _on_slider_change(self, variable: tk.IntVar, value: str) -> None:
        variable.set(round(float(value)))
        self._schedule_preview()

    def _schedule_preview(self, delay: int = 350) -> None:
        if not self.source_path:
            return
        self.preview_revision += 1
        revision = self.preview_revision
        self.result_path = None
        self._update_result_actions_state()
        if self.preview_timer:
            self.after_cancel(self.preview_timer)
        self.status.set("Настройки изменены — обновляю предпросмотр…")
        self.preview_timer = self.after(delay, lambda: self._start_preview(revision))

    def _start_preview(self, revision: int) -> None:
        if not self.source_path or revision != self.preview_revision:
            return
        if self.preview_running:
            self.preview_timer = self.after(250, lambda: self._start_preview(self.preview_revision))
            return
        self.preview_running = True
        options = EngravingOptions(
            contrast=self.contrast.get(),
            detail=self.detail.get(),
            shadows=self.shadows.get(),
            black_background=self.black_background.get(),
            export_bmp=self.export_bmp.get(),
            portrait_mode=self.portrait_mode.get(),
            stone_profile=self._stone_key(),
            enhance_4k=self.enhance_4k.get(),
            restoration_mode=RESTORATION_LABELS.get(self.restoration_mode.get(), "natural"),
            use_learned=self.use_learned.get(),
        )
        source = self.source_path
        if self.ai_source_path and self.ai_source_path.is_file() and options.enhance_4k:
            source, options = self.ai_source_path, replace(options, ai_retouched=True)
        thread = threading.Thread(target=self._run_preview, args=(revision, source, options), daemon=True)
        thread.start()

    def _stone_key(self) -> str:
        return STONE_LABELS.get(self.stone_profile.get(), "black_granite")

    def _run_preview(self, revision: int, source: Path, options: EngravingOptions) -> None:
        try:
            self.message_queue.put(("preview", (revision, render_preview(source, options))))
        except Exception as error:
            self.message_queue.put(("preview_error", error))

    def _check_updates(self, announce: bool = True) -> None:
        if self.update_check_in_progress:
            return
        self.update_check_in_progress = True
        if announce:
            self.status.set("Проверяю обновления…")
        thread = threading.Thread(target=self._run_update_check, args=(announce,), daemon=True)
        thread.start()

    def _run_update_check(self, announce: bool) -> None:
        try:
            self.message_queue.put(("update_check", (check_for_update(), announce)))
        except UpdateError as error:
            self.message_queue.put(("update_error", (error, announce)))

    def _install_available_update(self) -> None:
        if not self.available_update:
            return
        self.status.set(f"Скачиваю обновление v{self.available_update.version}…")
        thread = threading.Thread(target=self._run_update_download, args=(self.available_update,), daemon=True)
        thread.start()

    def _run_update_download(self, update: UpdateInfo) -> None:
        try:
            self.message_queue.put(("update_ready", download_installer(update)))
        except UpdateError as error:
            self.message_queue.put(("error", error))

    def _show_update_banner(self, update: UpdateInfo) -> None:
        self.available_update = update
        self.update_message.set(f"Доступна AI Graver v{update.version}")
        if not self.update_banner.winfo_ismapped():
            self.update_banner.pack(fill="x", pady=(16, 0), before=self.actions_card)

    def _log_ai_problem(self, text: str) -> None:
        try:
            self.training.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.training.log_dir / "ai_errors.log").open("a", encoding="utf-8") as file:
                file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        except OSError:
            pass

    def _ai_config(self) -> dict | None:
        keys = [self.openai_api_key.get().strip(), os.environ.get("OPENAI_API_KEY", "").strip()]
        keys = list(dict.fromkeys(key for key in keys if key))
        if not keys and not find_codex():
            self._log_ai_problem("нет ни Codex, ни ключа OpenAI")
            return None
        return {"api_keys": keys, "model": self.openai_model.get().strip()}

    def _run_processing(self, source: Path, options: EngravingOptions, ai_config: dict | None = None) -> None:
        if ai_config:
            try:
                stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem) or "portrait"
                source = retouch_with_openai(
                    source,
                    OUTPUT_DIR / f"{stem}_ai.png",
                    ai_config["api_keys"],
                    _with_wishes(load_template(AI_TEMPLATE_PATH), ai_config.get("wishes", "")),
                    AI_CACHE_DIR,
                    ai_config["model"],
                )
                # The retouch already restored the face: the local face model would only soften it.
                options = replace(options, ai_retouched=True)
                self.message_queue.put(("ai_ready", source))
            except AIRetouchError as error:
                self.message_queue.put(("ai_error", error))
        try:
            reports: list = []
            result = process_portrait(source, OUTPUT_DIR, options, reports)
            if reports:
                self.last_tone_report = reports[0].summary(get_stone(options.stone_profile))
            self.training.log_processing({
                "source": hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:10],  # names often hold surnames
                "stone": options.stone_profile,
                "ai_retouch": options.ai_retouched,
                "restoration_mode": options.restoration_mode,
                "wishes": bool(ai_config and ai_config.get("wishes")),  # the text itself stays on this PC
                "learned": bool(options.use_learned and self.learned_model),
                "contrast": options.contrast,
                "detail": options.detail,
                "shadows": options.shadows,
                "portrait_mode": options.portrait_mode,
                "tone": self.last_tone_report,
            })
            self.message_queue.put(("done", (result, options.ai_retouched)))
        except Exception as error:  # display full processing errors in GUI
            self.message_queue.put(("error", error))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "update_check":
                    self.update_check_in_progress = False
                    update, announce = payload  # type: ignore[misc]
                    if update:
                        self._show_update_banner(update)
                        if announce:
                            self.status.set(f"Найдена версия v{update.version}.")
                    elif announce:
                        self.status.set("Установлена последняя версия.")
                elif kind == "update_error":
                    self.update_check_in_progress = False
                    error, announce = payload  # type: ignore[misc]
                    if announce:
                        self.status.set(str(error))
                elif kind == "update_ready":
                    installer = payload
                    self.status.set("Устанавливаю обновление и перезапускаю программу…")
                    try:
                        start_silent_update(installer)
                    except UpdateError as error:
                        self.status.set(str(error))
                    else:
                        self.after(700, self.destroy)
                elif kind == "done":
                    self.process_button.state(["!disabled"])
                    self.result_path, ai_used = payload  # type: ignore[misc]
                    self._update_result_actions_state()
                    restored_path = self.result_path.with_name(self.result_path.name.replace("_graver_ready.png", "_restored_4k.png"))
                    if restored_path.is_file():
                        self.restored_path = restored_path
                        self._show_preview(self.restored_path, self.source_label, "original")
                    self._refresh_result_view()
                    method = "AI-ретушь ChatGPT" if ai_used else ("локальный GFPGAN" if self.enhance_4k.get() else "без AI")
                    self.status.set(f"Готово ({method}). {self.last_tone_report}. Двойной щелчок — полный размер.")
                    if self.auto_mode.get() and Path(self.graver_exe.get().strip()).is_file():
                        self.after(250, self._launch_graver)
                    self.reports.upload_in_background()
                elif kind == "training_progress":
                    self.status.set(str(payload))
                elif kind == "scan_done":
                    self._after_scan(payload)
                elif kind == "training_done":
                    self.training_running = False
                    model, stats = payload  # type: ignore[misc]
                    AUTOLEARN_STATE_PATH.write_text(
                        json.dumps({"trained_pairs": len(self.training.all_pairs()), "model": model.updated}), encoding="utf-8"
                    )
                    self.learned_model = model
                    set_learned_model(model)
                    compared = [item for item in stats if "difference_after" in item]
                    better = sum(item["difference_after"] < item["difference_before"] for item in compared)
                    self.status.set(f"{model.summary()}. Ближе к ручному финалу: {better} из {len(compared)} примеров.")
                    self._schedule_preview(delay=50)
                    self.reports.upload_in_background()
                elif kind == "training_error":
                    self.training_running = False
                    self.status.set(f"Обучение не выполнено: {payload}")
                    if not self.training_auto:
                        messagebox.showwarning("Обучение", str(payload))
                elif kind == "ai_ready":
                    self.ai_source_path = payload  # type: ignore[assignment]
                    self._show_preview(self.ai_source_path, self.source_label, "original")
                elif kind == "ai_error":
                    # Technical details (Codex, keys, balance) go to the log and the report, not on screen.
                    self.ai_failed = True
                    self._log_ai_problem(str(payload))
                    self.status.set("AI-ретушь недоступна — восстанавливаю лицо локально (GFPGAN), 1–3 минуты…")
                elif kind == "graver_launched":
                    self._update_result_actions_state()
                    self.status.set(payload.message)  # type: ignore[union-attr]
                elif kind == "graver_error":
                    self._update_result_actions_state()
                    messagebox.showerror("Не удалось запустить Graver", str(payload))
                elif kind == "preview":
                    self.preview_running = False
                    revision, image = payload  # type: ignore[misc]
                    if revision == self.preview_revision:
                        if self.stone_view.get():
                            image = simulate_on_stone(image, get_stone(self._stone_key()))
                        photo = ImageTk.PhotoImage(image)
                        self.result_label.configure(image=photo, text="")
                        self.result_preview = photo
                        self.status.set("Предпросмотр обновлён. Нажмите «Обработать повторно», чтобы сохранить файл.")
                    else:
                        self._schedule_preview(delay=20)
                elif kind == "preview_error":
                    self.preview_running = False
                    self.status.set(f"Не удалось обновить предпросмотр: {payload}")
                else:
                    self.process_button.state(["!disabled"])
                    error = payload
                    self.status.set("Не удалось обработать изображение.")
                    messagebox.showerror("Ошибка обработки", str(error))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _refresh_result_view(self) -> None:
        if not self.result_path or not self.result_path.is_file():
            self._schedule_preview(delay=50)
            return
        stone_path = self.result_path.with_name(self.result_path.name.replace("_graver_ready.png", "_on_stone.jpg"))
        if self.stone_view.get() and stone_path.is_file():
            self._show_preview(stone_path, self.result_label, "result")
        else:
            self._show_preview(self.result_path, self.result_label, "result")

    def _show_preview(self, image_path: Path, label: tk.Label, kind: str) -> None:
        with Image.open(image_path) as image:
            preview = ImageOps.contain(image.convert("RGB"), (500, 520))
        photo = ImageTk.PhotoImage(preview)
        label.configure(image=photo)
        if kind == "original":
            self.original_preview = photo
        else:
            self.result_preview = photo

    def _open_ai_settings(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("AI-ретушь OpenAI")
        dialog.configure(bg=self.colors["card"], padx=16, pady=14)
        dialog.resizable(False, False)
        dialog.transient(self)
        font = "Segoe UI"
        tk.Label(dialog, text="Ключ OpenAI API (platform.openai.com → API keys)", bg=self.colors["card"], fg=self.colors["ink"], font=(font, 9, "bold")).pack(anchor="w")
        key_entry = ttk.Entry(dialog, textvariable=self.openai_api_key, show="•", width=52, style="Modern.TEntry")
        key_entry.pack(fill="x", pady=(4, 10))
        tk.Label(dialog, text="Модель", bg=self.colors["card"], fg=self.colors["ink"], font=(font, 9, "bold")).pack(anchor="w")
        ttk.Entry(dialog, textvariable=self.openai_model, width=52, style="Modern.TEntry").pack(fill="x", pady=(4, 10))
        tk.Label(
            dialog,
            text="Шаблон — это текст-задание, которое отправляется вместе с каждым фото.\n"
            "Его можно поправить в Блокноте; изменения применяются к следующему фото.",
            bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8), justify="left",
        ).pack(anchor="w")
        buttons = tk.Frame(dialog, bg=self.colors["card"])
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Открыть шаблон", style="Secondary.TButton", command=lambda: os.startfile(ensure_template(AI_TEMPLATE_PATH))).pack(side="left")

        def save_and_close() -> None:
            self._save_settings()
            dialog.destroy()

        ttk.Button(buttons, text="Сохранить", style="Primary.TButton", command=save_and_close).pack(side="right")
        key_entry.focus_set()

    def _open_training(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Обучение AI Graver")
        dialog.configure(bg=self.colors["card"], padx=16, pady=14)
        dialog.resizable(False, False)
        dialog.transient(self)
        font = "Segoe UI"
        def info_text() -> str:
            model = self.learned_model.summary() if self.learned_model else "Модель ещё не обучена"
            folders = len(self.scanner.folders())
            return (
                f"{model}\nВаших примеров: {self.training.pair_count()}, "
                f"найдено в папках самообучения ({folders}): {len(self.scanner.pairs())}"
            )

        info = tk.StringVar(value=info_text())
        tk.Label(dialog, textvariable=info, bg=self.colors["card"], fg=self.colors["ink"], font=(font, 9, "bold"), justify="left").pack(anchor="w")
        tk.Label(
            dialog,
            text="Программа учится на парах «исходное фото → ваш готовый файл для станка»:\n"
            "где добавить белого, где чёрного и каким должен быть тон лица.",
            bg=self.colors["card"], fg=self.colors["muted"], font=(font, 8), justify="left",
        ).pack(anchor="w", pady=(4, 8))
        ttk.Checkbutton(dialog, text="Применять обучение при обработке", variable=self.use_learned, style="Modern.TCheckbutton", command=self._schedule_preview).pack(anchor="w", pady=(0, 8))

        def refresh() -> None:
            info.set(info_text())

        def add_scan_folder() -> None:
            folder = filedialog.askdirectory(parent=dialog, title="Папка, где лежат ваши заказы (исходники и файлы «на грав»)")
            if not folder:
                return
            self.scanner.add_folder(Path(folder))
            refresh()
            messagebox.showinfo(
                "Самообучение",
                "Папка добавлена. Программа сама изучит фото в фоне, найдёт пары «исходник → финал» "
                "и дообучится. Фото никуда не копируются. Новые заказы в этой папке будут изучаться при каждом запуске.",
                parent=dialog,
            )
            self._auto_learn()

        def add_final() -> None:
            if not self.source_path:
                messagebox.showwarning("Нет фото", "Сначала откройте исходное фото, затем укажите ваш готовый файл для станка.", parent=dialog)
                return
            final = filedialog.askopenfilename(parent=dialog, title="Ваш готовый файл для станка", filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")])
            if final:
                self.training.add_pair(self.source_path, Path(final))
                refresh()

        def import_folder() -> None:
            folder = filedialog.askdirectory(parent=dialog, title="Папка с заказами (в каждой подпапке — исходник и файл «на грав»)")
            if not folder:
                return
            added, skipped = self.training.import_folder(Path(folder))
            refresh()
            note = f"Добавлено пар: {added}."
            if skipped:
                note += f"\nПропущено подпапок: {len(skipped)} (не нашёл исходник или файл со словом «грав»/«финал» в названии)."
            messagebox.showinfo("Импорт", note, parent=dialog)

        def export_report() -> None:
            include = messagebox.askyesno(
                "Отчёт для анализа",
                "Добавить в отчёт уменьшенные копии фото из примеров (640 px)?\n\n"
                "«Да» — анализ будет точнее, но в архиве будут фото клиентов.\n«Нет» — только цифры и журналы.",
                parent=dialog,
            )
            archive = self.training.export_report(_desktop_dir(), include)
            messagebox.showinfo("Отчёт готов", f"Архив сохранён:\n{archive}\n\nОтправьте его разработчику.", parent=dialog)
            subprocess.Popen(["explorer", "/select,", str(archive)])

        buttons = [
            ("Папка для самообучения…", add_scan_folder),
            ("Мой финал для открытого фото…", add_final),
            ("Импорт папки с заказами…", import_folder),
            ("Открыть папку примеров", lambda: os.startfile(self.training.examples_dir)),
            ("Обучить на примерах", lambda: (self._start_training(), dialog.destroy())),
            ("Отчёт для анализа (zip на рабочий стол)", export_report),
        ]
        for text, command in buttons:
            ttk.Button(dialog, text=text, style="Primary.TButton" if text.startswith("Обучить") else "Secondary.TButton", command=command).pack(fill="x", pady=2)

    def _start_reports(self) -> None:
        # The operator agreed to reports in advance (see ИНСТРУКЦИЯ_ОПЕРАТОРУ.md), so no pop-up here.
        self.reports.upload_in_background()
        self._auto_learn()

    def _auto_learn(self) -> None:
        """Rescan the self-learning folders in the background; retrain when enough new pairs appear."""
        if self.scan_running or not self.scanner.folders():
            return
        self.scan_running = True

        def run() -> None:
            try:
                result = self.scanner.scan(lambda text: self.message_queue.put(("training_progress", text)))
                self.message_queue.put(("scan_done", result))
            except Exception as error:
                self.message_queue.put(("scan_done", error))

        threading.Thread(target=run, daemon=True).start()

    def _after_scan(self, result: object) -> None:
        self.scan_running = False
        if isinstance(result, Exception):
            self.status.set(f"Не удалось изучить папки: {result}")
            return
        files, found = result  # type: ignore[misc]
        total = len(self.training.all_pairs())
        trained = self._autolearn_state().get("trained_pairs", 0)
        if total >= MIN_PAIRS and total - trained >= AUTO_RETRAIN_NEW_PAIRS:
            self.status.set(f"Изучено фото: {files}, пар найдено: {found}. Дообучаюсь в фоне…")
            self._start_training(auto=True)
        else:
            self.status.set(f"Изучено фото: {files}, пар найдено: {found}. Новых пар для дообучения мало, жду новых заказов.")

    def _autolearn_state(self) -> dict:
        try:
            return json.loads(AUTOLEARN_STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _start_training(self, auto: bool = False) -> None:
        if self.training_running:
            return
        self.training_running = True
        self.training_auto = auto
        self.status.set("Обучение запущено — первый раз около 20 секунд на пример, дальше быстрее…")

        def run() -> None:
            try:
                result = self.training.train(lambda text: self.message_queue.put(("training_progress", text)))
                self.message_queue.put(("training_done", result))
            except Exception as error:
                self.message_queue.put(("training_error", error))

        threading.Thread(target=run, daemon=True).start()

    def _open_output_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(OUTPUT_DIR)])

    def _open_full_size(self, kind: str) -> None:
        image_path = (self.restored_path or self.source_path) if kind == "original" else self.result_path
        if image_path and image_path.is_file():
            os.startfile(image_path)

    def _launch_graver(self) -> None:
        executable = Path(self.graver_exe.get().strip())
        if not self.result_path:
            messagebox.showwarning("Нет результата", "Сначала обработайте фотографию.")
            return
        if not executable.is_file():
            messagebox.showwarning("Не найден Graver", "Укажите путь к файлу graver5_x64.exe.")
            return
        self.open_in_graver_button.state(["disabled"])
        self.status.set("Запускаю Graver и передаю изображение…")
        # Window automation waits up to ~35 s for Graver's dialogs: keep it off the GUI thread.
        threading.Thread(target=self._run_launch_graver, args=(executable, self.result_path), daemon=True).start()

    def _run_launch_graver(self, executable: Path, image_path: Path) -> None:
        try:
            self.message_queue.put(("graver_launched", launch_graver_with_image(executable, image_path)))
        except (OSError, RuntimeError) as error:
            self.message_queue.put(("graver_error", error))

    def _update_result_actions_state(self) -> None:
        state = ["!disabled"] if self.result_path and self.result_path.is_file() else ["disabled"]
        self.save_button.state(state)
        self.open_in_graver_button.state(state)

    def _save_result_as(self) -> None:
        if not self.result_path or not self.result_path.is_file():
            messagebox.showwarning("Нет результата", "Сначала обработайте фотографию.")
            return
        selected = filedialog.asksaveasfilename(
            title="Сохранить файл для Graver",
            initialfile=self.result_path.name,
            defaultextension=self.result_path.suffix,
            filetypes=[("PNG", "*.png"), ("Bitmap", "*.bmp"), ("Все файлы", "*.*")],
        )
        if not selected:
            return
        destination = Path(selected)
        try:
            with Image.open(self.result_path) as image:
                image.convert("L").save(destination, dpi=TARGET_DPI)
        except OSError as error:
            messagebox.showerror("Не удалось сохранить", str(error))
            return
        self.status.set(f"Сохранено: {destination}")

    def _on_close(self) -> None:
        self._save_settings()
        self.destroy()

    def _save_settings(self) -> None:
        save_settings(
            SETTINGS_PATH,
            AppSettings(
                contrast=self.contrast.get(),
                detail=self.detail.get(),
                shadows=self.shadows.get(),
                black_background=self.black_background.get(),
                export_bmp=self.export_bmp.get(),
                portrait_mode=self.portrait_mode.get(),
                graver_exe=self.graver_exe.get().strip(),
                auto_mode=True,
                stone_profile=self._stone_key(),
                enhance_4k=self.enhance_4k.get(),
                restoration_mode=RESTORATION_LABELS.get(self.restoration_mode.get(), "natural"),
                openai_api_key=self.openai_api_key.get().strip(),
                openai_model=self.openai_model.get().strip() or "gpt-image-1",
            ),
        )


if __name__ == "__main__":
    AIGraverApp().mainloop()
