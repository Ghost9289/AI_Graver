from __future__ import annotations

import ctypes
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from ctypes import wintypes

from PIL import Image


user32 = ctypes.WinDLL("user32", use_last_error=True)

WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
BM_CLICK = 0x00F5
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
MK_LBUTTON = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
WM_COMMAND = 0x0111
GRAVER_OPEN_COMMAND = 1002

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
EnumChildProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


@dataclass(frozen=True)
class GraverLaunchResult:
    transfer_path: Path
    verified: bool
    message: str


def _window_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _process_id(hwnd: int) -> int:
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    return int(process_id.value)


def _top_windows(process_id: int) -> list[int]:
    windows: list[int] = []

    @EnumWindowsProc
    def callback(hwnd: int, _lparam: int) -> bool:
        if _process_id(hwnd) == process_id and user32.IsWindowVisible(hwnd):
            windows.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return windows


def _children(hwnd: int) -> list[int]:
    children: list[int] = []

    @EnumChildProc
    def callback(child: int, _lparam: int) -> bool:
        children.append(child)
        return True

    user32.EnumChildWindows(hwnd, callback, 0)
    return children


def _wait_for_window(process_id: int, predicate, timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for hwnd in _top_windows(process_id):
            if predicate(hwnd):
                return hwnd
        time.sleep(0.1)
    return None


def _write_last_file(conf_path: Path, image_path: Path) -> None:
    if not conf_path.exists():
        return
    raw = conf_path.read_bytes()
    has_trailing_nul = raw.endswith(b"\0")
    text = raw.rstrip(b"\0").decode("cp1251", errors="replace")
    lines = text.splitlines()
    replacement = f"LASTFILE={image_path}"
    changed = False
    for index, line in enumerate(lines):
        if line.startswith("LASTFILE="):
            lines[index] = replacement
            changed = True
            break
    if not changed:
        lines.append(replacement)
    updated = ("\r\n".join(lines) + "\r\n").encode("cp1251", errors="replace")
    if has_trailing_nul:
        updated += b"\0"
    conf_path.write_bytes(updated)


def _prepare_transfer_file(image_path: Path) -> Path:
    """Create the 8-bit BMP that Graver 5's native bitmap loader expects."""
    transfer_dir = Path.home() / "AppData" / "Local" / "AI_Graver" / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    transfer_path = transfer_dir / "portrait_for_graver.bmp"
    with Image.open(image_path) as image:
        gray_image = image.convert("L")
        gray_image.save(transfer_path, format="BMP", dpi=(83, 83))
    return transfer_path


def _click_open_toolbar(main_window: int) -> None:
    # Graver 5.22.8 ignores command-line file arguments. Command 1002 is its
    # native File -> Open action and works regardless of Windows DPI scaling.
    user32.PostMessageW(main_window, WM_COMMAND, GRAVER_OPEN_COMMAND, 0)


def _fill_open_dialog(dialog: int, image_path: Path) -> None:
    children = _children(dialog)
    edits = [hwnd for hwnd in children if _class_name(hwnd) == "Edit" and user32.IsWindowVisible(hwnd)]
    if not edits:
        raise RuntimeError("Graver открыл окно выбора файла, но поле имени файла не найдено.")

    filename_edit = edits[-1]
    user32.SetWindowTextW(filename_edit, str(image_path))

    buttons = [hwnd for hwnd in children if _class_name(hwnd) == "Button" and user32.IsWindowVisible(hwnd)]
    open_button = next(
        (hwnd for hwnd in buttons if _window_text(hwnd).strip().lower() in {"открыть", "open", "&open"}),
        None,
    )
    if open_button is None:
        open_button = next((hwnd for hwnd in buttons if user32.GetDlgCtrlID(hwnd) == 1), None)
    if open_button is None:
        raise RuntimeError("В окне Graver не найдена кнопка «Открыть».")

    user32.SetForegroundWindow(dialog)
    user32.SetFocus(filename_edit)
    time.sleep(0.1)
    user32.keybd_event(0x0D, 0, 0, 0)
    user32.keybd_event(0x0D, 0, 0x0002, 0)


def _read_log(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    return log_path.read_bytes().decode("cp1251", errors="replace")


def _wait_for_log_confirmation(log_path: Path, previous_size: int, transfer_path: Path, timeout: float = 12.0) -> bool:
    """Confirm only that Graver accepted the image; never start CNC work."""
    deadline = time.monotonic() + timeout
    path_text = str(transfer_path).lower()
    while time.monotonic() < deadline:
        log_text = _read_log(log_path)
        if len(log_text) > previous_size:
            new_text = log_text[previous_size:].lower()
            if path_text in new_text or "bm newsize" in new_text:
                return True
        time.sleep(0.25)
    return False


def launch_graver_with_image(executable: str | Path, image_path: str | Path) -> GraverLaunchResult:
    executable = Path(executable).expanduser().resolve()
    image_path = Path(image_path).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Не найден Graver: {executable}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Не найден подготовленный файл: {image_path}")

    transfer_path = _prepare_transfer_file(image_path)
    log_path = executable.parent / "graver.log"
    previous_log_size = len(_read_log(log_path))
    _write_last_file(executable.parent / "conf.ini", transfer_path)

    process = subprocess.Popen(
        [str(executable)],
        cwd=str(executable.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    main_window = _wait_for_window(
        process.pid,
        lambda hwnd: _class_name(hwnd) != "#32770" and "грав" in _window_text(hwnd).lower(),
        timeout=15.0,
    )
    if main_window is None:
        raise RuntimeError("Graver запущен, но его главное окно не найдено.")

    user32.ShowWindow(main_window, 9)
    user32.SetForegroundWindow(main_window)
    time.sleep(0.7)
    _click_open_toolbar(main_window)
    open_dialog = _wait_for_window(process.pid, lambda hwnd: _class_name(hwnd) == "#32770", timeout=8.0)
    if open_dialog is None:
        raise RuntimeError("Graver открылся, но не показал окно загрузки изображения.")
    _fill_open_dialog(open_dialog, transfer_path)

    verified = _wait_for_log_confirmation(log_path, previous_log_size, transfer_path)
    if verified:
        message = "Graver принял BMP и начал подготовку. Запуск гравировки остаётся ручным."
    else:
        message = "BMP передан в Graver. Проверьте изображение в окне Graver перед запуском гравировки."
    return GraverLaunchResult(transfer_path=transfer_path, verified=verified, message=message)
