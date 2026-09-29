from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from version import APP_VERSION, UPDATE_REPOSITORY


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    installer_url: str


@dataclass(frozen=True)
class RemoteStatus:
    enabled: bool
    message: str


def check_remote_access(timeout: int = 5) -> RemoteStatus | None:
    """Read status.json from the update repository so access can be revoked remotely.

    Returns None when the check could not be completed (offline, repo unreachable,
    file missing, ...) so the caller fails OPEN and lets the program run rather than
    bricking it for a user with no internet connection at that moment.
    """
    request = Request(
        f"https://api.github.com/repos/{UPDATE_REPOSITORY}/contents/status.json",
        headers={"Accept": "application/vnd.github.raw+json", "User-Agent": "AI-Graver-Updater"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    message = str(
        payload.get("message")
        or "Программа временно отключена разработчиком. Свяжитесь с автором для повторного включения."
    )
    return RemoteStatus(enabled=bool(payload.get("enabled", True)), message=message)


def check_for_update(timeout: int = 6) -> UpdateInfo | None:
    request = Request(
        f"https://api.github.com/repos/{UPDATE_REPOSITORY}/releases/latest",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "AI-Graver-Updater"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            release = json.load(response)
    except OSError as error:
        raise UpdateError("Не удалось проверить обновления.") from error

    version = str(release.get("tag_name", "")).lstrip("vV")
    if not version or not _is_newer(version, APP_VERSION):
        return None
    asset = next(
        (item for item in release.get("assets", []) if str(item.get("name", "")).lower().endswith(".exe")),
        None,
    )
    if not asset or not asset.get("browser_download_url"):
        return None
    return UpdateInfo(version=version, installer_url=str(asset["browser_download_url"]))


def download_installer(update: UpdateInfo) -> Path:
    folder = Path(tempfile.gettempdir()) / "AI Graver Updates"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"AI_Graver_Setup_v{update.version}.exe"
    request = Request(update.installer_url, headers={"User-Agent": "AI-Graver-Updater"})
    try:
        with urlopen(request, timeout=30) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except OSError as error:
        destination.unlink(missing_ok=True)
        raise UpdateError("Не удалось скачать обновление.") from error
    return destination


def start_silent_update(installer: Path) -> None:
    if not installer.is_file() or installer.stat().st_size < 1024 * 1024:
        raise UpdateError("Загруженный файл обновления повреждён.")
    log_path = installer.with_suffix(".log")
    try:
        subprocess.Popen(
            [
                str(installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/CLOSEAPPLICATIONS",
                "/RESTARTAPPLICATIONS",
                f"/LOG={log_path}",
            ],
            close_fds=True,
        )
    except OSError as error:
        raise UpdateError("Не удалось запустить установку обновления.") from error


def _is_newer(candidate: str, current: str) -> bool:
    return _version_tuple(candidate) > _version_tuple(current)


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(number) for number in numbers[:4]) or (0,)
