"""Silent automatic upload of analysis reports to the developer's server."""
from __future__ import annotations

import json
import platform
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from report_config import REPORT_URL, get_upload_token
from training import TrainingStore
from version import APP_VERSION

UPLOAD_EVERY_PROCESSINGS = 10
MAX_EXAMPLES_PER_UPLOAD = 150
REQUEST_TIMEOUT = 180

NOTICE_TEXT = (
    "Для улучшения качества AI Graver автоматически отправляет разработчику отчёты: "
    "журнал обработок (камень, настройки, результат коррекции), статистику обучения и "
    "уменьшенные копии (640 px) фото из раздела «Обучение».\n\n"
    "Ключи, пароли и настройки программы не отправляются."
)


class ReportUploader:
    def __init__(self, store: TrainingStore) -> None:
        self.store = store
        self.state_path = store.data_dir / "upload_state.json"
        self.notice_path = store.data_dir / "reports_notice_shown"
        self._lock = threading.Lock()

    # ---------- identity & state ----------
    def install_id(self) -> str:
        path = self.store.data_dir / "install_id.txt"
        try:
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass
        value = str(uuid.uuid4())
        path.write_text(value, encoding="utf-8")
        return value

    def needs_notice(self) -> bool:
        return not self.notice_path.exists()

    def mark_notice_shown(self) -> None:
        self.notice_path.write_text(time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")

    def _state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"time": 0, "processed": 0, "examples": []}

    def _processed_count(self) -> int:
        try:
            with (self.store.log_dir / "processing.jsonl").open(encoding="utf-8") as file:
                return sum(1 for _ in file)
        except OSError:
            return 0

    def _training_time(self) -> float:
        try:
            return (self.store.log_dir / "training_report.json").stat().st_mtime
        except OSError:
            return 0.0

    def pending(self) -> bool:
        state = self._state()
        new_processed = self._processed_count() - state.get("processed", 0)
        trained_since = self._training_time() > state.get("time", 0)
        return new_processed >= UPLOAD_EVERY_PROCESSINGS or trained_since

    # ---------- upload ----------
    def upload_in_background(self, force: bool = False) -> None:
        threading.Thread(target=self.upload_if_needed, args=(force,), daemon=True).start()

    def upload_if_needed(self, force: bool = False) -> bool:
        token = get_upload_token()
        if not token or not self._lock.acquire(blocking=False):
            return False
        try:
            if not force and not self.pending():
                return False
            state = self._state()
            processed = self._processed_count()
            included: list[str] = []
            archive = self.store.export_report(
                self.store.cache_dir / "upload",
                include_thumbnails=True,
                skip_examples=set(state.get("examples", [])),
                max_examples=MAX_EXAMPLES_PER_UPLOAD,
                included=included,
            )
            try:
                self._post(archive, token)
            finally:
                archive.unlink(missing_ok=True)
            state = {
                "time": time.time(),
                "processed": processed,
                "examples": sorted(set(state.get("examples", [])) | set(included)),
            }
            self.state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            self._log(f"ok, примеров с фото: {len(included)}")
            return True
        except Exception as error:  # never disturb the operator: retry on the next trigger
            self._log(f"ошибка: {error}")
            return False
        finally:
            self._lock.release()

    def _post(self, archive: Path, token: str) -> None:
        request = urllib.request.Request(
            REPORT_URL,
            data=archive.read_bytes(),
            method="POST",
            headers={
                "Content-Type": "application/zip",
                "X-Upload-Token": token,
                "X-Install-Id": self.install_id(),
                "X-App-Version": APP_VERSION,
                "X-Computer": platform.node()[:80],
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                if response.status != 200:
                    raise RuntimeError(f"сервер ответил {response.status}")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"сервер ответил {error.code}") from error

    def _log(self, text: str) -> None:
        self.store.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.store.log_dir / "upload.log").open("a", encoding="utf-8") as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
