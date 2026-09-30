"""Collecting operator examples, retraining the learned style, logs and the analysis report."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from learning import IMAGE_SUFFIXES, LearnedModel, _face_grid, _load_gray, apply_model, find_pairs, learn, load_model, save_model
from processor import EngravingOptions, _detect_primary_face, _resource_path, render_preview
from version import APP_VERSION

# Bump when the automatic pipeline changes, so cached automatic results are rebuilt.
PIPELINE_VERSION = "1"
TRAIN_RENDER_SIZE = (1100, 1970)
MIN_PAIRS = 5
FINAL_KEYWORDS = ("грав", "final", "финал", "станок", "готов")
SOURCE_KEYWORDS = ("исход", "source", "ориг", "original")
RETOUCH_KEYWORDS = ("ретуш", "retouch", "согласов")


class TrainingStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.examples_dir = data_dir / "Обучение"
        self.cache_dir = data_dir / "train_cache"
        self.model_path = data_dir / "learned.json"
        self.log_dir = data_dir / "logs"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        # Pairs found by the folder scanner (see scanner.py): used in place, never copied.
        self.extra_pairs: Callable[[], list[tuple[Path, Path]]] | None = None

    # ---------- model ----------
    def load_active_model(self) -> LearnedModel | None:
        return load_model(self.model_path) or load_model(_resource_path("learned_default.json"))

    # ---------- examples ----------
    def add_pair(self, source: Path, final: Path) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = "".join(char if char.isalnum() or char in "-_" else "_" for char in source.stem)[:40] or "portrait"
        folder = self.examples_dir / f"{stamp}_{name}"
        number = 2
        while folder.exists():  # imports add many pairs per second, often with the same file name
            folder = self.examples_dir / f"{stamp}_{name}_{number}"
            number += 1
        folder.mkdir(parents=True)
        shutil.copyfile(source, folder / f"source{source.suffix.lower()}")
        shutil.copyfile(final, folder / f"final{final.suffix.lower()}")
        return folder

    def import_folder(self, root: Path) -> tuple[int, list[str]]:
        """Each subfolder = one order holding the original photo and the file that went to the machine."""
        added, skipped = 0, []
        for order in sorted(item for item in root.iterdir() if item.is_dir()):
            images = [item for item in order.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
            finals = [item for item in images if _has_keyword(item, FINAL_KEYWORDS)]
            sources = [item for item in images if _has_keyword(item, SOURCE_KEYWORDS)]
            if not sources:
                others = [item for item in images if item not in finals and not _has_keyword(item, RETOUCH_KEYWORDS)]
                sources = sorted(others, key=lambda item: item.stat().st_mtime)[:1]
            if len(finals) >= 1 and sources:
                final = max(finals, key=lambda item: item.stat().st_mtime)
                self.add_pair(sources[0], final)
                added += 1
            else:
                skipped.append(order.name)
        return added, skipped

    def pair_count(self) -> int:
        return len(find_pairs(self.examples_dir))

    def all_pairs(self) -> list[tuple[Path, Path]]:
        pairs = find_pairs(self.examples_dir, _resource_path("dataset/pairs"))
        if self.extra_pairs is not None:
            pairs += self.extra_pairs()
        return list(dict.fromkeys(pairs))

    # ---------- training ----------
    def train(self, progress: Callable[[str], None]) -> tuple[LearnedModel, list[dict]]:
        pairs = self.all_pairs()
        if len(pairs) < MIN_PAIRS:
            raise ValueError(f"Для обучения нужно минимум {MIN_PAIRS} пар «исходник → финал для станка», сейчас {len(pairs)}.")
        model = learn(pairs, self._auto_result, _detect_primary_face, progress)
        save_model(model, self.model_path)
        progress("Считаю отчёт по примерам…")
        stats = [self._pair_stats(source, final, model) for source, final in pairs]
        report = {
            "app_version": APP_VERSION,
            "trained": model.updated,
            "model": model.summary(),
            "pairs": stats,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return model, stats

    def _auto_result(self, source: Path) -> Image.Image:
        digest = hashlib.sha1(source.read_bytes()).hexdigest()[:20]
        cached = self.cache_dir / f"{digest}_v{PIPELINE_VERSION}.png"
        if cached.is_file():
            with Image.open(cached) as image:
                return image.convert("L")
        result = render_preview(source, EngravingOptions(enhance_4k=False, use_learned=False), size=TRAIN_RENDER_SIZE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        result.save(cached)
        return result

    def _pair_stats(self, source: Path, final_path: Path, model: LearnedModel) -> dict:
        if self.examples_dir in source.parents:
            example = source.parent.name
        else:  # scanned in the operator's folders: file names often hold surnames, send only a hash
            example = "scan-" + hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
        entry: dict = {"example": example}
        try:
            ours = self._auto_result(source)
            final = _load_gray(final_path)
            face_ours = _detect_primary_face(ours.convert("RGB"))
            face_final = _detect_primary_face(final.convert("RGB"))
            entry["face_found"] = bool(face_ours and face_final)
            if not (face_ours and face_final):
                return entry
            learned = apply_model(ours, model, face_ours)
            entry["difference_before"] = round(_difference(ours, face_ours, final, face_final), 1)
            entry["difference_after"] = round(_difference(learned, face_ours, final, face_final), 1)
            entry["final_face_tone"] = round(_face_median(final, face_final))
            entry["auto_face_tone"] = round(_face_median(ours, face_ours))
        except Exception as error:  # a broken example must not stop the report
            entry["error"] = str(error)
        return entry

    # ---------- logs & report ----------
    def log_processing(self, record: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "app_version": APP_VERSION, **record}
        with (self.log_dir / "processing.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_report(
        self,
        destination: Path,
        include_thumbnails: bool,
        skip_examples: set[str] | None = None,
        max_examples: int | None = None,
        included: list[str] | None = None,
    ) -> Path:
        """Zip for analysis. skip_examples / max_examples limit which example photos go in;
        names of the examples actually added are appended to `included`."""
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / f"AI_Graver_отчёт_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(
                "system.json",
                json.dumps({"app_version": APP_VERSION, "windows": platform.platform(), "examples": self.pair_count()}, ensure_ascii=False),
            )
            model = self.load_active_model()
            if model is not None:
                bundle.writestr("learned.json", json.dumps(asdict(model), ensure_ascii=False))
            for name in ("processing.jsonl", "training_report.json", "ai_errors.log", "upload.log"):
                path = self.log_dir / name
                if path.is_file():
                    bundle.write(path, f"logs/{name}")
            for name in ("stones.json", "ai_template.txt"):  # settings.ini holds the API key: never exported
                path = self.data_dir / name
                if path.is_file():
                    bundle.write(path, name)
            if include_thumbnails:
                added = 0
                for source, final in find_pairs(self.examples_dir):
                    if skip_examples and source.parent.name in skip_examples:
                        continue
                    if max_examples is not None and added >= max_examples:
                        break
                    added += 1
                    if included is not None:
                        included.append(source.parent.name)
                    for kind, path in (("source", source), ("final", final)):
                        with Image.open(path) as image:
                            thumbnail = ImageOps.contain(ImageOps.exif_transpose(image).convert("RGB"), (640, 640))
                        buffer_path = self.cache_dir / "_thumb.jpg"
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        thumbnail.save(buffer_path, quality=82)
                        bundle.write(buffer_path, f"examples/{source.parent.name}/{kind}.jpg")
        return archive


def _has_keyword(path: Path, keywords: tuple[str, ...]) -> bool:
    name = path.stem.lower()
    return any(keyword in name for keyword in keywords)


def _difference(image: Image.Image, face: tuple[int, int, int, int], final: Image.Image, final_face: tuple[int, int, int, int]) -> float:
    grid_a, cover_a = _face_grid(np.asarray(image, dtype=np.float32), face)
    grid_b, cover_b = _face_grid(np.asarray(final, dtype=np.float32), final_face)
    valid = (cover_a > 0.6) & (cover_b > 0.6)
    return float(np.abs(grid_a[valid] - grid_b[valid]).mean()) if valid.any() else float("nan")


def _face_median(image: Image.Image, face: tuple[int, int, int, int]) -> float:
    values = np.asarray(image, dtype=np.float32)
    x, y, w, h = face
    region = values[y + h // 6:y + h * 5 // 6, x + w // 6:x + w * 5 // 6]
    return float(np.median(region[region > 12])) if (region > 12).any() else 0.0
