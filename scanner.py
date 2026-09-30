"""Finds training pairs "source photo -> final file for the machine" in the operator's own folders.

Nothing is copied: the index keeps only paths. Finals are recognised by name ("грав", "финал", …)
or by look (grey portrait on a black background); retouch drafts are skipped; each final is
matched to its source by the surname in the file names, or else by the similarity of the face.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from learning import IMAGE_SUFFIXES

FINAL_WORDS = ("грав", "финал", "final", "станок", "готов")
DRAFT_WORDS = ("ретуш", "retouch", "согласов")
SOURCE_WORDS = ("исход", "source", "ориг", "original", "на ретушь")
STOP_WORDS = {"грав", "светлее", "темнее", "финал", "final", "станок", "готово", "готов", "ретушь", "retouch",
              "согласование", "копия", "лицо", "муж", "жена", "img", "image", "photo", "фото", "на", "для", "new"}
FACE_MATCH_THRESHOLD = 0.55
ANALYSIS_SIDE = 900


@dataclass
class ScannedFile:
    path: str
    mtime: float
    kind: str  # "final" | "draft" | "source" | "skip"
    face: list[float] | None  # face descriptor, when a face was found


@dataclass
class ScannedPair:
    source: str
    final: str
    reason: str  # "name" | "face"
    score: float


class PhotoScanner:
    def __init__(self, data_dir: Path) -> None:
        self.index_path = data_dir / "scan_index.json"
        self.folders_path = data_dir / "scan_folders.json"

    # ---------- configuration ----------
    def folders(self) -> list[Path]:
        try:
            return [Path(item) for item in json.loads(self.folders_path.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return []

    def add_folder(self, folder: Path) -> None:
        folders = [str(item) for item in self.folders()]
        if str(folder) not in folders:
            folders.append(str(folder))
        self.folders_path.write_text(json.dumps(folders, ensure_ascii=False), encoding="utf-8")

    def remove_folder(self, folder: Path) -> None:
        folders = [str(item) for item in self.folders() if item != folder]
        self.folders_path.write_text(json.dumps(folders, ensure_ascii=False), encoding="utf-8")

    # ---------- index ----------
    def _load_index(self) -> dict:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"files": {}, "pairs": []}

    def pairs(self) -> list[tuple[Path, Path]]:
        index = self._load_index()
        result = []
        for pair in index.get("pairs", []):
            source, final = Path(pair["source"]), Path(pair["final"])
            if source.is_file() and final.is_file():
                result.append((source, final))
        return result

    def scan(self, progress: Callable[[str], None] | None = None) -> tuple[int, int]:
        """Rescan all folders; only new or changed files are analysed. Returns (files, pairs)."""
        index = self._load_index()
        known: dict[str, dict] = index.get("files", {})
        files: dict[str, ScannedFile] = {}
        paths = [path for folder in self.folders() if folder.is_dir() for path in _images(folder)]
        for number, path in enumerate(paths, start=1):
            key = str(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            cached = known.get(key)
            if cached and cached.get("mtime") == mtime:
                files[key] = ScannedFile(**cached)
                continue
            if progress and number % 10 == 0:
                progress(f"Изучаю фото: {number} из {len(paths)}")
            files[key] = _analyse(path, mtime)
        pairs = _match(list(files.values()))
        index = {
            "scanned": time.strftime("%Y-%m-%d %H:%M"),
            "files": {key: asdict(value) for key, value in files.items()},
            "pairs": [asdict(pair) for pair in pairs],
        }
        self.index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
        return len(files), len(pairs)


def _images(folder: Path):
    for path in folder.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def _analyse(path: Path, mtime: float) -> ScannedFile:
    name = path.stem.lower()
    transparent_share = 0.0
    try:
        with Image.open(path) as opened:
            opened.draft("RGB", (ANALYSIS_SIDE * 2, ANALYSIS_SIDE * 2))  # fast JPEG decode of huge photos
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((ANALYSIS_SIDE, ANALYSIS_SIDE))
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGBA")
                transparent_share = float((np.asarray(image)[..., 3] < 20).mean())
                image = Image.alpha_composite(Image.new("RGBA", image.size, (0, 0, 0, 255)), image)
            image = image.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
        return ScannedFile(str(path), mtime, "skip", None)

    rgb = np.asarray(image, dtype=np.int16)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    colourful = float(np.abs(rgb[..., 0] - rgb[..., 2]).mean()) > 6
    black_share = float((gray < 18).mean())
    white_share = float((gray > 240).mean())

    if any(word in name for word in FINAL_WORDS):
        kind = "final"
    elif any(word in name for word in SOURCE_WORDS):  # "на ретушь" = sent to retouch = the original
        kind = "source"
    elif any(word in name for word in DRAFT_WORDS) or transparent_share > 0.05:
        kind = "draft"  # retouch stage; a cut-out with transparency is never the machine file
    elif not colourful and black_share > 0.35:
        kind = "final"  # grey portrait on black: what goes to the machine
    elif not colourful and white_share > 0.3:
        kind = "draft"  # grey cut-out on white: retouch stage
    else:
        kind = "source"

    face = _face_descriptor(gray)
    return ScannedFile(str(path), mtime, kind, face)


_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _face_descriptor(gray: np.ndarray) -> list[float] | None:
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    crop = gray[y:y + h, x:x + w]
    crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    crop = cv2.equalizeHist(crop).astype(np.float32)
    # gradients describe the face structure and survive retouch / tone changes better than raw pixels
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
    vector = np.concatenate([(crop - crop.mean()).ravel(), gx.ravel(), gy.ravel()])
    vector /= np.linalg.norm(vector) + 1e-6
    return [round(float(value), 4) for value in vector]


def _surnames(path: str) -> set[str]:
    words = re.findall(r"[a-zа-яё]+", Path(path).stem.lower())
    return {word for word in words if len(word) >= 4 and word not in STOP_WORDS}


def _named_final(item: ScannedFile) -> bool:
    return any(word in Path(item.path).stem.lower() for word in FINAL_WORDS)


def _match(files: list[ScannedFile]) -> list[ScannedPair]:
    named = [item for item in files if item.kind == "final" and _named_final(item)]
    guessed = [item for item in files if item.kind == "final" and not _named_final(item)]
    # A look-alike "final" of the same person as a named final is that order's retouch stage.
    guessed = [
        item for item in guessed
        if not any(
            (_surnames(item.path) & _surnames(other.path))
            or (item.face and other.face and float(np.dot(item.face, other.face)) > 0.8)
            for other in named
        )
    ]
    sources = [item for item in files if item.kind == "source"]
    candidates: list[tuple[float, str, ScannedFile, ScannedFile]] = []
    for final in named + guessed:
        final_names = _surnames(final.path)
        for source in sources:
            shared = final_names & _surnames(source.path)
            if shared:
                score, reason = 1.0 + len(shared), "name"
            elif final.face is not None and source.face is not None:
                score, reason = float(np.dot(final.face, source.face)), "face"
                if score < FACE_MATCH_THRESHOLD:
                    continue
                if Path(final.path).parent == Path(source.path).parent:
                    score += 0.05  # same order folder
            else:
                continue
            if final in named:
                score += 0.01  # a file named "…на грав…" wins a tie
            candidates.append((score, reason, final, source))

    # Most certain matches first, so a weak early guess never takes someone else's source.
    pairs: list[ScannedPair] = []
    used_finals: set[str] = set()
    used_sources: set[str] = set()
    for score, reason, final, source in sorted(candidates, key=lambda item: item[0], reverse=True):
        if final.path in used_finals or source.path in used_sources:
            continue
        used_finals.add(final.path)
        used_sources.add(source.path)
        pairs.append(ScannedPair(source.path, final.path, reason, round(score, 3)))
    return pairs
