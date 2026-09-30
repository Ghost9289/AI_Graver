"""Learns the operator's retouching style from pairs "source photo -> final file that went to Graver".

Two things are learned:
- a tone curve (how the operator lifts whites and deepens blacks overall);
- a face-aligned gain map: where on the portrait (forehead, cheeks, eye sockets, hair,
  shoulders) the operator adds white or black compared with the automatic result.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

GRID = (24, 36)  # cells across, cells down
# Region around the face used for the map, in face widths / heights from the face box corner.
REGION = (-1.3, -0.9, 2.3, 4.0)
MAX_LOG_GAIN = 0.35
SOURCE_NAMES = ("01_source", "source", "исходник")
FINAL_NAMES = ("03_final", "final", "финал", "готово")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class LearnedModel:
    pairs: int
    face_target: float
    white_level: float
    black_level: float
    curve: list[float]
    gain_map: list[list[float]]
    updated: str

    def summary(self) -> str:
        return (
            f"Обучено на {self.pairs} примерах: тон лица {self.face_target:.0f}, "
            f"белое до {self.white_level:.0f}, чёрное от {self.black_level:.0f}"
        )


def load_model(path: Path) -> LearnedModel | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return LearnedModel(**data)
    except (OSError, ValueError, TypeError):
        return None


def save_model(model: LearnedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), ensure_ascii=False), encoding="utf-8")


def find_pairs(*folders: Path) -> list[tuple[Path, Path]]:
    """Every subfolder holding a source photo and the operator's final file is one example."""
    pairs: list[tuple[Path, Path]] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for example in sorted(item for item in folder.iterdir() if item.is_dir()):
            source = _find_named(example, SOURCE_NAMES)
            final = _find_named(example, FINAL_NAMES)
            if source and final:
                pairs.append((source, final))
    return pairs


def _find_named(folder: Path, names: tuple[str, ...]) -> Path | None:
    for item in sorted(folder.iterdir()):
        if item.suffix.lower() in IMAGE_SUFFIXES and item.stem.lower() in names:
            return item
    return None


def learn(
    pairs: list[tuple[Path, Path]],
    auto_process: Callable[[Path], Image.Image],
    detect_face: Callable[[Image.Image], tuple[int, int, int, int] | None],
    progress: Callable[[str], None] | None = None,
) -> LearnedModel:
    luts: list[np.ndarray] = []
    log_gains: list[np.ndarray] = []
    face_levels: list[float] = []
    white_levels: list[float] = []
    black_levels: list[float] = []
    seen: set[bytes] = set()

    for index, (source, final_path) in enumerate(pairs, start=1):
        if progress:
            progress(f"Обучение: пример {index} из {len(pairs)} — {source.parent.name}")
        final = _load_gray(final_path)
        fingerprint = np.asarray(final.resize((32, 32))).tobytes()
        if final.width > final.height or fingerprint in seen:
            continue  # double portraits / duplicates are not single-face examples
        seen.add(fingerprint)
        ours = auto_process(source)
        face_ours = detect_face(ours.convert("RGB"))
        face_final = detect_face(final.convert("RGB"))
        if face_ours is None or face_final is None:
            continue

        ours_values = np.asarray(ours, dtype=np.float32)
        final_values = np.asarray(final, dtype=np.float32)
        # Only the portrait around the face: finals also carry names, dates and epitaphs in white.
        ours_subject = _region_pixels(ours_values, face_ours)
        final_subject = _region_pixels(final_values, face_final)
        if ours_subject.size < 1000 or final_subject.size < 1000:
            continue

        quantiles = np.linspace(0, 100, 33)
        lut = np.interp(np.arange(256), np.percentile(ours_subject, quantiles), np.percentile(final_subject, quantiles))
        luts.append(lut)

        toned = np.where(ours_values > 12, lut[np.clip(ours_values, 0, 255).astype(np.uint8)], ours_values)
        grid_ours, cover_ours = _face_grid(toned, face_ours)
        grid_final, cover_final = _face_grid(final_values, face_final)
        valid = (cover_ours > 0.6) & (cover_final > 0.6)
        gain = np.full(grid_ours.shape, np.nan, dtype=np.float32)
        gain[valid] = np.log((grid_final[valid] + 10.0) / (grid_ours[valid] + 10.0))
        log_gains.append(gain)

        x, y, w, h = face_final
        face = final_values[y + h // 6:y + h * 5 // 6, x + w // 6:x + w * 5 // 6]
        face_levels.append(float(np.median(face[face > 12])))
        white_levels.append(float(np.percentile(final_subject, 99.5)))
        black_levels.append(float(np.percentile(final_subject, 1.0)))

    if not luts:
        raise ValueError("Не нашлось ни одного подходящего примера (нужно лицо на исходнике и на финале).")

    stack = np.stack(log_gains)
    counted = np.sum(~np.isnan(stack), axis=0)
    enough = counted >= max(2, len(log_gains) // 3)
    with np.errstate(all="ignore"):
        gain_map = np.where(enough, np.nanmedian(np.where(enough, stack, 0.0), axis=0), 0.0)
    gain_map = cv2.GaussianBlur(np.nan_to_num(gain_map).astype(np.float32), (0, 0), 1.2)
    gain_map = np.clip(gain_map, -MAX_LOG_GAIN, MAX_LOG_GAIN)

    return LearnedModel(
        pairs=len(luts),
        face_target=float(np.median(face_levels)),
        white_level=float(np.median(white_levels)),
        black_level=float(np.median(black_levels)),
        curve=[round(float(value), 2) for value in np.median(np.stack(luts), axis=0)],
        gain_map=[[round(float(value), 4) for value in row] for row in gain_map],
        updated=time.strftime("%Y-%m-%d %H:%M"),
    )


def apply_model(gray: Image.Image, model: LearnedModel, face: tuple[int, int, int, int] | None) -> Image.Image:
    """Apply the learned curve and, if a face is found, the learned white/black map."""
    values = np.asarray(gray, dtype=np.float32)
    subject = values > 12
    lut = np.asarray(model.curve, dtype=np.float32)
    result = np.where(subject, lut[np.clip(values, 0, 255).astype(np.uint8)], values)

    if face is not None:
        x, y, w, h = face
        left, top = round(x + REGION[0] * w), round(y + REGION[1] * h)
        right, bottom = round(x + REGION[2] * w), round(y + REGION[3] * h)
        gain_small = np.asarray(model.gain_map, dtype=np.float32)
        gain = cv2.resize(gain_small, (right - left, bottom - top), interpolation=cv2.INTER_CUBIC)
        full = np.zeros(values.shape, dtype=np.float32)
        src_left, src_top = max(0, -left), max(0, -top)
        dst_left, dst_top = max(0, left), max(0, top)
        dst_right, dst_bottom = min(values.shape[1], right), min(values.shape[0], bottom)
        if dst_right > dst_left and dst_bottom > dst_top:
            full[dst_top:dst_bottom, dst_left:dst_right] = gain[
                src_top:src_top + dst_bottom - dst_top, src_left:src_left + dst_right - dst_left
            ]
        full = cv2.GaussianBlur(full, (0, 0), max(2.0, w / 12))
        result = np.where(subject, (result + 10.0) * np.exp(full) - 10.0, result)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")


def render_gain_map(model: LearnedModel, size: tuple[int, int] = (360, 540)) -> Image.Image:
    """Picture of the learned map: red = operator adds white, blue = operator adds black."""
    gain = np.asarray(model.gain_map, dtype=np.float32) / MAX_LOG_GAIN
    gain = cv2.resize(gain, size, interpolation=cv2.INTER_CUBIC)
    rgb = np.full((size[1], size[0], 3), 245.0, dtype=np.float32)
    rgb[..., 1] -= np.abs(gain) * 200
    rgb[..., 2] -= np.clip(gain, 0, 1) * 200
    rgb[..., 0] -= np.clip(-gain, 0, 1) * 200
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    # outline of the face box for orientation
    face_left = round((0 - REGION[0]) / (REGION[2] - REGION[0]) * size[0])
    face_right = round((1 - REGION[0]) / (REGION[2] - REGION[0]) * size[0])
    face_top = round((0 - REGION[1]) / (REGION[3] - REGION[1]) * size[1])
    face_bottom = round((1 - REGION[1]) / (REGION[3] - REGION[1]) * size[1])
    array = np.asarray(image).copy()
    cv2.rectangle(array, (face_left, face_top), (face_right, face_bottom), (60, 60, 60), 1)
    return Image.fromarray(array)


def _face_grid(values: np.ndarray, face: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x, y, w, h = face
    left, top = round(x + REGION[0] * w), round(y + REGION[1] * h)
    right, bottom = round(x + REGION[2] * w), round(y + REGION[3] * h)
    canvas = np.zeros((bottom - top, right - left), dtype=np.float32)
    src = values[max(0, top):min(values.shape[0], bottom), max(0, left):min(values.shape[1], right)]
    canvas[max(0, -top):max(0, -top) + src.shape[0], max(0, -left):max(0, -left) + src.shape[1]] = src
    cover = (canvas > 12).astype(np.float32)
    grid = cv2.resize(canvas, GRID, interpolation=cv2.INTER_AREA)
    cover_grid = cv2.resize(cover, GRID, interpolation=cv2.INTER_AREA)
    subject_sum = cv2.resize(canvas * cover, GRID, interpolation=cv2.INTER_AREA)
    mean_subject = np.where(cover_grid > 0, subject_sum / np.maximum(cover_grid, 1e-6), grid)
    return mean_subject, cover_grid


def _region_pixels(values: np.ndarray, face: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = face
    top, bottom = max(0, round(y + REGION[1] * h)), min(values.shape[0], round(y + REGION[3] * h))
    left, right = max(0, round(x + REGION[0] * w)), min(values.shape[1], round(x + REGION[2] * w))
    region = values[top:bottom, left:right]
    return region[region > 12]


def _load_gray(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (0, 0, 0, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("L")
