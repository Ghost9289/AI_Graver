from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


TARGET_SIZE = (3084, 5526)
TARGET_DPI = (83, 83)
FOUR_K_PORTRAIT = (2160, 3840)
_BACKGROUND_MASK_CACHE: dict[tuple[str, int, str], Image.Image] = {}
_RESTORED_IMAGE_CACHE: dict[tuple[str, int, str], Image.Image] = {}
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_SEGMENTER = None
_SEGMENTER_LOCK = threading.Lock()
_FACE_RESTORER = None
_FACE_RESTORER_SIZE = 512
_FACE_RESTORER_KIND = "gfpgan"
_FACE_RESTORER_LOCK = threading.Lock()
_SUPER_RESOLVER = None
_SUPER_RESOLVER_LOCK = threading.Lock()


class ProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class EngravingOptions:
    contrast: int = 26
    detail: int = 46
    shadows: int = 18
    black_background: bool = True
    export_bmp: bool = True
    portrait_mode: str = "chest"
    stone_profile: str = "black_granite"
    enhance_4k: bool = True
    restoration_mode: str = "natural"


def process_portrait(source: Path, output_dir: Path, options: EngravingOptions) -> Path:
    """Create a clean 8-bit tonal portrait suitable for Graver's own dithering step."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(source.stem)
    restored_output = output_dir / f"{stem}_restored_4k.png" if options.enhance_4k else None
    portrait = _build_processed_image(source, options, TARGET_SIZE, restored_output)
    output_path = output_dir / f"{stem}_graver_ready.png"
    portrait.save(output_path, format="PNG", dpi=TARGET_DPI, optimize=True)
    if options.export_bmp:
        portrait.save(output_dir / f"{stem}_graver_ready.bmp", format="BMP", dpi=TARGET_DPI)
    return output_path


def render_preview(source: Path, options: EngravingOptions, size: tuple[int, int] = (390, 700)) -> Image.Image:
    """Create a fast, unsaved preview using exactly the same tonal controls as export."""
    return _build_processed_image(source, options, size)


def _build_processed_image(
    source: Path,
    options: EngravingOptions,
    target_size: tuple[int, int],
    restored_output: Path | None = None,
) -> Image.Image:
    try:
        with Image.open(source) as input_image:
            image = ImageOps.exif_transpose(input_image).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ProcessingError(f"Не удалось открыть изображение: {source}") from error

    image = _crop_for_composition(image, options.portrait_mode, TARGET_SIZE)
    face_focus = _face_focus_score(image)
    face_resolution = _face_resolution(image)
    background_mask = _cached_background_mask(source, image, options.portrait_mode) if options.black_background else None
    if options.enhance_4k:
        image = _cached_face_restoration(source, image, options.portrait_mode)
        image = _enhance_source_resolution(image, target_size == TARGET_SIZE, options.restoration_mode)

    color_portrait = _fit_to_color_canvas(image, target_size)
    if background_mask is not None:
        color_portrait.paste((0, 0, 0), mask=_fit_to_portrait_canvas(background_mask, target_size))
    if restored_output is not None:
        color_portrait.save(restored_output, format="PNG", dpi=TARGET_DPI, compress_level=2)

    gray = ImageOps.grayscale(color_portrait)
    if not options.enhance_4k:
        gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = _auto_levels(gray)
    contrast, detail, shadows = _stone_adjustments(options)
    if options.restoration_mode == "natural":
        contrast, detail, shadows = _adaptive_tone(
            contrast,
            detail,
            shadows,
            face_focus,
            face_resolution,
        )
    gray = _preserve_midtones(gray, shadows)
    gray = ImageEnhance.Contrast(gray).enhance(1 + contrast / 180)
    gray = _detail_enhancement(gray, detail)
    if options.stone_profile == "marble":
        gray = gray.filter(ImageFilter.GaussianBlur(radius=0.35))

    return gray


def _cached_background_mask(source: Path, image: Image.Image, portrait_mode: str) -> Image.Image:
    try:
        modified = source.stat().st_mtime_ns
    except OSError:
        modified = 0
    key = (str(source.resolve()), modified, portrait_mode)
    cached = _BACKGROUND_MASK_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    mask = _build_background_mask(image)
    if len(_BACKGROUND_MASK_CACHE) >= 12:
        _BACKGROUND_MASK_CACHE.pop(next(iter(_BACKGROUND_MASK_CACHE)))
    _BACKGROUND_MASK_CACHE[key] = mask.copy()
    return mask


def _auto_levels(image: Image.Image) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(values, (0.5, 99.5))
    if high - low < 18:
        return image.copy()
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    protected = 8.0 + normalized * 232.0
    return Image.fromarray(protected.astype(np.uint8), mode="L")


def _preserve_midtones(image: Image.Image, shadow_strength: int) -> Image.Image:
    gamma = min(1.2, 1.0 + shadow_strength / 500)
    lut = [min(255, round(255 * ((value / 255) ** gamma))) for value in range(256)]
    return image.point(lut)


def _detail_enhancement(image: Image.Image, detail_strength: int) -> Image.Image:
    """Reveal facial relief without turning smooth skin into harsh noise."""
    strength = min(100, max(0, detail_strength)) / 100.0
    values = np.asarray(image, dtype=np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=1.2 + strength * 1.4,
        tileGridSize=(12, 12),
    )
    local = clahe.apply(values)
    broad = cv2.GaussianBlur(local, (0, 0), 1.65)
    fine = cv2.GaussianBlur(local, (0, 0), 0.55)
    medium_detail = local.astype(np.float32) - broad.astype(np.float32)
    micro_detail = local.astype(np.float32) - fine.astype(np.float32)
    enhanced = np.clip(
        values.astype(np.float32) * 0.58
        + local.astype(np.float32) * 0.42
        + medium_detail * (0.42 + strength * 0.82)
        + micro_detail * (0.12 + strength * 0.26),
        0,
        255,
    ).astype(np.uint8)
    result = Image.fromarray(enhanced, mode="L")
    return result.filter(
        ImageFilter.UnsharpMask(
            radius=0.72,
            percent=round(48 + strength * 82),
            threshold=1,
        )
    )


def _adaptive_tone(
    contrast: int,
    detail: int,
    shadows: int,
    face_focus: float | None,
    face_resolution: int | None,
) -> tuple[int, int, int]:
    """Keep a clean scanned portrait clean instead of engraving its paper grain."""
    if face_resolution is not None and face_resolution >= 340:
        return min(contrast, 14), min(detail, 8), min(shadows, 12)
    if face_resolution is not None and face_resolution >= 260:
        return min(contrast, 18), min(detail, 18), min(shadows, 14)
    if face_focus is not None and face_focus >= 26:
        return min(contrast, 16), min(detail, 12), min(shadows, 12)
    if face_focus is not None and face_focus >= 16:
        return min(contrast, 20), min(detail, 24), min(shadows, 15)
    return contrast, detail, shadows


def _enhance_source_resolution(image: Image.Image, full_resolution: bool, restoration_mode: str = "natural") -> Image.Image:
    target_width, target_height = FOUR_K_PORTRAIT if full_resolution else (900, 1600)
    if restoration_mode in {"strong", "old_photo"}:
        try:
            return _super_resolve_portrait(image, (target_width, target_height))
        except (FileNotFoundError, AttributeError, cv2.error):
            pass  # fsrcnn model or opencv-contrib module unavailable; fall back to Lanczos below
    scale = max(target_width / image.width, target_height / image.height, 1.0)
    if scale <= 1.0:
        return image.filter(ImageFilter.UnsharpMask(radius=0.7, percent=30, threshold=4))
    enlarged = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    return enlarged.filter(ImageFilter.UnsharpMask(radius=0.9, percent=48, threshold=4))


def _get_super_resolver():
    global _SUPER_RESOLVER
    if _SUPER_RESOLVER is not None:
        return _SUPER_RESOLVER
    model_path = _resource_path("models/FSRCNN_x4.pb")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    resolver = cv2.dnn_superres.DnnSuperResImpl_create()
    resolver.readModel(str(model_path))
    resolver.setModel("fsrcnn", 4)
    _SUPER_RESOLVER = resolver
    return _SUPER_RESOLVER


def _super_resolve_portrait(image: Image.Image, target_size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    input_size = ((target_size[0] + 3) // 4, (target_size[1] + 3) // 4)
    prepared = image.resize(input_size, Image.Resampling.LANCZOS)
    bgr = cv2.cvtColor(np.asarray(prepared), cv2.COLOR_RGB2BGR)
    with _SUPER_RESOLVER_LOCK:
        restored = _get_super_resolver().upsample(bgr)
    rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb, mode="RGB").filter(ImageFilter.UnsharpMask(radius=0.65, percent=42, threshold=3))


def _cached_face_restoration(source: Path, image: Image.Image, portrait_mode: str) -> Image.Image:
    try:
        modified = source.stat().st_mtime_ns
    except OSError:
        modified = 0
    key = (str(source.resolve()), modified, portrait_mode)
    cached = _RESTORED_IMAGE_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    restored = _restore_primary_face(image)
    if len(_RESTORED_IMAGE_CACHE) >= 8:
        _RESTORED_IMAGE_CACHE.pop(next(iter(_RESTORED_IMAGE_CACHE)))
    _RESTORED_IMAGE_CACHE[key] = restored.copy()
    return restored


def _get_face_restorer():
    global _FACE_RESTORER
    global _FACE_RESTORER_KIND
    global _FACE_RESTORER_SIZE
    if _FACE_RESTORER is not None:
        return _FACE_RESTORER

    import onnxruntime as ort

    model_path = _resource_path("models/GFPGANv1.4.onnx")
    if not model_path.is_file():
        _FACE_RESTORER_KIND = "gpen"
        model_path = _resource_path("models/gpen_bfr_512.onnx")
        if not model_path.is_file():
            model_path = _resource_path("models/gpen_bfr_256.onnx")
            _FACE_RESTORER_SIZE = 256
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    session_options.intra_op_num_threads = 4
    _FACE_RESTORER = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    return _FACE_RESTORER


def _enhance_face_texture(image: Image.Image, strength: float) -> Image.Image:
    rgb = np.asarray(image, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.35 + strength * 0.35, tileGridSize=(8, 8))
    local = clahe.apply(lightness)
    broad = cv2.GaussianBlur(local, (0, 0), 2.0)
    micro = cv2.GaussianBlur(local, (0, 0), 0.75)
    broad_detail = local.astype(np.float32) - broad.astype(np.float32)
    micro_detail = local.astype(np.float32) - micro.astype(np.float32)
    enhanced = np.clip(
        local.astype(np.float32) + broad_detail * (0.5 + strength * 0.35) + micro_detail * 0.35,
        0,
        255,
    ).astype(np.uint8)
    restored_lab = cv2.merge((enhanced, channel_a, channel_b))
    restored_rgb = cv2.cvtColor(restored_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(restored_rgb, mode="RGB")


def _restore_primary_face(image: Image.Image) -> Image.Image:
    face = _detect_primary_face(image)
    if face is None:
        return image

    face_x, face_y, face_width, face_height = face
    crop_size = min(image.width, image.height, round(max(face_width, face_height) * 1.35))
    center_x = face_x + face_width // 2
    center_y = face_y + face_height // 2
    left = min(max(0, center_x - crop_size // 2), image.width - crop_size)
    top = min(max(0, center_y - crop_size // 2), image.height - crop_size)
    crop = image.crop((left, top, left + crop_size, top + crop_size))
    focus_score = _crop_focus_score(crop)

    model_size = _FACE_RESTORER_SIZE
    model_image = crop.resize((model_size, model_size), Image.Resampling.LANCZOS)
    model_input = np.asarray(model_image, dtype=np.float32) / 127.5 - 1.0
    model_input = np.transpose(model_input, (2, 0, 1))[None]
    with _FACE_RESTORER_LOCK:
        model_output = _get_face_restorer().run(None, {"input": model_input})[0][0]
    restored_values = np.transpose(model_output, (1, 2, 0))
    restored_values = np.clip((restored_values + 1.0) * 127.5, 0, 255).astype(np.uint8)
    restored_raw = Image.fromarray(restored_values, mode="RGB").resize(crop.size, Image.Resampling.LANCZOS)
    face_resolution = min(face_width, face_height)
    focus_blend = 0.94 if focus_score < 12 else 0.76 if focus_score < 20 else 0.42 if focus_score < 30 else 0.22
    resolution_blend = 0.46 if face_resolution >= 360 else 0.6 if face_resolution >= 280 else 0.74 if face_resolution >= 200 else 0.94
    face_blend = min(focus_blend, resolution_blend)
    hair_blend = min(0.42, face_blend * 0.5)
    focus_texture = 0.86 if focus_score < 12 else 0.54 if focus_score < 20 else 0.22 if focus_score < 30 else 0.1
    resolution_texture = 0.24 if face_resolution >= 360 else 0.4 if face_resolution >= 280 else 0.56 if face_resolution >= 200 else 0.86
    texture_strength = min(focus_texture, resolution_texture)
    restored_raw = _enhance_face_texture(restored_raw, texture_strength)
    restored_hair = Image.blend(crop, restored_raw, hair_blend)
    restored_hair = restored_hair.filter(ImageFilter.UnsharpMask(radius=0.8, percent=48, threshold=3))
    restored_face = Image.blend(crop, restored_raw, face_blend)
    restored_face = restored_face.filter(ImageFilter.UnsharpMask(radius=1.35, percent=72, threshold=2))

    relative_x = face_x - left
    relative_y = face_y - top
    hair_mask = Image.new("L", crop.size, 0)
    hair_draw = ImageDraw.Draw(hair_mask)
    hair_padding = round(crop_size * 0.035)
    hair_draw.rounded_rectangle(
        (hair_padding, hair_padding, crop_size - hair_padding, crop_size - hair_padding),
        radius=round(crop_size * 0.16),
        fill=225,
    )
    hair_mask = hair_mask.filter(ImageFilter.GaussianBlur(radius=max(8, round(crop_size * 0.035))))
    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(
        (
            relative_x - round(face_width * 0.1),
            relative_y - round(face_height * 0.08),
            relative_x + round(face_width * 1.1),
            relative_y + round(face_height * 1.14),
        ),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(6, round(face_width * 0.075))))
    result = image.copy()
    result.paste(restored_hair, (left, top), hair_mask)
    result.paste(restored_face, (left, top), mask)
    return result


def _face_focus_score(image: Image.Image) -> float | None:
    face = _detect_primary_face(image)
    if face is None:
        return None
    face_x, face_y, face_width, face_height = face
    padding = round(max(face_width, face_height) * 0.2)
    left = max(0, face_x - padding)
    top = max(0, face_y - padding)
    right = min(image.width, face_x + face_width + padding)
    bottom = min(image.height, face_y + face_height + padding)
    return _crop_focus_score(image.crop((left, top, right, bottom)))


def _face_resolution(image: Image.Image) -> int | None:
    face = _detect_primary_face(image)
    if face is None:
        return None
    return min(face[2], face[3])


def _crop_focus_score(image: Image.Image) -> float:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    quality_sample = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return float(cv2.Laplacian(quality_sample, cv2.CV_64F).var())


def _stone_adjustments(options: EngravingOptions) -> tuple[int, int, int]:
    if options.stone_profile == "gray_granite":
        return round(options.contrast * 0.82), round(options.detail * 0.9), round(options.shadows * 0.72)
    if options.stone_profile == "marble":
        return round(options.contrast * 0.62), round(options.detail * 0.58), round(options.shadows * 0.48)
    return options.contrast, options.detail, options.shadows


def _fit_to_portrait_canvas(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("L", target_size, 0)
    fitted = ImageOps.contain(image, target_size, Image.Resampling.LANCZOS)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _fit_to_color_canvas(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    fitted = ImageOps.contain(image.convert("RGB"), target_size, Image.Resampling.LANCZOS)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _crop_for_composition(image: Image.Image, portrait_mode: str, target_size: tuple[int, int]) -> Image.Image:
    if portrait_mode != "chest":
        return image

    target_ratio = target_size[0] / target_size[1]
    face = _detect_primary_face(image)
    if face is not None:
        face_x, face_y, face_width, face_height = face
        desired_height = min(image.height, max(round(face_height * 3.05), round(image.height * 0.68)))
        desired_width = round(desired_height * target_ratio)
        if desired_width > image.width:
            desired_width = image.width
            desired_height = min(image.height, round(desired_width / target_ratio))
        center_x = face_x + face_width // 2
        left = min(max(0, center_x - desired_width // 2), image.width - desired_width)
        top = min(max(0, round(face_y - face_height * 0.85)), image.height - desired_height)
        return image.crop((left, top, left + desired_width, top + desired_height))

    desired_height = max(1, round(image.height * 0.66))
    desired_width = max(1, round(desired_height * target_ratio))
    if desired_width > image.width:
        desired_width = image.width
        desired_height = min(image.height, round(desired_width / target_ratio))
    desired_height = min(desired_height, image.height)
    left = max(0, (image.width - desired_width) // 2)
    top = min(max(0, round(image.height * 0.04)), image.height - desired_height)
    return image.crop((left, top, left + desired_width, top + desired_height))


def _detect_primary_face(image: Image.Image) -> tuple[int, int, int, int] | None:
    thumbnail = ImageOps.contain(image, (1000, 1000), Image.Resampling.BILINEAR)
    gray = cv2.cvtColor(np.asarray(thumbnail.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(45, 45))
    if len(faces) == 0:
        return None
    face_x, face_y, face_width, face_height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    scale_x = image.width / thumbnail.width
    scale_y = image.height / thumbnail.height
    return (
        round(face_x * scale_x),
        round(face_y * scale_y),
        round(face_width * scale_x),
        round(face_height * scale_y),
    )


def _build_background_mask(image: Image.Image) -> Image.Image:
    try:
        return _build_mediapipe_background_mask(image)
    except Exception:
        try:
            return _build_grabcut_background_mask(image)
        except cv2.error:
            return _build_edge_background_mask(image)


def _resource_path(relative_path: str) -> Path:
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


def _get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER

    import mediapipe as mp

    model_path = _resource_path("models/selfie_segmenter.tflite")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    _SEGMENTER = mp.tasks.vision.ImageSegmenter.create_from_options(options)
    return _SEGMENTER


def _build_mediapipe_background_mask(image: Image.Image) -> Image.Image:
    import mediapipe as mp

    thumbnail = ImageOps.contain(image, (720, 960), Image.Resampling.BILINEAR)
    rgb = np.ascontiguousarray(np.asarray(thumbnail.convert("RGB")))
    media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _SEGMENTER_LOCK:
        result = _get_segmenter().segment(media_image)
    if not result.confidence_masks:
        raise ProcessingError("Модель не смогла выделить человека на фотографии.")

    confidence = np.squeeze(result.confidence_masks[0].numpy_view()).copy()
    foreground = np.clip((confidence - 0.28) / 0.42, 0.0, 1.0)
    component_mask = _primary_subject_component(thumbnail, foreground)
    foreground *= component_mask
    foreground = cv2.GaussianBlur(foreground.astype(np.float32), (7, 7), 0)
    background = np.clip((1.0 - foreground) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(background, mode="L").resize(image.size, Image.Resampling.LANCZOS)


def _primary_subject_component(image: Image.Image, foreground: np.ndarray) -> np.ndarray:
    binary = (foreground >= 0.42).astype(np.uint8)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.ones_like(foreground, dtype=np.float32)

    chosen = 0
    face = _detect_primary_face(image)
    if face is not None:
        face_x, face_y, face_width, face_height = face
        center_x = min(labels.shape[1] - 1, max(0, face_x + face_width // 2))
        center_y = min(labels.shape[0] - 1, max(0, face_y + face_height // 2))
        chosen = int(labels[center_y, center_x])
    if chosen == 0:
        chosen = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))

    selected = (labels == chosen).astype(np.uint8)
    kernel_size = max(3, round(min(image.size) * 0.008))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel)
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
    return selected.astype(np.float32)


def _build_grabcut_background_mask(image: Image.Image) -> Image.Image:
    """Separate a centered portrait from its background with GrabCut."""
    thumbnail = ImageOps.contain(image, (360, 540), Image.Resampling.BILINEAR)
    width, height = thumbnail.size
    rgb = np.asarray(thumbnail.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)

    border = max(2, round(min(width, height) * 0.015))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(28, 28))
    if len(faces) == 0:
        return _build_edge_background_mask(image)

    face_x, face_y, face_width, face_height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    face_center_x = face_x + face_width // 2
    face_center_y = face_y + face_height // 2
    cv2.ellipse(
        mask,
        (face_center_x, face_center_y),
        (max(8, round(face_width * 0.65)), max(8, round(face_height * 0.78))),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1,
    )
    mask[
        max(border, face_y + round(face_height * 0.12)):min(height - border, face_y + round(face_height * 0.9)),
        max(border, face_x + round(face_width * 0.12)):min(width - border, face_x + round(face_width * 0.88)),
    ] = cv2.GC_FGD

    torso_top = min(height - border - 1, face_y + round(face_height * 0.78))
    torso_bottom = height - border - 1
    torso = np.array(
        [
            [max(border, face_center_x - round(face_width * 0.48)), torso_top],
            [min(width - border - 1, face_center_x + round(face_width * 0.48)), torso_top],
            [min(width - border - 1, face_center_x + round(face_width * 0.72)), torso_bottom],
            [max(border, face_center_x - round(face_width * 0.72)), torso_bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, torso, cv2.GC_PR_FGD)
    core = np.array(
        [
            [max(border, face_center_x - round(face_width * 0.2)), torso_top],
            [min(width - border - 1, face_center_x + round(face_width * 0.2)), torso_top],
            [min(width - border - 1, face_center_x + round(face_width * 0.3)), torso_bottom],
            [max(border, face_center_x - round(face_width * 0.3)), torso_bottom],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, core, cv2.GC_FGD)

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, None, background_model, foreground_model, 2, cv2.GC_INIT_WITH_MASK)
    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
    background = 255 - foreground
    return Image.fromarray(background, mode="L").resize(image.size, Image.Resampling.LANCZOS)


def _build_edge_background_mask(image: Image.Image) -> Image.Image:
    """Fallback mask for a mostly uniform backdrop connected to the image edge."""
    thumbnail = ImageOps.contain(image, (700, 1250), Image.Resampling.BILINEAR)
    width, height = thumbnail.size
    values = list(thumbnail.getdata())
    edge_values = []
    for x in range(width):
        edge_values.append(values[x])
        edge_values.append(values[(height - 1) * width + x])
    for y in range(1, height - 1):
        edge_values.append(values[y * width])
        edge_values.append(values[y * width + width - 1])

    reference = tuple(sorted(pixel[channel] for pixel in edge_values)[len(edge_values) // 2] for channel in range(3))
    distances = sorted(_color_distance(pixel, reference) for pixel in edge_values)
    tolerance = max(42, min(150, distances[round((len(distances) - 1) * 0.9)] * 2 + 18))
    background = bytearray(width * height)
    pending: deque[tuple[int, int]] = deque()

    def add_if_background(x: int, y: int) -> None:
        index = y * width + x
        if not background[index] and _color_distance(values[index], reference) <= tolerance:
            background[index] = 255
            pending.append((x, y))

    for x in range(width):
        add_if_background(x, 0)
        add_if_background(x, height - 1)
    for y in range(height):
        add_if_background(0, y)
        add_if_background(width - 1, y)

    while pending:
        x, y = pending.popleft()
        if x:
            add_if_background(x - 1, y)
        if x + 1 < width:
            add_if_background(x + 1, y)
        if y:
            add_if_background(x, y - 1)
        if y + 1 < height:
            add_if_background(x, y + 1)

    return Image.frombytes("L", (width, height), bytes(background)).resize(image.size, Image.Resampling.NEAREST)


def _color_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) + abs(first[2] - second[2])


def _safe_name(value: str) -> str:
    invalid = '<>:"/\\|?*'
    result = "".join("_" if character in invalid else character for character in value).strip(". ")
    return result or "portrait"
