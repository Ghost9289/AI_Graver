from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError


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
_FACE_RESTORER_LOCK = threading.Lock()
_SUPER_RESOLVER = None
_SUPER_RESOLVER_LOCK = threading.Lock()
# Eye, eye, nose tip, mouth corner, mouth corner of an FFHQ-aligned 512px face (the GFPGAN/GPEN training layout).
_FFHQ_TEMPLATE_512 = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32,
)


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
    # The source is already an AI retouch (Codex/OpenAI): skip the local face model, it would only blur it.
    ai_retouched: bool = False


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
        if not options.ai_retouched:
            image = _cached_face_restoration(source, image, options.portrait_mode, options.restoration_mode)
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


def _cached_face_restoration(source: Path, image: Image.Image, portrait_mode: str, restoration_mode: str) -> Image.Image:
    try:
        modified = source.stat().st_mtime_ns
    except OSError:
        modified = 0
    key = (str(source.resolve()), modified, f"{portrait_mode}:{restoration_mode}")
    cached = _RESTORED_IMAGE_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    restored = _restore_primary_face(_suppress_print_noise(image, restoration_mode), restoration_mode)
    if len(_RESTORED_IMAGE_CACHE) >= 8:
        _RESTORED_IMAGE_CACHE.pop(next(iter(_RESTORED_IMAGE_CACHE)))
    _RESTORED_IMAGE_CACHE[key] = restored.copy()
    return restored


def _suppress_print_noise(image: Image.Image, restoration_mode: str) -> Image.Image:
    """Scanned prints carry film grain and halftone dots; detail enhancement would turn them into speckles."""
    strength = {"strong": 5, "old_photo": 8}.get(restoration_mode)
    if strength is None:
        return image
    denoised = cv2.fastNlMeansDenoisingColored(np.asarray(image.convert("RGB")), None, strength, strength, 7, 21)
    return Image.fromarray(denoised, mode="RGB")


def _get_face_restorer():
    global _FACE_RESTORER
    global _FACE_RESTORER_SIZE
    if _FACE_RESTORER is not None:
        return _FACE_RESTORER

    import onnxruntime as ort

    model_path = _resource_path("models/GFPGANv1.4.onnx")
    if not model_path.is_file():
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


def _detect_face_landmarks(image: Image.Image) -> np.ndarray | None:
    """Five landmarks (eyes, nose tip, mouth corners) of the largest face, in image coordinates."""
    model_path = _resource_path("models/face_detection_yunet_2023mar.onnx")
    if not model_path.is_file():
        return None
    thumbnail = ImageOps.contain(image, (1280, 1280), Image.Resampling.BILINEAR)
    bgr = cv2.cvtColor(np.asarray(thumbnail.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (bgr.shape[1], bgr.shape[0]), 0.6, 0.3, 50)
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        return None
    face = max(faces, key=lambda item: float(item[2]) * float(item[3]))
    landmarks = face[4:14].reshape(5, 2).astype(np.float32)
    landmarks[:, 0] *= image.width / thumbnail.width
    landmarks[:, 1] *= image.height / thumbnail.height
    return landmarks


def _restore_primary_face(image: Image.Image, restoration_mode: str = "natural") -> Image.Image:
    """GFPGAN on the face aligned to the FFHQ template, then pasted back through the inverse transform.

    GFPGAN only works on faces aligned the way it was trained; an unaligned crop gives a smeared face.
    """
    landmarks = _detect_face_landmarks(image)
    if landmarks is None:
        return image
    _get_face_restorer()  # loads the model and fixes its input size
    model_size = _FACE_RESTORER_SIZE
    template = _FFHQ_TEMPLATE_512 * (model_size / 512)
    affine, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if affine is None:
        return image

    rgb = np.asarray(image.convert("RGB"))
    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
    aligned = cv2.warpAffine(
        rgb,
        affine,
        (model_size, model_size),
        flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135, 133, 132),
    )
    model_input = np.transpose(aligned.astype(np.float32) / 127.5 - 1.0, (2, 0, 1))[None]
    with _FACE_RESTORER_LOCK:
        model_output = _get_face_restorer().run(None, {"input": model_input})[0][0]
    restored = np.clip((np.transpose(model_output, (1, 2, 0)) + 1.0) * 127.5, 0, 255).astype(np.uint8)

    inverse = cv2.invertAffineTransform(affine)
    size = (rgb.shape[1], rgb.shape[0])
    restored_back = cv2.warpAffine(restored, inverse, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    edge = max(4, model_size // 24)
    mask = np.zeros((model_size, model_size), dtype=np.float32)
    mask[edge:-edge, edge:-edge] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), edge * 1.5)
    mask_back = cv2.warpAffine(mask, inverse, size, flags=cv2.INTER_LINEAR)

    # Stronger modes trust the model more; a face much larger than the model output keeps more of its own detail.
    strength = {"natural": 0.7, "strong": 0.85, "old_photo": 1.0}.get(restoration_mode, 0.7)
    if scale < 0.8:
        strength *= 0.8
    alpha = (mask_back * strength)[..., None]
    blended = rgb.astype(np.float32) * (1 - alpha) + restored_back.astype(np.float32) * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


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
