from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class StoneProfile:
    """How a stone shows a laser-engraved tonal portrait.

    base / engraved: colour of untouched polished stone and of a fully engraved spot.
    face_target: median tone of the face in the Graver file (0-255) that reads best on this stone.
    white_limit / black_floor: tonal range of the person (background stays black = not engraved).
    grain: natural speckle of the stone; coarse grain eats micro-detail, so detail is lowered.
    """

    key: str
    name: str
    group: str
    base: tuple[int, int, int]
    engraved: tuple[int, int, int]
    grain: int
    face_target: int
    white_limit: int
    black_floor: int
    contrast_mul: float
    detail_mul: float
    shadows_mul: float
    blur: float = 0.0


# Starting values; calibrate on a real test plate of each stone and edit stones.json.
DEFAULT_STONES: tuple[StoneProfile, ...] = (
    StoneProfile("black_granite", "Габбро-диабаз (Карелия)", "Чёрные", (22, 22, 24), (205, 205, 200), 5, 140, 238, 14, 1.0, 1.0, 1.0),
    StoneProfile("shanxi_black", "Шанси Блэк (Китай)", "Чёрные", (16, 16, 18), (215, 215, 212), 3, 136, 240, 12, 0.95, 1.1, 1.0),
    StoneProfile("absolute_black", "Абсолют Блэк (Индия)", "Чёрные", (14, 14, 16), (212, 212, 208), 3, 136, 240, 12, 0.95, 1.1, 1.0),
    StoneProfile("mongolia_black", "Монголия Блэк", "Чёрные", (34, 34, 36), (200, 200, 198), 8, 146, 236, 16, 1.08, 0.92, 1.0),
    StoneProfile("nero_marquina", "Мрамор чёрный (Неро Маркина)", "Чёрные", (26, 26, 28), (190, 190, 190), 4, 148, 232, 16, 1.1, 0.8, 0.9, 0.3),
    StoneProfile("galaxy_black", "Гранит «Галактика» (Индия)", "Тёмные с зерном", (26, 24, 22), (205, 203, 195), 14, 150, 236, 18, 1.15, 0.75, 0.95),
    StoneProfile("dymovsky", "Дымовский (тёмно-зелёный)", "Тёмные с зерном", (48, 52, 50), (205, 208, 205), 10, 152, 238, 20, 1.18, 0.85, 0.9),
    StoneProfile("amphibolite", "Амфиболит", "Тёмные с зерном", (45, 48, 47), (198, 200, 198), 12, 154, 236, 20, 1.2, 0.8, 0.9),
    StoneProfile("labradorite", "Лабрадорит / габбро с иризацией", "Тёмные с зерном", (55, 58, 64), (200, 200, 205), 16, 158, 236, 22, 1.25, 0.7, 0.85),
    StoneProfile("gray_granite", "Серый гранит (общий)", "Серые", (120, 120, 120), (215, 215, 215), 20, 160, 240, 24, 0.82, 0.9, 0.72),
    StoneProfile("mansurovsky", "Мансуровский (серый)", "Серые", (132, 132, 128), (220, 220, 218), 22, 162, 240, 26, 0.9, 0.78, 0.7, 0.2),
    StoneProfile("pokostovsky", "Покостовский (светло-серый)", "Серые", (150, 148, 145), (225, 224, 222), 24, 165, 240, 28, 0.95, 0.7, 0.65, 0.3),
    StoneProfile("kapustinsky", "Капустинский (красный)", "Красные / коричневые", (130, 70, 60), (215, 190, 180), 22, 162, 240, 26, 1.05, 0.72, 0.7, 0.3),
    StoneProfile("leznikovsky", "Лезниковский (красный)", "Красные / коричневые", (120, 55, 50), (210, 180, 170), 20, 160, 240, 24, 1.05, 0.75, 0.72, 0.25),
    StoneProfile("baltic_brown", "Балтик Браун", "Красные / коричневые", (95, 70, 50), (200, 185, 170), 26, 162, 238, 26, 1.1, 0.65, 0.7, 0.35),
    StoneProfile("marble", "Мрамор белый", "Мрамор", (225, 225, 222), (245, 245, 243), 6, 150, 236, 20, 0.62, 0.58, 0.48, 0.35),
    StoneProfile("coelga", "Мрамор Коелга (белый)", "Мрамор", (215, 213, 205), (240, 238, 232), 8, 150, 236, 20, 0.65, 0.58, 0.5, 0.35),
)

_catalog: dict[str, StoneProfile] = {stone.key: stone for stone in DEFAULT_STONES}


def load_catalog(path: Path) -> list[StoneProfile]:
    """Load the editable stone list; write defaults on first run. Bad entries are skipped."""
    global _catalog
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(stone) for stone in DEFAULT_STONES], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    catalog = {stone.key: stone for stone in DEFAULT_STONES}
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entries = []
    names = {field.name for field in fields(StoneProfile)}
    for entry in entries if isinstance(entries, list) else []:
        try:
            values = {name: entry[name] for name in names if name in entry}
            values["base"] = tuple(values["base"])
            values["engraved"] = tuple(values["engraved"])
            stone = StoneProfile(**values)
        except (KeyError, TypeError, ValueError):
            continue
        catalog[stone.key] = stone
    _catalog = catalog
    return list(catalog.values())


def get_stone(key: str) -> StoneProfile:
    return _catalog.get(key) or _catalog["black_granite"]


def all_stones() -> list[StoneProfile]:
    return list(_catalog.values())


@dataclass(frozen=True)
class ToneReport:
    face_found: bool
    face_before: int
    face_after: int
    target: int
    highlights_clipped: float
    passes: int

    def summary(self, stone: StoneProfile) -> str:
        region = "лицо" if self.face_found else "центр портрета"
        return (
            f"Коррекция под «{stone.name}»: {region} {self.face_before} → {self.face_after} "
            f"(цель {self.target}), пересвет {self.highlights_clipped:.1f}%, проходов {self.passes}"
        )


def fit_tones_to_stone(
    gray: Image.Image,
    stone: StoneProfile,
    face_box: tuple[int, int, int, int] | None,
    max_passes: int = 4,
) -> tuple[Image.Image, ToneReport]:
    """Check the prepared portrait and correct its tones until the face sits on the stone's target.

    Background (near-black, not engraved) is left untouched. Each pass measures the
    result again, so the report reflects the file that goes to Graver.
    """
    values = np.asarray(gray, dtype=np.float32)
    subject = values > 12
    if subject.sum() < values.size * 0.02:
        return gray, ToneReport(face_box is not None, 0, 0, stone.face_target, 0.0, 0)

    face_region = _face_region(values.shape, face_box) & subject
    if face_region.sum() < 50:
        face_region = subject

    low, high = np.percentile(values[subject], (1.0, 99.6))
    normalized = np.clip((values - low) / max(high - low, 1.0), 0.0, 1.0)
    span = stone.white_limit - stone.black_floor
    target = (stone.face_target - stone.black_floor) / span
    target = min(max(target, 0.05), 0.95)
    face_before = int(np.median(values[face_region]))

    gamma = 1.0
    result = values
    passes = 0
    for passes in range(1, max_passes + 1):
        mapped = normalized ** gamma
        result = np.where(subject, stone.black_floor + mapped * span, values)
        face_median = float(np.median(result[face_region]))
        if abs(face_median - stone.face_target) <= 3:
            break
        current = min(max((face_median - stone.black_floor) / span, 0.02), 0.98)
        gamma = float(np.clip(gamma * np.log(target) / np.log(current), 0.45, 2.2))

    clipped = float((result[subject] >= stone.white_limit - 1).mean() * 100)
    output = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="L")
    if stone.blur > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=stone.blur))
    report = ToneReport(
        face_found=face_box is not None,
        face_before=face_before,
        face_after=int(np.median(np.asarray(output, dtype=np.float32)[face_region])),
        target=stone.face_target,
        highlights_clipped=clipped,
        passes=passes,
    )
    return output, report


def _face_region(shape: tuple[int, ...], face_box: tuple[int, int, int, int] | None) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    if face_box is None:
        mask[round(height * 0.12):round(height * 0.45), round(width * 0.3):round(width * 0.7)] = True
        return mask
    x, y, w, h = face_box
    mask[max(0, y + h // 6):min(height, y + h * 5 // 6), max(0, x + w // 6):min(width, x + w * 5 // 6)] = True
    return mask


def simulate_on_stone(gray: Image.Image, stone: StoneProfile, seed: int = 7) -> Image.Image:
    """Approximate look of the engraving on polished stone (for the operator's eye only)."""
    tone = np.asarray(gray, dtype=np.float32)[..., None] / 255.0
    base = np.array(stone.base, dtype=np.float32)
    engraved = np.array(stone.engraved, dtype=np.float32)
    height, width = tone.shape[:2]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
    grain_image = Image.fromarray(np.clip(noise * 40 + 128, 0, 255).astype(np.uint8), mode="L")
    grain_image = grain_image.filter(ImageFilter.GaussianBlur(radius=max(0.6, width / 900)))
    grain = (np.asarray(grain_image, dtype=np.float32) - 128.0)[..., None] / 40.0 * stone.grain
    result = base + (engraved - base) * tone + grain * (1.0 - tone * 0.6)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
