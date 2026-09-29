from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppSettings:
    profile_version: int = 6
    contrast: int = 26
    detail: int = 46
    shadows: int = 18
    black_background: bool = True
    export_bmp: bool = True
    portrait_mode: str = "chest"
    graver_exe: str = ""
    auto_mode: bool = True
    stone_profile: str = "black_granite"
    enhance_4k: bool = True
    restoration_mode: str = "natural"


def load_settings(path: Path) -> AppSettings:
    parser = ConfigParser()
    if not path.exists():
        return AppSettings()
    parser.read(path, encoding="utf-8")
    section = parser["AI_GRAVER"] if parser.has_section("AI_GRAVER") else {}
    profile_version = int(section.get("profile_version", 0))
    use_new_profile = profile_version < 6
    restoration_mode = str(section.get("restoration_mode", "natural"))
    if restoration_mode not in {"natural", "strong", "old_photo"}:
        restoration_mode = "natural"
    return AppSettings(
        profile_version=6,
        contrast=26 if use_new_profile else int(section.get("contrast", 26)),
        detail=46 if use_new_profile else int(section.get("detail", 46)),
        shadows=18 if use_new_profile else int(section.get("shadows", 18)),
        black_background=str(section.get("black_background", "true")).lower() == "true",
        export_bmp=str(section.get("export_bmp", "true")).lower() == "true",
        portrait_mode=str(section.get("portrait_mode", "chest")),
        graver_exe=str(section.get("graver_exe", "")),
        auto_mode=str(section.get("auto_mode", "true")).lower() == "true",
        stone_profile=str(section.get("stone_profile", "black_granite")),
        enhance_4k=str(section.get("enhance_4k", "true")).lower() == "true",
        restoration_mode=restoration_mode,
    )


def save_settings(path: Path, settings: AppSettings) -> None:
    parser = ConfigParser()
    parser["AI_GRAVER"] = {
        "profile_version": str(settings.profile_version),
        "contrast": str(settings.contrast),
        "detail": str(settings.detail),
        "shadows": str(settings.shadows),
        "black_background": str(settings.black_background).lower(),
        "export_bmp": str(settings.export_bmp).lower(),
        "portrait_mode": settings.portrait_mode,
        "graver_exe": settings.graver_exe,
        "auto_mode": str(settings.auto_mode).lower(),
        "stone_profile": settings.stone_profile,
        "enhance_4k": str(settings.enhance_4k).lower(),
        "restoration_mode": settings.restoration_mode,
    }
    with path.open("w", encoding="utf-8") as file:
        parser.write(file)
