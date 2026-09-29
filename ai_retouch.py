from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


OPENAI_EDITS_URL = "https://api.openai.com/v1/images/edits"
DEFAULT_MODEL = "gpt-image-1"
OUTPUT_SIZE = "1024x1536"
MAX_UPLOAD_SIDE = 2048
REQUEST_TIMEOUT = 300
CODEX_TIMEOUT = 900
CODEX_INSTRUCTIONS = (
    "\n\nUse your image generation tool to edit the attached photo following the instructions above. "
    "Output one portrait image with 2:3 aspect ratio. Save the generated image as result.png in the "
    "current directory. Do nothing else and do not write any code."
)

DEFAULT_TEMPLATE = """\
Professional photo retouch of this portrait for laser engraving on a black granite memorial.

IDENTITY IS THE TOP PRIORITY:
- Keep exactly the same person: same face shape, eyes, eyelids, nose, lips, ears, eyebrows, hairline, age and expression.
- Keep all real wrinkles, folds, moles and scars. Do not beautify, do not make younger, do not change weight.
- Do not invent a different face. If a detail is not visible in the source, keep it soft rather than making it up.

RESTORATION:
- Remove blur, noise, JPEG artefacts, scratches, dust, creases and stains of an old print.
- Make eyes, eyelashes, wrinkles and hair strands crisp and clearly readable.
- Correct exposure: even soft studio light, open up dark shadows on the face, recover blown highlights.

COMPOSITION:
- Pure black background (#000000), nothing else behind the person, clean soft edge around hair.
- Neat, dark, classic clothing (dark suit or dark blouse/dress); fix wrinkled or cut-off clothing, keep the original style if it is already neat.
- Head and shoulders, person centred, looking the same direction as in the source.

OUTPUT STYLE:
- Black-and-white monochrome photograph, rich tonal range, strong but natural local contrast,
  high micro-detail on skin and hair, suitable for engraving on dark granite.
- Photographic realism, no drawing, no painting, no text, no frame, no watermark.
"""


class AIRetouchError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# Errors that belong to the key itself (revoked, no access, limit or empty balance): try the next key.
_KEY_ERRORS = {401, 403, 429}


def ensure_template(path: Path) -> Path:
    """Create the editable prompt template on first run."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")
    return path


def load_template(path: Path) -> str:
    text = ensure_template(path).read_text(encoding="utf-8").strip()
    return text or DEFAULT_TEMPLATE


def find_codex() -> str | None:
    """Codex CLI signed in with ChatGPT: image generation is paid by the ChatGPT subscription."""
    found = shutil.which("codex")
    if found:
        return found
    candidate = Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd"
    return str(candidate) if candidate.is_file() else None


def retouch_with_openai(
    source: Path,
    output_path: Path,
    api_keys: str | list[str],
    template: str,
    cache_dir: Path,
    model: str = DEFAULT_MODEL,
    use_codex: bool = True,
) -> Path:
    """Retouch the photo with the template prompt: Codex (ChatGPT subscription) first, then API keys in order.

    Results are cached by photo content + prompt, so the same photo is never sent twice.
    """
    keys = [api_keys] if isinstance(api_keys, str) else list(api_keys)
    keys = [key.strip() for key in keys if key and key.strip()]
    codex = find_codex() if use_codex else None
    if not keys and not codex:
        raise AIRetouchError("Нет ни Codex (вход через ChatGPT), ни ключа OpenAI API.")
    model = model.strip() or DEFAULT_MODEL

    upload = _prepare_upload(source)
    cache_key = hashlib.sha256(upload + template.encode("utf-8")).hexdigest()[:32]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{cache_key}.png"
    if not cached.is_file():
        errors: list[str] = []
        image_bytes: bytes | None = None
        if codex:
            try:
                image_bytes = _request_codex(codex, upload, template)
            except AIRetouchError as error:
                errors.append(f"Codex: {error}")
        if image_bytes is None and keys:
            try:
                image_bytes = _request_with_any_key(upload, keys, template, model)
            except AIRetouchError as error:
                errors.append(str(error))
        if image_bytes is None:
            raise AIRetouchError(" | ".join(errors))
        temporary = cached.with_suffix(".tmp")
        temporary.write_bytes(image_bytes)
        temporary.replace(cached)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, output_path)
    return output_path


def _prepare_upload(source: Path) -> bytes:
    try:
        with Image.open(source) as input_image:
            image = ImageOps.exif_transpose(input_image).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise AIRetouchError(f"Не удалось открыть изображение: {source}") from error
    if max(image.size) > MAX_UPLOAD_SIDE:
        image = ImageOps.contain(image, (MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _request_codex(codex: str, upload: bytes, template: str) -> bytes:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="ai_graver_codex_", ignore_cleanup_errors=True) as folder:
        workdir = Path(folder)
        (workdir / "source.png").write_bytes(upload)
        command = [codex, "exec", "--skip-git-repo-check", "--sandbox", "workspace-write", "-C", str(workdir), "-i", "source.png", "-"]
        try:
            completed = subprocess.run(
                command,
                input=template + CODEX_INSTRUCTIONS,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CODEX_TIMEOUT,
                cwd=workdir,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as error:
            raise AIRetouchError("Codex не ответил за 15 минут.") from error
        except OSError as error:
            raise AIRetouchError(f"Не удалось запустить Codex: {error}") from error
        # Codex keeps every generated picture in its own folder (readable by us); the copy it
        # writes inside its sandbox can carry sandbox-only permissions, so it is only a fallback.
        generated = Path.home() / ".codex" / "generated_images"
        fresh = [item for item in generated.glob("*/*.png") if item.stat().st_mtime >= started] if generated.is_dir() else []
        if fresh:
            return max(fresh, key=lambda item: item.stat().st_mtime).read_bytes()
        try:
            result = (workdir / "result.png").read_bytes()
            if result:
                return result
        except OSError:
            pass
        tail = " ".join((completed.stdout + completed.stderr).split()[-40:])
        raise AIRetouchError(f"Codex не создал изображение (код {completed.returncode}). {tail}")


def _request_with_any_key(upload: bytes, keys: list[str], template: str, model: str) -> bytes:
    errors: list[str] = []
    for number, key in enumerate(keys, start=1):
        try:
            return _request_edit(upload, key, template, model)
        except AIRetouchError as error:
            if error.status not in _KEY_ERRORS:
                raise
            errors.append(f"ключ {number}: {error}")
    raise AIRetouchError("Ни один ключ OpenAI не сработал. " + " | ".join(errors), 429)


def _request_edit(upload: bytes, api_key: str, template: str, model: str) -> bytes:
    fields = {
        "model": model,
        "prompt": template,
        "size": OUTPUT_SIZE,
        "quality": "high",
        "n": "1",
    }
    if model.startswith("gpt-image-1"):
        # Keeps the face much closer to the source photo on gpt-image-1 models.
        fields["input_fidelity"] = "high"
    try:
        return _post_edit(upload, api_key, fields)
    except AIRetouchError as error:
        if "input_fidelity" in fields and "input_fidelity" in str(error):
            fields.pop("input_fidelity")
            return _post_edit(upload, api_key, fields)
        raise


def _post_edit(upload: bytes, api_key: str, fields: dict[str, str]) -> bytes:
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for name, value in fields.items():
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode("utf-8"))
        body.write(value.encode("utf-8"))
        body.write(b"\r\n")
    body.write(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"portrait.png\"\r\n"
        "Content-Type: image/png\r\n\r\n".encode("utf-8")
    )
    body.write(upload)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        OPENAI_EDITS_URL,
        data=body.getvalue(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise AIRetouchError(_describe_http_error(error), error.code) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise AIRetouchError(f"Нет связи с OpenAI: {error}") from error

    try:
        return base64.b64decode(payload["data"][0]["b64_json"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise AIRetouchError("OpenAI вернул ответ без изображения.") from error


def _describe_http_error(error: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(error.read().decode("utf-8")).get("error", {}).get("message", "")
    except (ValueError, OSError, AttributeError):
        detail = ""
    if error.code == 401:
        return "OpenAI отклонил ключ API (401)."
    if error.code == 429:
        return f"OpenAI: превышен лимит или закончились средства на балансе (429). {detail}".strip()
    if error.code == 403:
        return f"OpenAI: нет доступа к модели (403). {detail}".strip()
    return f"OpenAI вернул ошибку {error.code}. {detail}".strip()
