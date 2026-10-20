"""Fridge photo -> shopping list. Same local model, different question.

The model does not get to invent a shopping list. It gets a fixed list of staples
and has to say which ones it can see. Everything it does not confirm goes on the list.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path

from .config import Config
from .llm import OllamaBackend, ensure_local, extract_json

SPACE = " "
CHECKBOX = f"- [{SPACE}]{SPACE}"

VISION_SYSTEM = (
    "You check fridge photos against a shopping list of staples. "
    "Reply with ONE JSON object and nothing else."
)


def vision_prompt(staples: list[str]) -> str:
    listing = chr(10).join(f"- {staple}" for staple in staples)
    return f"""Look at this fridge photo and decide, one by one, which of these staples you can actually see:
{listing}

Return JSON only, exactly this shape:
{{"present": ["butter"], "absent": ["milk", "eggs"]}}

Rules:
- put a staple in present only if you are sure you can see it
- every staple you are sure about goes in exactly one of the two lists
- if you cannot tell, leave it out of present
"""


def encode_image(path: Path) -> str:
    """Base64 the photo. If Pillow happens to be installed, straighten and shrink it first."""
    raw = Path(path).expanduser().read_bytes()
    try:
        from PIL import Image, ImageOps  # optional, not required

        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        raw = buffer.getvalue()
    except Exception:
        # ponytail: no Pillow, no problem, the model can read the original bytes
        pass
    return base64.b64encode(raw).decode()


def _matches(staple: str, names: list[str]) -> bool:
    key = staple.lower().strip()
    return any(key in name.lower() or name.lower() in key for name in names)


def photo_to_items(cfg: Config, image_path: Path) -> dict:
    """Ask the local vision model which staples it can see, then diff against the staples."""
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such image: {path}")
    ensure_local(cfg.ollama_host)
    backend = OllamaBackend(cfg.ollama_host, cfg.vl_model)
    if not backend.available():
        raise RuntimeError(
            f"no ollama model '{cfg.vl_model}' at {cfg.ollama_host}. "
            f"pull it with: ollama pull {cfg.vl_model}"
        )
    staples = list(cfg.staples)
    raw = backend.chat(
        VISION_SYSTEM,
        vision_prompt(staples),
        images=[encode_image(path)],
        max_tokens=500,
        temperature=0.0,
    )
    data = extract_json(raw) or {}
    present_names = [str(name) for name in (data.get("present") or []) if str(name).strip()]
    present = [staple for staple in staples if _matches(staple, present_names)]
    missing = [staple for staple in staples if staple not in present]
    return {"present": present, "missing": missing, "raw": raw}


def render_list(present: list[str], missing: list[str]) -> str:
    lines = [f"# Shopping list - {datetime.now().strftime('%Y-%m-%d %H:%M')}", "", "## Buy", ""]
    if missing:
        lines += [f"{CHECKBOX}{item}" for item in missing]
    else:
        lines.append("_the fridge looks full_")
    lines += ["", "## Confirmed in the photo", ""]
    lines += [f"- {item}" for item in present] if present else ["_nothing the model would swear to_"]
    return "\n".join(lines) + "\n"
