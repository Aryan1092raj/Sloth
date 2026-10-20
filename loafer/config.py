"""Config: CLI flags > env vars > config file > defaults."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_FILE = Path(os.environ.get("LOAFER_CONFIG", "~/.config/loafer/config.json")).expanduser()

DEFAULT_INPUT = Path("~/loafer/inbox").expanduser()
DEFAULT_OUT = Path("~/loafer/digests").expanduser()
DEFAULT_MODEL = "qwen3:0.6b"
DEFAULT_VL_MODEL = "qwen2.5vl:3b"
# Presets: model is a preference, not a requirement. If it is not installed the
# backend falls back to the best model that is installed locally.
QUALITY_PRESETS = {
    "fast": {"model": "qwen3:0.6b", "temperature": 0.1, "max_tokens": 300, "chunk": False},
    "balanced": {"model": "qwen3:1.7b", "temperature": 0.2, "max_tokens": 400, "chunk": True},
    "deep": {"model": "qwen3:4b", "temperature": 0.3, "max_tokens": 800, "chunk": True},
}

DEFAULT_STAPLES = (
    "milk",
    "eggs",
    "butter",
    "cheese",
    "yogurt",
    "tomatoes",
    "onions",
    "garlic",
    "rice",
    "pasta",
    "cooking oil",
    "coffee",
)

# Tried in order when LOAFER_MODEL is unset and the backend is "auto".
MODEL_PREFERENCE = (
    "qwen3:0.6b",
    "qwen3:1.7b",
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "gemma3:1b",
    "smollm2:1.7b",
)


@dataclass
class Config:
    input_dir: Path = DEFAULT_INPUT
    out_dir: Path = DEFAULT_OUT
    model: str = DEFAULT_MODEL
    vl_model: str = DEFAULT_VL_MODEL
    quality: str = "balanced"
    max_tokens: int = 400
    chunk: bool = True
    backend: str = "auto"  # auto | ollama | llamacpp | heuristic
    ollama_host: str = "http://127.0.0.1:11434"
    llamacpp_url: str = "http://127.0.0.1:8080/v1"
    temperature: float = 0.2
    max_file_bytes: int = 400_000
    max_chars: int = 6_000
    interval: int = 300
    since: str = ""
    staples: tuple = DEFAULT_STAPLES
    extra: dict = field(default_factory=dict)

    @property
    def state_file(self) -> Path:
        return self.out_dir / ".loafer-state.json"


def _env(name: str, default=None):
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def load_config(**overrides) -> Config:
    """Build a Config. None overrides fall back to env, then config file, then defaults."""
    file_cfg: dict = {}
    if CONFIG_FILE.exists():
        try:
            file_cfg = json.loads(CONFIG_FILE.read_text())
        except (ValueError, OSError):
            # ValueError covers json.JSONDecodeError and UnicodeDecodeError
            file_cfg = {}

    def pick(key, env_name, default):
        value = overrides.get(key)
        if value is not None:
            return value
        env_value = _env(env_name)
        if env_value is not None:
            return env_value
        return file_cfg.get(key, default)

    quality = str(pick("quality", "LOAFER_QUALITY", "balanced"))
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])
    explicit_model = overrides.get("model") or _env("LOAFER_MODEL")

    cfg = Config(
        input_dir=Path(pick("input_dir", "LOAFER_INPUT", DEFAULT_INPUT)).expanduser(),
        out_dir=Path(pick("out_dir", "LOAFER_OUT", DEFAULT_OUT)).expanduser(),
        model=explicit_model or preset["model"],
        vl_model=pick("vl_model", "LOAFER_VL_MODEL", DEFAULT_VL_MODEL),
        backend=pick("backend", "LOAFER_BACKEND", "auto"),
        ollama_host=pick("ollama_host", "LOAFER_OLLAMA_HOST", "http://127.0.0.1:11434"),
        llamacpp_url=pick("llamacpp_url", "LOAFER_LLAMACPP_URL", "http://127.0.0.1:8080/v1"),
        temperature=float(pick("temperature", "LOAFER_TEMPERATURE", preset["temperature"])),
        max_tokens=int(pick("max_tokens", "LOAFER_MAX_TOKENS", preset["max_tokens"])),
        quality=quality,
        chunk=bool(preset["chunk"]),
        max_file_bytes=int(pick("max_file_bytes", "LOAFER_MAX_FILE_BYTES", 400_000)),
        max_chars=int(pick("max_chars", "LOAFER_MAX_CHARS", 6_000)),
        interval=int(pick("interval", "LOAFER_INTERVAL", 300)),
        since=str(pick("since", "LOAFER_SINCE", "")),
    )
    staples_env = _env("LOAFER_STAPLES")
    if staples_env:
        cfg.staples = tuple(s.strip() for s in staples_env.split(",") if s.strip())
    elif file_cfg.get("staples"):
        cfg.staples = tuple(str(s).strip() for s in file_cfg["staples"] if str(s).strip())
    return cfg
    return cfg
