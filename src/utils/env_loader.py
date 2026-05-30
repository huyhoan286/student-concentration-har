"""Nạp biến môi trường từ file .env ở thư mục gốc project."""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_env(project_root: Path | None = None) -> bool:
    """
    Đọc .env vào os.environ (không ghi đè biến đã có sẵn).

    Returns:
        True nếu tìm thấy và đọc được file .env.
    """
    root = project_root or get_project_root()
    env_file = root / ".env"
    if not env_file.is_file():
        return False

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return True
    except ImportError:
        return _load_env_manual(env_file)


def _load_env_manual(env_file: Path) -> bool:
    loaded = False
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded
