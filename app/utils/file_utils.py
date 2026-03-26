from __future__ import annotations

from pathlib import Path
from typing import Iterable


def save_uploaded_files(upload_dir: str, uploaded_files: Iterable) -> list[str]:
    base = Path(upload_dir)
    base.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for uploaded in uploaded_files:
        target = base / uploaded.name
        target.write_bytes(uploaded.getbuffer())
        saved_paths.append(str(target))
    return saved_paths
