from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Dict, Any


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_parent_dir(path: Path) -> None:
    """Ensure that the parent directory of `path` exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dictionaries to CSV with the given fieldnames."""
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    """Write a JSON object to disk with UTF-8 encoding."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    """Write plain text (e.g., Markdown) to disk."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
