from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[assignment]


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_global_config(path: Path) -> Dict[str, Any]:
    """Load the global TOML configuration."""
    return _load_toml(path)


def load_steps_config(path: Path) -> Dict[str, Any]:
    """Load the step-level TOML configuration."""
    return _load_toml(path)


def resolve_paths(root_cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Resolve core root paths from the global config.

    Returns a dict with keys: dataset_dir, output_dir, manifests_dir, reports_dir, docs_dir, logs_dir.
    """
    from pathlib import Path as _Path

    paths = root_cfg.get("paths", {})
    return {
        "dataset_dir": _Path(paths.get("dataset_dir", "dataset")).resolve(),
        "output_dir": _Path(paths.get("output_dir", "output")).resolve(),
        "docs_dir": _Path(paths.get("docs_dir", "docs")).resolve(),
        "manifests_dir": _Path(paths.get("manifests_dir", "output/manifests")).resolve(),
        "reports_dir": _Path(paths.get("reports_dir", "output/reports")).resolve(),
        "logs_dir": _Path(paths.get("logs_dir", "logs")).resolve(),
    }
