"""Load src/config.yaml and resolve its ${a.b.c} references."""

import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "src" / "config.yaml"
LOCAL_CONFIG_NAME = "config.local.yaml"
_REF = re.compile(r"\$\{([^}]+)\}")


def _lookup(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _resolve(value: Any, cfg: dict) -> Any:
    if isinstance(value, str):
        # Repeat so a reference pointing at another reference still resolves.
        for _ in range(10):
            new = _REF.sub(lambda m: str(_lookup(cfg, m.group(1))), value)
            if new == value:
                return new
            value = new
        raise ValueError(f"unresolved or cyclic reference in {value!r}")
    if isinstance(value, dict):
        return {k: _resolve(v, cfg) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, cfg) for v in value]
    return value


def _merge(base: dict, override: dict) -> dict:
    """Recursively overlay `override` on `base`; only the keys it names are replaced."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Return the parsed config with every ${...} reference substituted.

    An untracked config.local.yaml beside the config is overlaid first. It exists for the
    one thing that legitimately differs between machines - where the external checkouts
    live - so that moving machines never means editing (and committing) the tracked file.
    Protocol values do not belong in it.
    """
    import yaml

    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    local = path.with_name(LOCAL_CONFIG_NAME)
    if local.exists():
        _merge(cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})
    return _resolve(cfg, cfg)


def results_dir(cfg: dict, *parts: str) -> Path:
    """Absolute path under results/, creating it if needed."""
    path = ROOT / cfg["paths"]["results"]
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path
