import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: str | Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            value = yaml.safe_load(value)
        except Exception:
            pass
        d = updates
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value
    return updates


def load_config(paths: List[str], overrides: List[str] | None = None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for path in paths:
        if not path:
            continue
        data = load_yaml(path)
        deep_update(cfg, data)
    if overrides:
        deep_update(cfg, parse_overrides(overrides))
    return cfg


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
