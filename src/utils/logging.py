import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .env import get_env_info
from .io import ensure_dir, save_json, save_yaml


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _safe_git_cmd(args) -> str:
    try:
        out = subprocess.check_output(["git"] + args, stderr=subprocess.DEVNULL)
        return out.decode()
    except Exception:
        return ""


class RunLogger:
    def __init__(self, output_dir: str, exp_name: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = ensure_dir(Path(output_dir) / f"{exp_name}_{ts}")
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self._metrics_file = open(self.metrics_path, "a", encoding="utf-8")

    def log_config(self, cfg: Dict[str, Any]) -> None:
        save_yaml(self.run_dir / "config.yaml", cfg)

    def log_env(self) -> None:
        save_json(self.run_dir / "env.json", get_env_info())

    def log_code_state(self) -> None:
        status = _safe_git_cmd(["status", "--short"])
        diff = _safe_git_cmd(["diff"])
        if status or diff:
            with open(self.run_dir / "code_state.txt", "w", encoding="utf-8") as f:
                f.write("# git status --short\n")
                f.write(status)
                f.write("\n# git diff\n")
                f.write(diff)

    def log_metrics(self, row: Dict[str, Any]) -> None:
        self._metrics_file.write(json.dumps(row) + "\n")
        self._metrics_file.flush()

    def log_summary(self, summary: Dict[str, Any]) -> None:
        save_json(self.run_dir / "summary.json", summary)

    def close(self) -> None:
        try:
            self._metrics_file.close()
        except Exception:
            pass
