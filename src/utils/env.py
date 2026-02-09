import platform
import subprocess
import sys
from typing import Dict

import torch


def _get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def get_env_info() -> Dict[str, str]:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "git_commit": _get_git_commit(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = "unknown"
    try:
        import timm

        info["timm"] = timm.__version__
    except Exception:
        info["timm"] = "unknown"
    return info
