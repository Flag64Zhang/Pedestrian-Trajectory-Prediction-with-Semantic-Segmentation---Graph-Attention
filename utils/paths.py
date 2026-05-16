from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 项目根目录（utils 的上一级），作为所有相对路径的基准。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_project_env() -> Path:
    """将工作目录切换到项目根，并确保可导入项目内模块。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def resolve_path(path: str | Path) -> Path:
    """将配置或参数中的相对路径解析为基于项目根目录的绝对路径。"""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def to_rel_path(path: str | Path) -> str:
    """转为相对项目根的路径字符串，便于日志与配置展示。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """从 YAML 加载配置（路径相对于项目根）。"""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load configs.") from exc
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """解析配置中的路径字段，保持 YAML 内仍为相对路径写法。"""
    data = cfg.get("data", {})
    for key in ("train_dir", "val_dir"):
        if data.get(key):
            data[key] = str(to_rel_path(resolve_path(data[key])))
    scene_dir: Optional[str] = data.get("scene_dir")
    if scene_dir:
        data["scene_dir"] = str(to_rel_path(resolve_path(scene_dir)))

    train = cfg.get("train", {})
    if train.get("save_dir"):
        train["save_dir"] = str(to_rel_path(resolve_path(train["save_dir"])))

    eval_cfg = cfg.get("eval", {})
    if eval_cfg.get("checkpoint"):
        eval_cfg["checkpoint"] = str(to_rel_path(resolve_path(eval_cfg["checkpoint"])))

    vis = cfg.get("visualize", {})
    if vis.get("output_dir"):
        vis["output_dir"] = str(to_rel_path(resolve_path(vis["output_dir"])))

    return cfg
