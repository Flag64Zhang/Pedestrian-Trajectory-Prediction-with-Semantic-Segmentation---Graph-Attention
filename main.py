from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from utils.paths import PROJECT_ROOT, setup_project_env

ROOT = PROJECT_ROOT

_COMMANDS = {
    "preprocess": "preprocess_eth_ucy",
    "train": "train",
    "eval": "eval",
    "visualize": "visualize",
}


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    command: str | None
    argv: tuple[str, ...] = ()


MENU: tuple[MenuItem, ...] = (
    MenuItem("1", "数据预处理", "preprocess", (
        "--eth_dir", "data/ewap_dataset",
        "--ucy_dir", "data/crowds/data",
        "--out_dir", "data/processed",
    )),
    MenuItem("2", "模型训练", "train", ("--config", "configs/default.yaml")),
    MenuItem("3", "模型评估", "eval", (
        "--config", "configs/default.yaml",
        "--plot-ablation",
    )),
    MenuItem("4", "轨迹可视化（10张）", "visualize", (
        "--config", "configs/visualize.yaml",
        "--num", "10",
    )),
    MenuItem("0", "退出", None),
)

_MENU_BY_KEY = {item.key: item for item in MENU}


def _load_script_module(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Script not found: {path}")
    # 避免 scripts/visualize.py 与 utils.visualize 模块名冲突及旧 .pyc 缓存。
    for mod_name in (f"scripts.{name}", "utils.plotting", "utils.visualize"):
        sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(f"scripts.{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_command(command: str, argv: list[str]) -> None:
    script_name = _COMMANDS[command]
    setup_project_env()

    old_argv = sys.argv
    sys.argv = [f"main.py {command}", *argv]
    try:
        module = _load_script_module(script_name)
        module.main()
    finally:
        sys.argv = old_argv


def _print_menu() -> None:
    print()
    print("=" * 36)
    print("  行人轨迹预测系统 (CSGAT-Net)")
    print("=" * 36)
    for item in MENU:
        print(f"  {item.key}. {item.label}")
    print("=" * 36)


def _run_menu_item(item: MenuItem) -> bool:
    """执行菜单项。返回 False 表示应退出主循环。"""
    if item.command is None:
        print("已退出。")
        return False

    print(f"\n>>> {item.label}")
    try:
        _run_command(item.command, list(item.argv))
    except KeyboardInterrupt:
        print("\n已中断。")
    except Exception as exc:
        import traceback

        print(f"执行失败: {exc}")
        traceback.print_exc()
    return True


def main() -> None:
    setup_project_env()
    if len(sys.argv) > 1 and sys.argv[1] in _MENU_BY_KEY:
        item = _MENU_BY_KEY[sys.argv[1]]
        if item.command is None:
            return
        _run_command(item.command, list(item.argv))
        return

    while True:
        _print_menu()
        try:
            choice = input("请输入数字: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if choice not in _MENU_BY_KEY:
            print("无效选项，请重新输入。")
            continue

        if not _run_menu_item(_MENU_BY_KEY[choice]):
            break

        input("\n按回车键返回菜单...")


if __name__ == "__main__":
    main()
