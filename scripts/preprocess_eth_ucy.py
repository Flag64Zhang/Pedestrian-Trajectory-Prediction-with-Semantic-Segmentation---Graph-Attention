from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from utils.paths import resolve_path, setup_project_env, to_rel_path


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]


def _write_rows(path: Path, rows: List[Tuple[int, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for frame, pid, x, y in rows:
            f.write(f"{frame} {pid} {x:.6f} {y:.6f}\n")


# 解析 ETH obsmat.txt（frame, id, x, z, y, vx, vz, vy）为 (frame, id, x, y)。
def _parse_eth_obsmat(path: Path) -> List[Tuple[int, int, float, float]]:
    rows: List[Tuple[int, int, float, float]] = []
    for line in _read_lines(path):
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        frame = int(float(parts[0]))
        pid = int(float(parts[1]))
        x = float(parts[2])
        y = float(parts[4])
        rows.append((frame, pid, x, y))
    return rows


# UCY .vsp 文件包含行内注释，先去除再解析。
def _strip_inline_comment(line: str) -> str:
    if " - " in line:
        return line.split(" - ", 1)[0].strip()
    return line.strip()


# 将控制点线性插值为逐帧轨迹。
def _interpolate_points(points: List[Tuple[int, float, float]]) -> List[Tuple[int, float, float]]:
    if len(points) < 2:
        return points

    points = sorted(points, key=lambda p: p[0])
    dense: List[Tuple[int, float, float]] = []
    for (f0, x0, y0), (f1, x1, y1) in zip(points[:-1], points[1:]):
        if f1 <= f0:
            continue
        for f in range(f0, f1 + 1):
            t = (f - f0) / (f1 - f0)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            dense.append((f, x, y))
    return dense


# 解析 UCY .vsp（样条控制点）并插值为逐帧轨迹。
def _parse_ucy_spline(path: Path) -> List[Tuple[int, int, float, float]]:
    lines = _read_lines(path)
    cleaned: List[str] = []
    for ln in lines:
        if not ln:
            continue
        if ln.lstrip().startswith("-"):
            continue
        ln = _strip_inline_comment(ln)
        if ln:
            cleaned.append(ln)
    filtered = cleaned
    if not filtered:
        return []

    idx = 0
    try:
        num_splines = int(float(filtered[idx]))
    except ValueError:
        return []
    idx += 1

    rows: List[Tuple[int, int, float, float]] = []
    for spline_id in range(1, num_splines + 1):
        if idx >= len(filtered):
            break
        try:
            num_pts = int(float(filtered[idx]))
        except ValueError:
            break
        idx += 1

        control_points: List[Tuple[int, float, float]] = []
        for _ in range(num_pts):
            if idx >= len(filtered):
                break
            parts = filtered[idx].split()
            idx += 1
            if len(parts) < 3:
                continue
            x = float(parts[0])
            y = float(parts[1])
            frame = int(float(parts[2]))
            control_points.append((frame, x, y))

        for frame, x, y in _interpolate_points(control_points):
            rows.append((frame, spline_id, x, y))
    return rows


# 收集 UCY 标注文件（.vsp/.txt），排除格式说明文件。
def _collect_ucy_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".txt", ".vsp"}:
            continue
        if path.name == "crowd_file_format.txt":
            continue
        files.append(path)
    return files


# 按场景名称划分 train/val 并输出标准 txt。
def _split_by_scene(
    scenes: Dict[str, List[Tuple[int, int, float, float]]],
    train_scenes: List[str],
    val_scenes: List[str],
    out_dir: Path,
) -> None:
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    for name, rows in scenes.items():
        if name in train_scenes:
            _write_rows(train_dir / f"{name}.txt", rows)
        elif name in val_scenes:
            _write_rows(val_dir / f"{name}.txt", rows)


def main() -> None:
    setup_project_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eth_dir", default="data/ewap_dataset", help="相对项目根的 ETH 数据目录")
    parser.add_argument("--ucy_dir", default="data/crowds/data", help="相对项目根的 UCY 数据目录")
    parser.add_argument("--out_dir", default="data/processed", help="相对项目根的输出目录")
    parser.add_argument(
        "--train_scenes",
        default="seq_eth,seq_hotel,crowds_zara01,crowds_zara02,crowds_zara03,students001",
        help="Comma-separated scene names for training",
    )
    parser.add_argument(
        "--val_scenes",
        default="students003,arxiepiskopi1,uni_examples",
        help="Comma-separated scene names for validation",
    )
    args = parser.parse_args()

    eth_dir = resolve_path(args.eth_dir)
    ucy_dir = resolve_path(args.ucy_dir)
    out_dir = resolve_path(args.out_dir)

    scenes: Dict[str, List[Tuple[int, int, float, float]]] = {}

    # 读取 ETH 两个场景。
    for seq_name in ["seq_eth", "seq_hotel"]:
        obsmat = eth_dir / seq_name / "obsmat.txt"
        if obsmat.exists():
            scenes[seq_name] = _parse_eth_obsmat(obsmat)

    # 读取 UCY 各场景标注。
    for path in _collect_ucy_files(ucy_dir):
        name = path.stem
        rows = _parse_ucy_spline(path)
        if rows:
            scenes[name] = rows

    train_scenes = [s.strip() for s in args.train_scenes.split(",") if s.strip()]
    val_scenes = [s.strip() for s in args.val_scenes.split(",") if s.strip()]

    _split_by_scene(scenes, train_scenes, val_scenes, out_dir)
    print(f"已写入: {to_rel_path(out_dir)}")


if __name__ == "__main__":
    main()
