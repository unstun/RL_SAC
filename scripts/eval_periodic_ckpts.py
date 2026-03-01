#!/usr/bin/env python3
"""对 periodic checkpoint 逐个跑 infer，汇总最佳 epoch。

用法:
  python scripts/eval_periodic_ckpts.py \
    --train-dir runs/v9p2-periodic/train_20260224_101533 \
    --profile v9p2 --runs 3
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


def run_infer(
    profile: str,
    models_dir: str,
    out_name: str,
    runs: int,
) -> Path | None:
    """运行一次 infer.py，返回结果目录。"""
    cmd = [
        sys.executable, "-m", "forest_vehicle_dqn.cli.infer",
        "--profile", profile,
        "--models", models_dir,
        "--out", out_name,
        "--runs", str(runs),
        "--baselines",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  WARNING: infer 返回非零退出码 {result.returncode}")
        return None

    # 找到最新的 infer 结果目录
    out_dir = Path("runs") / out_name
    if not out_dir.exists():
        return None
    latest_txt = out_dir / "latest.txt"
    if latest_txt.exists():
        ts = latest_txt.read_text().strip()
        return out_dir / ts
    # fallback: 取最新子目录
    subdirs = sorted(
        [d for d in out_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return subdirs[-1] if subdirs else None


def parse_kpis(result_dir: Path) -> list[dict]:
    """从 table2_kpis_mean.csv 解析 CNN-DDQN 的指标。"""
    csv_path = result_dir / "table2_kpis_mean.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "CNN-DDQN" in row.get("Algorithm name", ""):
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--profile", default="v9p2")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    train_dir = Path(args.train_dir)
    periodic_dir = train_dir / "models" / "forest_a" / "periodic"
    best_pt = train_dir / "models" / "forest_a" / "cnn-ddqn.pt"

    if not periodic_dir.exists():
        print(f"ERROR: {periodic_dir} 不存在")
        sys.exit(1)

    ckpts = sorted(periodic_dir.glob("ep*.pt"))
    print(f"找到 {len(ckpts)} 个 periodic checkpoint")
    print(f"best checkpoint: {best_pt}")
    print()

    # 收集结果: {label: {suite: {sr, path, time}}}
    all_results: dict[str, list[dict]] = {}

    for ckpt in ckpts:
        label = ckpt.stem  # e.g. ep00010
        out_name = f"v9p2-peval-{label}"
        print(f"=== {label} ({ckpt.name}) ===")

        # 备份 best，替换为当前 ckpt
        best_bak = best_pt.with_suffix(".pt.bak")
        shutil.copy2(best_pt, best_bak)
        shutil.copy2(ckpt, best_pt)

        try:
            rdir = run_infer(args.profile, str(train_dir), out_name, args.runs)
            if rdir:
                kpis = parse_kpis(rdir)
                all_results[label] = kpis
                for k in kpis:
                    env = k["Environment"]
                    sr = k["Success rate"]
                    path_m = k["Average path length (m)"]
                    time_s = k["Path time (s)"]
                    print(f"  {env}: SR={sr}, path={path_m}m, time={time_s}s")
            else:
                print(f"  WARNING: 未找到结果目录")
        finally:
            # 恢复 best checkpoint
            shutil.move(str(best_bak), str(best_pt))
        print()

    # 也评测 best checkpoint 本身
    print("=== best (训练器选出的最优) ===")
    out_name_best = "v9p2-peval-best"
    rdir = run_infer(args.profile, str(train_dir), out_name_best, args.runs)
    if rdir:
        kpis = parse_kpis(rdir)
        all_results["best"] = kpis
        for k in kpis:
            env = k["Environment"]
            sr = k["Success rate"]
            path_m = k["Average path length (m)"]
            time_s = k["Path time (s)"]
            print(f"  {env}: SR={sr}, path={path_m}m, time={time_s}s")
    print()

    # 打印汇总表
    print_summary(all_results)


def print_summary(all_results: dict[str, list[dict]]):
    """打印汇总对比表。"""
    print("=" * 80)
    print("汇总表: 各 checkpoint 的推理指标")
    print("=" * 80)

    suites = ["short", "long"]
    header = "| Checkpoint | Suite | SR | Path(m) | Time(s) |"
    sep = "|------------|-------|------|---------|---------|"
    print(header)
    print(sep)

    best_by_suite: dict[str, tuple[str, float]] = {}

    for label, kpis in sorted(all_results.items()):
        for k in kpis:
            env = k["Environment"]
            suite = "short" if "short" in env else ("long" if "long" in env else "mid")
            sr = k["Success rate"]
            path_m = float(k["Average path length (m)"])
            time_s = k["Path time (s)"]
            print(f"| {label:10s} | {suite:5s} | {sr:4s} | {path_m:7.2f} | {time_s:>7s} |")

            if suite in ("short", "long"):
                key = suite
                if key not in best_by_suite or path_m < best_by_suite[key][1]:
                    if float(sr) >= 0.8:
                        best_by_suite[key] = (label, path_m)

    print()
    print("最优 checkpoint（按 avg_path_length 最短，SR>=80%）:")
    for suite, (label, path_m) in sorted(best_by_suite.items()):
        print(f"  {suite}: {label} (path={path_m:.2f}m)")


if __name__ == "__main__":
    main()
