#!/usr/bin/env bash
# eval_periodic_ckpts.sh — 对 periodic checkpoint 逐个跑 infer，汇总结果
# 用法: bash scripts/eval_periodic_ckpts.sh <train_run_dir> <profile> <runs>
# 例:   bash scripts/eval_periodic_ckpts.sh runs/v9p2-periodic/train_20260224_101533 v9p2 3

set -euo pipefail

TRAIN_DIR="${1:?用法: $0 <train_run_dir> <profile> <runs>}"
PROFILE="${2:-v9p2}"
RUNS="${3:-3}"

PERIODIC_DIR="$TRAIN_DIR/models/forest_a/periodic"
BEST_PT="$TRAIN_DIR/models/forest_a/cnn-ddqn.pt"
RESULTS_DIR="$TRAIN_DIR/eval_periodic"

if [ ! -d "$PERIODIC_DIR" ]; then
  echo "ERROR: $PERIODIC_DIR not found"; exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=== Evaluating periodic checkpoints ==="
echo "Train dir : $TRAIN_DIR"
echo "Profile   : $PROFILE"
echo "Runs      : $RUNS"
echo "Checkpoints: $(ls "$PERIODIC_DIR"/ep*.pt | wc -l)"
echo ""

# 评测函数：复制 ckpt -> cnn-ddqn.pt，跑 infer
eval_ckpt() {
  local ckpt_path="$1"
  local label="$2"
  local out_name="v9p2-periodic-eval-${label}"

  echo "[$(date +%H:%M:%S)] Evaluating $label ..."

  # 临时替换 best checkpoint
  cp "$BEST_PT" "$BEST_PT.bak"
  cp "$ckpt_path" "$BEST_PT"

  # 跑 infer (short + long)
  conda run --cwd /home/sun/phdproject/dqn/RL_sac \
    -n ros2py310 python -m forest_vehicle_dqn.cli.infer \
    --profile "$PROFILE" \
    --models "$TRAIN_DIR" \
    --out "$out_name" \
    --runs "$RUNS" \
    --skip-baselines \
    2>&1 | tee "$RESULTS_DIR/${label}.log"

  # 恢复 best checkpoint
  mv "$BEST_PT.bak" "$BEST_PT"

  echo "[$(date +%H:%M:%S)] Done: $label"
  echo ""
}

# 逐个 periodic ckpt 评测
for pt in $(ls "$PERIODIC_DIR"/ep*.pt | sort); do
  ep_label=$(basename "$pt" .pt)  # e.g. ep00010
  eval_ckpt "$pt" "$ep_label"
done

# 也评测 best checkpoint
eval_ckpt "$BEST_PT" "best"

echo ""
echo "=== All evaluations done ==="
echo "Results in: $RESULTS_DIR/"
echo ""

# 汇总结果
echo "=== Summary ==="
echo "label | suite | SR | path_m | time_s"
for logf in $(ls "$RESULTS_DIR"/*.log | sort); do
  label=$(basename "$logf" .log)
  # 从 infer 日志提取 KPI 行
  grep -E "KPI|success_rate|avg_path_length|path_time" "$logf" \
    | sed "s/^/$label | /" || true
done
