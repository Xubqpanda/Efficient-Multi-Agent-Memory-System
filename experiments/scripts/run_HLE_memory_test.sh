#!/usr/bin/env bash
# experiments/scripts/run_HLE_memory_test.sh
#
# 三种 Memory 方法的 HLE 适配测试脚本。
#
# 用法：
#   # 正式跑（每种 50 题）
#   bash experiments/scripts/run_HLE_memory_test.sh
#
#   # 冒烟测试（每种 2 题）
#   bash experiments/scripts/run_HLE_memory_test.sh --smoke
#
#   # 只跑某一种（generative / voyager / memorybank）
#   bash experiments/scripts/run_HLE_memory_test.sh --only generative

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ── API 配置 ──────────────────────────────────────────────────────────────────
export OPENAI_API_KEY="sk-eb639510e766dc2868bc1974e678a055f6cba2bb351a74cb2696e46d24d360f3"
export OPENAI_API_BASE="https://gmn.chuangzuoli.com"

# ── 路径 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ENV_CFG="experiments/configs/envs/hle.yaml"
SOLVER_CFG="experiments/configs/solver/single_agent.yaml"
TOOL_CFG="experiments/configs/tool/default.yaml"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
LIMIT=50
ONLY=""
for arg in "$@"; do
    case $arg in
        --smoke) LIMIT=2 ;;
        --only)  shift; ONLY="$1" ;;
    esac
    shift 2>/dev/null || true
done

MODEL="gpt-5.2"
OVERRIDES="evaluation.limit=$LIMIT evaluation.text_only=true model.solver=$MODEL model.judge=$MODEL model.base_url=https://gmn.chuangzuoli.com"

# ── 运行函数 ──────────────────────────────────────────────────────────────────
run_memory() {
    local method=$1
    local mem_cfg="experiments/configs/memory/${method}.yaml"

    echo ""
    echo "============================================================"
    echo "  Memory: $method  |  Limit: $LIMIT  |  Model: $MODEL"
    echo "============================================================"
    echo ""

    python experiments/run_experiment.py \
        --env    "$ENV_CFG" \
        --solver "$SOLVER_CFG" \
        --tool   "$TOOL_CFG" \
        --memory "$mem_cfg" \
        --override $OVERRIDES
}

# ── 执行 ──────────────────────────────────────────────────────────────────────
METHODS=("generative" "voyager" "memorybank")

if [ -n "$ONLY" ]; then
    run_memory "$ONLY"
else
    for method in "${METHODS[@]}"; do
        run_memory "$method"
    done
fi

echo ""
echo "All experiments completed."
