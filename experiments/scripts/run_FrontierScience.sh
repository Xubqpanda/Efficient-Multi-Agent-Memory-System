#!/bin/bash
# experiments/scripts/run_FrontierScience.sh
#
# 用法：
#   bash experiments/scripts/run_FrontierScience.sh --smoke
#   bash experiments/scripts/run_FrontierScience.sh --method experiments/configs/methods/noagent_emptymemory.yaml
#   bash experiments/scripts/run_FrontierScience.sh --all      # 并行跑所有 methods

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── API Key ───────────────────────────────────────────────────────────────────
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[ERROR] OPENAI_API_KEY 未设置"
    exit 1
fi

# ── 固定：benchmark config ────────────────────────────────────────────────────
BENCHMARK_CFG="experiments/configs/benchmarks/frontierscience.yaml"

# ── 默认值 ────────────────────────────────────────────────────────────────────
SMOKE=false
RUN_ALL=false
METHOD_CFG="experiments/configs/methods/noagent_emptymemory.yaml"
EXTRA_OVERRIDES=()

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)     SMOKE=true;           shift ;;
        --all)       RUN_ALL=true;         shift ;;
        --method)    METHOD_CFG="$2";      shift 2 ;;
        *)           EXTRA_OVERRIDES+=("$1"); shift ;;
    esac
done

# smoke test 小规模 override
SMOKE_OVERRIDES=()
if [ "$SMOKE" = true ]; then
    SMOKE_OVERRIDES=(
        "tracks.olympiad.limit=2"
        "tracks.olympiad.num_trials=2"
        "tracks.research.limit=2"
        "tracks.research.num_trials=2"
        "output.verbose=true"
        "output.max_workers=1"
    )
    echo "[Smoke test mode]"
fi

# ── 单个 method 的运行函数 ────────────────────────────────────────────────────
# 注意：日志由 run_experiment.py 内部的 setup_logging 统一写入
# experiments/logs/{benchmark}/{exp_name}/{timestamp}.log
# 这里不再用 tee，避免日志重复
run_single() {
    local method_cfg="$1"
    local method_name
    method_name=$(basename "$method_cfg" .yaml)

    echo "→ Starting: $method_name"

    local all_overrides=("${SMOKE_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")
    local override_args=()
    if [ ${#all_overrides[@]} -gt 0 ]; then
        override_args=("--override" "${all_overrides[@]}")
    fi

    python experiments/run_experiment.py \
        --benchmark "$BENCHMARK_CFG"  \
        --method    "$method_cfg"     \
        "${override_args[@]}"
}

# ── 执行 ──────────────────────────────────────────────────────────────────────
if [ "$RUN_ALL" = true ]; then
    pids=()
    for method_cfg in experiments/configs/methods/*.yaml; do
        run_single "$method_cfg" &
        pids+=($!)
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || { echo "[WARN] 子进程 $pid 失败"; failed=1; }
    done
    exit $failed
else
    run_single "$METHOD_CFG"
fi