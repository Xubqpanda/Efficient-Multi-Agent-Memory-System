# experiments/run_experiment.py
#!/usr/bin/env python3
"""
experiments/run_experiment.py

将 benchmark config 与 method config 解耦，支持任意组合批量运行。

用法：
    # 单次运行
    python experiments/run_experiment.py \
        --benchmark experiments/configs/benchmarks/frontierscience.yaml \
        --method    experiments/configs/methods/noagent_emptymemory.yaml

    # 临时覆盖某个参数（调试用）
    python experiments/run_experiment.py \
        --benchmark experiments/configs/benchmarks/frontierscience.yaml \
        --method    experiments/configs/methods/noagent_emptymemory.yaml \
        --override  tracks.olympiad.limit=2 tracks.olympiad.num_trials=2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path, benchmark_name: str, exp_name: str) -> logging.Logger:
    """
    同时输出到控制台和日志文件。
    日志文件路径：experiments/logs/{benchmark}/{exp_name}/{timestamp}.log
    """
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / benchmark_name / exp_name / f"{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("emams")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台：INFO 及以上
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # 文件：DEBUG 及以上（完整记录）
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Log file → {log_path}")
    return logger


# ── Config 工具 ───────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(benchmark_cfg: dict, method_cfg: dict) -> dict:
    """将 benchmark 和 method 两个 config 合并为统一的运行时 config。"""
    return {**benchmark_cfg, **method_cfg}


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """key.sub=value 形式的覆盖，value 经 YAML 解析。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override 格式错误（需要 key=value）: {item}")
        key_path, raw = item.split("=", 1)
        val  = yaml.safe_load(raw)
        node = cfg
        for k in key_path.split(".")[:-1]:
            node = node.setdefault(k, {})
        node[key_path.split(".")[-1]] = val
    return cfg


# ── Benchmark runners ─────────────────────────────────────────────────────────

def run_frontierscience(cfg: dict, logger: logging.Logger):
    benchmark_root = REPO_ROOT / "experiments" / "benchmarks" / "FrontierScience"
    sys.path.insert(0, str(benchmark_root))

    from benchmarks.FrontierScience.src.data_loader import FrontierScienceDataset
    from benchmarks.FrontierScience.src.evaluator   import FrontierScienceEvaluator

    # 数据
    data_path = REPO_ROOT / cfg["benchmark"]["data_path"]
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        sys.exit(1)

    logger.info(f"Loading: {data_path}")
    dataset = FrontierScienceDataset(str(data_path))
    stats   = dataset.get_statistics()
    logger.info(f"Dataset — olympiad={stats['olympiad_problems']}  research={stats['research_problems']}")

    # Evaluator
    exp        = cfg["experiment"]
    model_cfg  = cfg["model"]
    out_cfg    = cfg["output"]
    output_dir = REPO_ROOT / out_cfg["dir"] / cfg["benchmark"]["name"] / exp["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = FrontierScienceEvaluator(
        dataset=dataset,
        model=model_cfg["solver"],
        judge_model=model_cfg["judge"],
        reasoning_effort=model_cfg.get("reasoning_effort"),
        output_dir=str(output_dir),
        verbose=out_cfg.get("verbose", False),
        max_workers=out_cfg.get("max_workers", 1),
    )

    results = {}

    # Olympiad
    oly = cfg["tracks"].get("olympiad", {})
    if oly.get("enabled", False):
        logger.info("=" * 55)
        logger.info(f"OLYMPIAD  trials={oly['num_trials']}  "
                    f"limit={oly.get('limit')}  subject={oly.get('subject')}")
        logger.info("=" * 55)
        r = evaluator.evaluate_olympiad(
            subject=oly.get("subject"),
            limit=oly.get("limit"),
            num_trials=oly["num_trials"],
        )
        results["olympiad"] = r
        logger.info(f"accuracy : {r['accuracy']:.2%}  ({r['correct']}/{r['total']})")
        logger.info(f"runtime  : {r['runtime_seconds']:.1f}s  "
                    f"({r['avg_seconds_per_trial']:.1f}s/trial)")

    # Research
    res = cfg["tracks"].get("research", {})
    if res.get("enabled", False):
        logger.info("=" * 55)
        logger.info(f"RESEARCH  trials={res['num_trials']}  "
                    f"limit={res.get('limit')}  threshold={res.get('success_threshold', 7.0)}")
        logger.info("=" * 55)
        r = evaluator.evaluate_research(
            subject=res.get("subject"),
            limit=res.get("limit"),
            num_trials=res["num_trials"],
            success_threshold=res.get("success_threshold", 7.0),
        )
        results["research"] = r
        logger.info(f"accuracy   : {r['accuracy']:.2%}  ({r['successful']}/{r['total']})")
        logger.info(f"avg rubric : {r['avg_rubric_score']:.2f}/10")
        logger.info(f"runtime    : {r['runtime_seconds']:.1f}s  "
                    f"({r['avg_seconds_per_trial']:.1f}s/trial)")

    # 汇总 JSON
    ts           = time.strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump({"experiment": exp, "model": model_cfg,
                   "timestamp": ts, **results}, f, indent=2)
    logger.info(f"结果保存 → {summary_path}")


# ── Benchmark 注册表 ──────────────────────────────────────────────────────────

BENCHMARK_RUNNERS = {
    "frontierscience": run_frontierscience,
    # "hle_verified": run_hle_verified,
}


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EMAMS experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmark", required=True,
                        help="benchmark config 路径")
    parser.add_argument("--method",    required=True,
                        help="method config 路径")
    parser.add_argument("--override",  nargs="*", default=[],
                        metavar="key=value",
                        help="临时覆盖 config 中的值，如 tracks.olympiad.limit=2")
    args = parser.parse_args()

    # 合并 config
    cfg = merge_configs(load_yaml(args.benchmark), load_yaml(args.method))
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    exp            = cfg["experiment"]
    benchmark_name = cfg["benchmark"]["name"]
    log_dir        = REPO_ROOT / "experiments" / "logs"

    # 初始化 logger（此后所有输出都走 logger）
    logger = setup_logging(log_dir, benchmark_name, exp["name"])

    logger.info("=" * 55)
    logger.info(f"Experiment : {exp['name']}")
    logger.info(f"Benchmark  : {benchmark_name}")
    logger.info(f"Memory     : {exp['memory_method']}")
    logger.info(f"Framework  : {exp['agent_framework']}")
    logger.info(f"Solver     : {cfg['model']['solver']}")
    logger.info(f"Judge      : {cfg['model']['judge']}")
    logger.info("=" * 55)

    # 分发
    if benchmark_name not in BENCHMARK_RUNNERS:
        logger.error(f"未知 benchmark: {benchmark_name}，可选: {list(BENCHMARK_RUNNERS.keys())}")
        sys.exit(1)

    BENCHMARK_RUNNERS[benchmark_name](cfg, logger)


if __name__ == "__main__":
    main()