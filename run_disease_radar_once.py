"""Run exactly one missing-disease experiment and draw a paper-style radar figure.

The script intentionally uses a dedicated result directory and writes a completion
marker.  Once a successful run has completed, a second invocation exits without
starting another experiment.  Remove the dedicated result directory only when an
intentional fresh run is required.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

from config import Config
from draw_radar import (
    aggregate_results,
    apply_paper_category_mapping,
    discover_csv_files,
    draw_radar,
    filter_complete_categories,
    load_category_results,
    resolve_categories,
    resolve_methods,
)
from utils.experiment import run_experiment


ROOT = Path(__file__).resolve().parent
METRICS = ["auc", "aupr", "recall@10"]


def _absolute_from_root(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _prepare_single_run_directory(results_dir: Path) -> tuple[Path, Path]:
    completed = results_dir / ".single_run_completed.json"
    running = results_dir / ".single_run_running.json"

    if completed.exists():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        raise SystemExit(
            "This disease-only experiment has already completed once.\n"
            f"Completed at: {payload.get('completed_at', 'unknown')}\n"
            f"Seed: {payload.get('random_state', 'unknown')}\n"
            f"Result directory: {results_dir}\n"
            "No second run was started. To intentionally run a fresh experiment, "
            "remove this dedicated result directory first."
        )

    # A previous interrupted attempt does not count as the one successful run.
    # Remove only this dedicated directory, never the project's general results.
    if results_dir.exists():
        stale_files = list(results_dir.rglob("*"))
        if stale_files:
            print(f"检测到未完成的旧运行，清理专用目录: {results_dir}")
            shutil.rmtree(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    running.write_text(
        json.dumps(
            {
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "pid": os.getpid(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return completed, running


def _draw_single_run_figure(config: Config, results_dir: Path) -> tuple[Path, Path, list[str]]:
    category_dir = results_dir / "category_metrics"
    paths = discover_csv_files([str(category_dir)])
    if not paths:
        raise RuntimeError(f"No category metric CSV was generated under {category_dir}")

    frame = load_category_results(paths)
    if "mode" in frame.columns:
        frame = frame[frame["mode"].astype(str) == "disease"]
    protocol = str(config.eval_protocol).strip().lower()
    if "protocol" in frame.columns:
        frame = frame[frame["protocol"].astype(str).str.lower() == protocol]
    if frame.empty:
        raise RuntimeError(f"No disease/{protocol} category rows remain for plotting")

    frame = apply_paper_category_mapping(frame)
    aggregate = aggregate_results(frame, METRICS, aggregation="mean")
    methods = resolve_methods(aggregate["method"], requested=[str(config.radar_method_name)])
    categories = resolve_categories(
        aggregate["category"],
        requested=list(getattr(config, "radar_class_order", [])),
    )
    aggregate, categories = filter_complete_categories(
        aggregate=aggregate,
        methods=methods,
        categories=categories,
        metrics=METRICS,
        missing_policy="drop",
    )
    aggregate = aggregate[
        aggregate["method"].isin(methods) & aggregate["category"].isin(categories)
    ].copy()

    figure_path = results_dir / "disease_icd11_radar_once.png"
    table_path = results_dir / "disease_icd11_radar_once_data.csv"
    draw_radar(
        aggregate=aggregate,
        methods=methods,
        categories=categories,
        metrics=METRICS,
        output=figure_path,
        table_output=table_path,
        language="en",
        radial_min=0.4,
        radial_max=1.0,
        title="",
        mode_label="",
        style="example",
        panel_label="c",
    )
    return figure_path, table_path, categories


@hydra.main(version_base=None, config_path="conf", config_name="disease_radar_once")
def main(cfg: DictConfig) -> None:
    os.chdir(ROOT)
    config = Config.from_omegaconf(cfg)

    modes = [str(mode).strip().lower() for mode in config.experiment_modes]
    if modes != ["disease"]:
        raise ValueError(
            "The standalone entry only permits experiment_modes=[disease]. "
            f"Received: {modes}"
        )
    protocol = str(config.eval_protocol).strip().lower()
    if protocol not in {
        "category_stratified_holdout",
        "radar_holdout",
        "stratified_holdout",
        "independent_test",
        "train_test",
        "external_test",
    }:
        raise ValueError(
            "The standalone entry permits one category-stratified holdout or "
            "one independent train-test run"
        )
    if len(config.graph_ablation_configs) != 1:
        raise ValueError(
            "The standalone entry requires exactly one graph_ablation_configs entry "
            "to prevent repeated experiments."
        )

    if int(config.random_state) < 0:
        config.random_state = secrets.randbelow(2_000_000_000) + 1
    config.wandb_run_index = 1

    results_dir = _absolute_from_root(config.results_dir)
    completed_marker, running_marker = _prepare_single_run_directory(results_dir)

    print("=" * 72)
    print("单次 missing-disease ICD-11 类别雷达统计")
    print(f"随机种子: {config.random_state}")
    print(f"输出目录: {results_dir}")
    print(f"评估协议: {protocol}")
    print("本入口只执行 1 个 mode × 1 个协议 × 1 个模型配置。")
    print("=" * 72)

    try:
        run_experiment(config)
        figure_path, table_path, categories = _draw_single_run_figure(config, results_dir)

        completion_payload = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "random_state": int(config.random_state),
            "mode": "disease",
            "protocol": protocol,
            "method": str(config.radar_method_name),
            "categories": categories,
            "figure": str(figure_path),
            "table": str(table_path),
        }
        completed_marker.write_text(
            json.dumps(completion_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("=" * 72)
        print("单次运行完成。完成标记已写入，后续调用不会再次训练。")
        print(f"雷达图: {figure_path}")
        print(f"绘图数据: {table_path}")
        print("=" * 72)
    except Exception:
        failure_path = results_dir / ".single_run_failed.json"
        failure_path.write_text(
            json.dumps(
                {
                    "failed_at": datetime.now().isoformat(timespec="seconds"),
                    "random_state": int(config.random_state),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    finally:
        running_marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
