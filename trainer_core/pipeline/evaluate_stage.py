from __future__ import annotations

import json
import os
from pathlib import Path

from trainer_core.config.loader import load_config
from trainer_core.dataprep.outputs import organize_training_outputs
from trainer_core.evaluation import extras, reports, validate
from trainer_core.pipeline.train_stage import (
    load_model_from_weights,
    load_persisted_train_result,
    resolve_baseline_model,
)
from trainer_core.plugins.replay import build_or_update_replay_set


def _delete_unused_folders() -> None:
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return
    for folder in runs_dir.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def _print_export_guidance(train_output_dir: Path, experiment_name: str) -> None:
    print("\n" + "=" * 70)
    print("Training complete! Next steps:")
    print("=" * 70)
    print("\nTo make this run the baseline for future comparisons:\n")
    print(f"  python tools/export_baseline.py --run-dir {train_output_dir}")
    print("\nThen track it with DVC:\n")
    print("  dvc add models/current_best/best.pt models/current_best/metadata.yaml")
    print("  dvc push")
    print("  git add models/current_best/best.pt.dvc models/current_best/metadata.yaml.dvc")
    print(f"  git commit -m \"Update baseline to {experiment_name}\"")
    print("\n" + "=" * 70 + "\n")


def run_evaluate_stage(args, train_result=None, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    val_split = float(getattr(args, "val_split", cfg.prepare.val_split))
    split_evaluate_mode = train_result is None

    if split_evaluate_mode:
        persisted = load_persisted_train_result()
        model, _display_name = load_model_from_weights(
            persisted.best_weights_path,
            metadata_override=persisted.reload_metadata,
        )
        if model is None:
            raise FileNotFoundError(
                f"Could not load trained model from persisted path: {persisted.best_weights_path}"
            )
        experiment_name = persisted.experiment_name
        train_output_dir = persisted.train_output_dir
        test_path = persisted.test_path
        training_path = persisted.training_path
        image_size = int(persisted.image_size)
        train_epochs = int(persisted.train_epochs)
        baseline_weights_path = cfg.evaluation.baseline_weights_path
        if baseline_weights_path is None:
            baseline_weights_path = persisted.baseline_weights_path
        fallback_checkpoint = persisted.fallback_checkpoint
        finetune_weights_path = persisted.finetune_weights_path
        params = cfg.model_dump()
    else:
        model = train_result.model
        experiment_name = train_result.experiment_name
        train_output_dir = train_result.train_output_dir
        test_path = train_result.test_path
        training_path = train_result.training_path
        image_size = int(train_result.image_size)
        train_epochs = int(train_result.train_epochs)
        baseline_weights_path = train_result.baseline_weights_path
        fallback_checkpoint = train_result.fallback_checkpoint
        finetune_weights_path = train_result.finetune_weights_path
        params = train_result.params

    evaluation_output_dir = Path("results_comparison") if split_evaluate_mode else train_output_dir
    evaluation_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, baseline_display_name = resolve_baseline_model(
        baseline_weights_path,
        fallback_checkpoint,
        finetune_weights_path=finetune_weights_path,
    )

    baseline_results = None
    if baseline_model is not None:
        _, baseline_results = validate.evaluate_and_log_model_results(
            model=baseline_model,
            model_name=baseline_display_name or "baseline",
            test_path=test_path,
            image_size=image_size,
            output_dir=evaluation_output_dir,
            val_split=val_split,
            train_epochs=0,
            is_original=True,
            baseline_model=None,
        )

    retrained_metadata, _ = validate.evaluate_and_log_model_results(
        model=model,
        model_name=experiment_name,
        test_path=test_path,
        image_size=image_size,
        output_dir=evaluation_output_dir,
        val_split=val_split,
        train_epochs=train_epochs,
        baseline_model=baseline_model,
        baseline_display_name=baseline_display_name,
        baseline_results=baseline_results,
    )

    organize_training_outputs(evaluation_output_dir, training_path, test_path, retrained_metadata)

    if baseline_model is not None:
        extras.generate_side_by_side_comparisons(
            original_model=baseline_model,
            retrained_model=model,
            test_img_dir=test_path / "val" / "images",
            output_dir=evaluation_output_dir,
        )

    data_cfg = params.get("data", {}) if isinstance(params, dict) else {}
    custom_classes = list(data_cfg.get("custom_classes") or [])
    class_mapping_config = dict(data_cfg.get("class_mapping") or {})
    raw_test_path = Path("raw_data/test")

    if class_mapping_config and raw_test_path.exists():
        has_merged = any(
            src != target
            for target, sources in class_mapping_config.items()
            for src in (sources if isinstance(sources, list) else [sources])
        )
        if has_merged:
            merged_class_results: list[tuple[str, dict]] = []
            if baseline_model is not None:
                baseline_merged = extras.evaluate_merged_class_subsets(
                    baseline_model,
                    baseline_display_name or "baseline",
                    test_path,
                    raw_test_path,
                    class_mapping_config,
                    custom_classes,
                    imgsz=image_size,
                )
                if baseline_merged:
                    merged_class_results.append((baseline_display_name or "baseline", baseline_merged))

            trained_merged = extras.evaluate_merged_class_subsets(
                model,
                experiment_name,
                test_path,
                raw_test_path,
                class_mapping_config,
                custom_classes,
                imgsz=image_size,
            )
            if trained_merged:
                merged_class_results.append((experiment_name, trained_merged))

            if merged_class_results:
                os.makedirs("./results_comparison", exist_ok=True)
                reports.write_merged_class_results("./results_comparison", merged_class_results)

                metrics_json_path = Path("metrics.json")
                if metrics_json_path.exists():
                    with open(metrics_json_path, "r", encoding="utf-8") as f:
                        metrics_data = json.load(f)
                    for src_class, m in trained_merged.items():
                        tgt = m["target_class"]
                        for k in ("ap50", "ap", "precision", "recall", "f1_score", "n_objects"):
                            metrics_data[f"{src_class}_as_{tgt}_{k}"] = m[k]
                    with open(metrics_json_path, "w", encoding="utf-8") as f:
                        json.dump(metrics_data, f, indent=4)

                results_csv = "./results_comparison/results.csv"
                if os.path.exists(results_csv):
                    reports.create_formatted_table(results_csv)

    auto_replay_cfg = (params.get("prepare", {}) or {}).get("auto_replay", {}) if isinstance(params, dict) else {}
    if auto_replay_cfg and auto_replay_cfg.get("enabled", False):
        build_or_update_replay_set(
            model=model,
            training_path=training_path,
            train_output_dir=train_output_dir,
            config=auto_replay_cfg,
        )

    if not split_evaluate_mode:
        _delete_unused_folders()
    _print_export_guidance(train_output_dir, experiment_name)
