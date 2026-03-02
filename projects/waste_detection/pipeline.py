from __future__ import annotations

from trainer_core.cli import main as core_main
from trainer_core.pipeline.evaluate_stage import run_evaluate_stage
from trainer_core.pipeline.prepare_stage import run_prepare_stage
from trainer_core.pipeline.train_stage import run_train_stage


def run_train_eval_stage(args):
    train_result = run_train_stage(args)
    run_evaluate_stage(args, train_result=train_result)
    return train_result


def run_all_stages(args):
    run_prepare_stage(args)
    train_result = run_train_stage(args)
    run_evaluate_stage(args, train_result=train_result)


def main():
    return core_main()


if __name__ == "__main__":
    raise SystemExit(main())
