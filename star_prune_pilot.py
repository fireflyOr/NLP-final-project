"""TabSTAR + SAE concept pruning pilot.

    python star_prune_pilot.py run --datasets BIN_TEXT_FAKE_JOB_POSTING --folds 0 1 2 --out_dir runs_star
    python sae_pilot.py summarize --out_dir runs_star --baselines tabstar-p2 tabstar

Each fold fine-tunes TabSTAR once and writes four results: tabstar, tabstar-p2, tabstar-prune, tabstar-rand
(see multabench/star/prune_star.py). The key comparison is tabstar-prune vs tabstar-p2 (same retraining, with and
without pruning), and vs tabstar-rand (same amount of pruning, random concepts).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

VARIANTS = ["tabstar", "tabstar-p2", "tabstar-prune", "tabstar-rand"]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--size", default="10k")
    r.add_argument("--out_dir", default="runs_star")
    r.add_argument("--time_limit", type=int, default=None, help="TabSTAR fine-tuning budget in seconds (default: its own 10h)")
    args = p.parse_args()

    from multabench.baselines.tabstar_v1 import get_tabstar_paper_mapping
    from multabench.benchmark.load import load_multabench_dataset, split_dataset
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.star.prune_star import run_fold
    from sae_pilot import course_metric

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    for ds_name in args.datasets:
        dataset = load_multabench_dataset(MulTaBenchDatasetID[ds_name])
        for fold in args.folds:
            paths = {v: os.path.join(args.out_dir, f"star_{ds_name}_{v}_{fold}.json") for v in VARIANTS}
            if all(os.path.exists(pth) for pth in paths.values()):
                continue
            split = split_dataset(dataset, fold=fold, size=args.size)
            t0 = time.time()
            ckpt = os.path.join(".tabstar_ckpt", f"{ds_name}_f{fold}_{os.getenv('SLURM_JOB_ID', 'local')}_{os.getpid()}")
            preds, y_true, diag = run_fold(split.x_train.copy(), split.y_train.copy(), split.x_test.copy(),
                                           split.y_test.copy(), is_cls=dataset.is_cls,
                                           pretrain=get_tabstar_paper_mapping(ds_name), device=device,
                                           output_dir=ckpt, time_limit=args.time_limit)
            for v in VARIANTS:
                yp = np.asarray(preds[v])
                if dataset.is_cls and yp.ndim == 2 and yp.shape[1] == 2:
                    yp = yp[:, 1]
                res = {"model": "star", "dataset": ds_name, "encoder": v, "fold": fold, "size": args.size,
                       **course_metric(np.asarray(y_true), yp, dataset.is_cls),
                       "runtime_s": time.time() - t0, "diagnostics": diag}
                with open(paths[v], "w") as f:
                    json.dump(res, f, indent=1, default=str)
                print(f"{ds_name} fold {fold} {v:14s} {res['course_metric']} {res['course_score']:.4f}", flush=True)
            print(f"   pruning: {json.dumps(diag['pruning'])}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
