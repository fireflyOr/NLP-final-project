"""One CSV with every reported score: mean AUROC (classification) / R^2 (regression) per dataset, arm and learner.

    python paper/make_results_csv.py --out results/scores_per_dataset.csv
"""
import argparse
import glob
import json
import os

import pandas as pd

RUNS = {"final": "results/runs_tarstack", "earlier_full": "results/runs_full", "cross_fitted": "results/runs_xf"}
LEARNER = {"light": "LightGBM (default)", "lgbt": "LightGBM (tuned)", "cat": "CatBoost", "late": "late fusion"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default="results/scores_per_dataset.csv")
    args = a.parse_args()
    frames = []
    for run, d in RUNS.items():
        df = pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))])
        df["run"] = run
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["learner"] = df["learner"].fillna(df["model"]) if "learner" in df else df["model"]
    df["learner"] = df["learner"].map(LEARNER).fillna(df["learner"])
    out = (df.groupby(["run", "dataset", "course_metric", "learner", "encoder"])["course_score"]
             .agg(mean="mean", std="std", n_folds="count").reset_index()
             .rename(columns={"course_metric": "metric", "encoder": "arm"}))
    out["task"] = out["dataset"].str[:3].map({"BIN": "binary", "MUL": "multiclass", "REG": "regression"})
    out = out[["run", "dataset", "task", "metric", "learner", "arm", "mean", "std", "n_folds"]]
    out.sort_values(["run", "learner", "arm", "dataset"]).round(4).to_csv(args.out, index=False)
    print(f"{len(out)} rows -> {args.out}")


if __name__ == "__main__":
    main()
