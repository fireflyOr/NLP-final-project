"""Write one benchmark fold as train/test CSVs, to check run_csv.py against the experiment on the same rows.

    python export_fold.py --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --out-dir csv_check
    python run_csv.py --train csv_check/train.csv --test csv_check/test.csv --target target --out csv_check/pred.csv

The target is written as a column named `target`. Score to compare with: results/runs_tarstack/
lgbt_<dataset>_e5-small-tar-fstack_<fold>.json (`course_score`). Reading through a CSV can change column types
(e.g. dates become strings), so expect a close, not bit-identical, score.
"""
import argparse
import os


def main():
    from multabench.benchmark.load import load_multabench_dataset, split_dataset
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    a = argparse.ArgumentParser()
    a.add_argument("--dataset", required=True)
    a.add_argument("--fold", type=int, default=0)
    a.add_argument("--size", default="10k")
    a.add_argument("--out-dir", default="csv_check")
    args = a.parse_args()
    ds = load_multabench_dataset(MulTaBenchDatasetID[args.dataset])
    split = split_dataset(ds, fold=args.fold, size=args.size)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, x, y in (("train", split.x_train, split.y_train), ("test", split.x_test, split.y_test)):
        assert "target" not in x.columns
        x.assign(target=y.to_numpy()).to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
        print(f"{name}: {len(x)} rows -> {args.out_dir}/{name}.csv")
    print(f"task: {ds.task_type}")


if __name__ == "__main__":
    main()
