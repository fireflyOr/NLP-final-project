"""Run the method (Joint TAR + text stacking, then a tree learner) on any text-tabular CSV.

    python run_csv.py --train train.csv --test test.csv --target price --out predictions.csv
    python run_csv.py --train data.csv --target label --test-size 0.2 --out predictions.csv   # hold out 20%

What it does, with the same code as the experiments in the paper:
  1. Column types (numerical, categorical, date, free text) are detected by the benchmark's preprocessing.
  2. Each text column: E5-small-v2 fine-tuned with LoRA on the target (Joint TAR), embedding reduced to 30 PCA dims.
  3. Text stacking: an MLP on the frozen E5 embeddings of all text columns predicts the target; its out-of-fold
     predictions (5 folds inside the training rows, identical texts kept in one fold) are added as features.
  4. The learner (default: tuned LightGBM) is fitted on the table columns + the text features.
Everything is fitted on the training rows only; test rows are only transformed and predicted.

Output CSV, one row per test row: `prediction` (regression), or `predicted_label` and one `proba_<class>` column per
class (classification). If the test rows contain the target, AUROC (classification; one-vs-rest macro for multiclass)
or R^2 (regression) is printed.

A GPU is strongly recommended: step 2 fine-tunes a transformer. `--encoder e5-small-tar` runs the Joint TAR baseline
and `--encoder e5-small` the Joint Frozen baseline on the same data, for comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

ENCODERS = {"e5-small-tar-fstack": "TAR + text stacking (this work)",
            "e5-small-tar": "Joint TAR baseline",
            "e5-small": "Joint Frozen baseline"}
LEARNERS = {"lgbt": "LightGBM, tuned on a validation split",
            "cat": "CatBoost, default settings",
            "light": "LightGBM, default settings (the benchmark's learner)"}
MAX_CLASSES_FOR_INT_TARGET = 20


def infer_task(y: pd.Series) -> str:
    """'binary' / 'multiclass' / 'regression' from the target column; raises when an integer target is ambiguous."""
    y = y.dropna()
    n = y.nunique()
    if n < 2:
        raise ValueError("the target has fewer than 2 distinct values")
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "binary" if n == 2 else "multiclass"
    if n == 2:
        return "binary"
    if np.allclose(y, np.round(y)) and n <= MAX_CLASSES_FOR_INT_TARGET:
        raise ValueError(f"the target is integer-valued with {n} distinct values: it could be classes or a count. "
                         f"Pass --task multiclass or --task regression.")
    return "regression"


def labels(y: pd.Series) -> pd.Series:
    """Class labels as strings; integer-valued numbers print without '.0', so 1 and 1.0 are the same class."""
    if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
        v = y.dropna()
        if len(v) and np.allclose(v, np.round(v)):
            return y.round().astype("Int64").astype(str).where(y.notna())
    return y.astype(str).where(y.notna())


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, task: str = "auto", learner: str = "lgbt",
                encoder: str = "e5-small-tar-fstack", e5_batch_size: int = 256,
                e5_overrides: Optional[dict] = None, device=None):
    """Fit on `train`, predict `test`. Returns (predictions DataFrame indexed like `test`, metric dict or None)."""
    import torch
    from multabench.datasets.objects import SupervisedTask
    from multabench.e5.constants import TEXT_ENCODERS
    from multabench.finetune.train_args import E5TrainArgs
    from sae_pilot import _model_cls, course_metric

    if target not in train.columns:
        raise ValueError(f"target column {target!r} not in the training file (columns: {list(train.columns)})")
    n_missing = int(train[target].isna().sum())
    if n_missing:
        print(f"dropping {n_missing} training rows with a missing target")
        train = train[train[target].notna()]
    task = infer_task(train[target]) if task == "auto" else task
    problem = {"binary": SupervisedTask.BINARY, "multiclass": SupervisedTask.MULTICLASS,
               "regression": SupervisedTask.REGRESSION}[task]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and TEXT_ENCODERS[encoder].tune_encoder:
        print("warning: no GPU found; fine-tuning E5 on CPU is very slow")

    enc = TEXT_ENCODERS[encoder]
    e5_kwargs = None
    if enc.tune_encoder:
        e5_kwargs = dict(vars(E5TrainArgs()), batch_size=e5_batch_size)
        e5_kwargs.update(e5_overrides or {})
    model = _model_cls(learner)(problem_type=problem, device=device, verbose=False, tune_e5=enc.tune_encoder,
                                e5_train_kwargs=e5_kwargs, e5_model_name=enc.encoder_name)
    x_train, y_train = train.drop(columns=[target]), train[target]
    if task != "regression":
        y_train = labels(y_train)                    # one label type, so the encoder and the output agree
    model.fit(x_train.copy(), y_train.copy())
    print(f"task: {task}; text columns: {sorted(c for c in model.text_transformers if not c.startswith('__'))}; "
          f"categorical: {sorted(model.categorical_features)}; numerical: {sorted(model.numerical_features)}")
    if not model.text_transformers:
        print("warning: no free-text column detected; the model uses the table columns only")

    x_test = test.drop(columns=[target], errors="ignore")
    pred = np.asarray(model.predict(x_test.copy()))
    if task == "regression":
        out = pd.DataFrame({"prediction": pred}, index=test.index)
    else:
        classes = list(model.target_transformer.classes_)
        probs = np.column_stack([1 - pred, pred]) if pred.ndim == 1 else pred
        assert probs.shape[1] == len(classes), (probs.shape, classes)
        out = pd.DataFrame(probs, columns=[f"proba_{c}" for c in classes], index=test.index)
        out.insert(0, "predicted_label", np.asarray(classes, dtype=object)[probs.argmax(axis=1)])

    metric = None
    if target in test.columns and test[target].notna().any():
        known = test[target].notna()
        if task == "regression":
            metric = course_metric(test.loc[known, target].to_numpy(dtype=float), pred[known.to_numpy()], False)
        else:
            y_test = labels(test[target])
            known &= y_test.isin(classes)
            if (~known).sum():
                print(f"{int((~known).sum())} test rows have a missing label or a class unseen in training; "
                      f"left out of the metric")
            k = known.to_numpy()
            y_idx = model.target_transformer.transform(y_test[known])
            metric = course_metric(y_idx, pred[k], True)
    return out, metric


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--train", required=True, help="training CSV (must contain the target column)")
    a.add_argument("--test", help="CSV to predict; if it has the target column, the score is printed")
    a.add_argument("--test-size", type=float, help="no --test: hold out this fraction of --train as the test set")
    a.add_argument("--target", required=True, help="name of the target column")
    a.add_argument("--task", default="auto", choices=["auto", "binary", "multiclass", "regression"])
    a.add_argument("--learner", default="lgbt", choices=list(LEARNERS), help="; ".join(f"{k}: {v}" for k, v in LEARNERS.items()))
    a.add_argument("--encoder", default="e5-small-tar-fstack", choices=list(ENCODERS),
                   help="; ".join(f"{k}: {v}" for k, v in ENCODERS.items()))
    a.add_argument("--compare", action="store_true",
                   help="also run the Joint TAR baseline on the same fine-tuned E5 and print both scores")
    a.add_argument("--drop-cols", nargs="*", default=[], help="columns to ignore, e.g. row ids")
    a.add_argument("--e5-batch-size", type=int, default=256, help="fine-tuning batch size (lower it on GPU out-of-memory)")
    a.add_argument("--seed", type=int, default=0, help="seed of the --test-size split")
    a.add_argument("--out", default="predictions.csv")
    args = a.parse_args(argv)

    os.environ.setdefault("MULTABENCH_DYNAMIC_PADDING", "1")   # same results as max-length padding, faster
    try:
        import torch
        if torch.cuda.is_available():
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # required by the LoRA fine-tuning code
    except ImportError:
        pass

    train = pd.read_csv(args.train).drop(columns=args.drop_cols)
    if args.target not in train.columns:
        a.error(f"target column {args.target!r} not in {args.train}")
    if args.test:
        test = pd.read_csv(args.test).drop(columns=args.drop_cols, errors="ignore")
    elif args.test_size:
        from sklearn.model_selection import train_test_split
        train = train[train[args.target].notna()]
        task = infer_task(train[args.target]) if args.task == "auto" else args.task
        strat = labels(train[args.target]) if task != "regression" else None
        train, test = train_test_split(train, test_size=args.test_size, random_state=args.seed, stratify=strat)
    else:
        a.error("give --test or --test-size")

    encoders = ["e5-small-tar", args.encoder] if args.compare and args.encoder != "e5-small-tar" else [args.encoder]
    scores = {}
    for enc in encoders:            # in one process the fine-tuned E5 is shared, so only the step after it differs
        out, metric = fit_predict(train, test, args.target, task=args.task, learner=args.learner, encoder=enc,
                                  e5_batch_size=args.e5_batch_size)
        scores[enc] = metric
    out.to_csv(args.out, index_label="row")
    print(f"{len(out)} predictions of {args.encoder} written to {args.out}")
    for enc, metric in scores.items():
        if metric:
            print(f"{enc} ({ENCODERS[enc]}): {metric['course_metric']} {metric['course_score']:.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
