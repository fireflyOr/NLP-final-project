"""SAE pilot: compare text-column transforms under the same tabular learner.

    python sae_pilot.py run --model light --datasets BIN_TEXT_FAKE_JOB_POSTING MUL_TEXT_WINE_REVIEW REG_TEXT_BOOK_READABILITY
    python sae_pilot.py summarize

Encoders (all frozen E5-small-v2; only the step after embedding differs):
    e5-small           official Joint Frozen baseline: PCA to 30 dims, fitted on the train fold
    e5-small-full      control: all 384 dims (is any gain just "more columns"?)
    e5-small-sae       trained TopK SAE, continuous codes
    e5-small-sae-bin   trained TopK SAE, on/off codes
    e5-small-rsae-bin  control: untrained random TopK SAE, on/off codes (is any gain from learning?)
    e5-small-fsae      small TopK SAE (30 latents, 4 active) trained on the fold's own text, continuous codes
    e5-small-fsae-bin  same, on/off codes
    e5-small-fsae50    small TopK SAE with 50 latents (4 active), continuous codes
    e5-small-fsae50-bin  same, on/off codes
    e5-small-pca50     PCA with 50 components: the fair comparison for the 50-latent SAE
    e5-small-tar       optional reference (LoRA fine-tuned, slow)

Every run is written to <out_dir>/<model>_<dataset>_<encoder>_<fold>.json and skipped if it exists,
so the script can be re-run after a Colab disconnect. Frozen E5 embeddings are cached in memory and
shared across encoders, so each text column is embedded once per fold.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

DEFAULT_DATASETS = ["BIN_TEXT_FAKE_JOB_POSTING", "MUL_TEXT_WINE_REVIEW", "REG_TEXT_BOOK_READABILITY"]
DEFAULT_ENCODERS = ["e5-small", "e5-small-full", "e5-small-sae", "e5-small-sae-bin", "e5-small-rsae-bin"]
BASELINE = "e5-small"


def _model_cls(short: str):
    if short == "light":
        from multabench.baselines.lgbm import LightGBM
        return LightGBM
    if short == "cat":
        from multabench.baselines.catboost import CatBoost
        return CatBoost
    if short == "xgb":
        from multabench.baselines.xgboost import XGBoost
        return XGBoost
    if short == "tabm":
        from multabench.baselines.tabm import TabM
        return TabM
    if short in ("tabpfnv2", "tabpfnv2p5"):
        from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
        return TabPFNv2 if short == "tabpfnv2" else TabPFNv2p5
    raise ValueError(short)


def _install_embedding_cache():
    """Memoize frozen E5 embeddings (keyed by model object, column and texts) inside the text pipeline."""
    import multabench.baselines.preprocessing.text_embeddings as te
    if getattr(te, "_pilot_cache_installed", False):
        return
    original = te.encode_texts_with_e5
    cache: dict = {}

    def cached(texts, col_name, model, tokenizer, device, **kw):
        h = hashlib.sha1("\x1f".join(texts).encode("utf-8", "replace")).hexdigest()
        key = (id(model), col_name, len(texts), h)
        if key not in cache:
            cache[key] = original(texts=texts, col_name=col_name, model=model, tokenizer=tokenizer, device=device, **kw)
        return cache[key]

    te.encode_texts_with_e5 = cached
    te._pilot_cache_installed = True
    te._pilot_cache = cache


def course_metric(y_true: np.ndarray, y_pred: np.ndarray, is_cls: bool) -> dict:
    """The course reports AUROC for classification (one-vs-rest macro for multiclass) and R^2 for regression."""
    from sklearn.metrics import r2_score, roc_auc_score
    if not is_cls:
        return {"course_metric": "r2", "course_score": float(r2_score(y_true, y_pred))}
    if y_pred.ndim == 1:
        return {"course_metric": "auroc", "course_score": float(roc_auc_score(y_true, y_pred))}
    labels = np.arange(y_pred.shape[1])
    present = np.isin(labels, np.unique(y_true))
    probs = y_pred[:, present] / y_pred[:, present].sum(axis=1, keepdims=True)
    return {"course_metric": "auroc_ovr_macro",
            "course_score": float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels[present]))}


def run(args):
    import torch
    from multabench.baselines.preprocessing.target import transform_preprocess_y
    from multabench.baselines.training.metrics import calculate_metric
    from multabench.benchmark.load import load_multabench_dataset, split_dataset
    from multabench.datasets.all_datasets import MulTaBenchDatasetID
    from multabench.e5.constants import TEXT_ENCODERS
    from multabench.finetune.train_args import E5TrainArgs

    _install_embedding_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = _model_cls(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    for ds_name in args.datasets:
        dataset = load_multabench_dataset(MulTaBenchDatasetID[ds_name])
        for fold in args.folds:
            split = split_dataset(dataset, fold=fold, size=args.size)
            for enc_name in args.encoders:
                out_path = os.path.join(args.out_dir, f"{args.model}_{ds_name}_{enc_name}_{fold}.json")
                if os.path.exists(out_path):
                    continue
                enc = TEXT_ENCODERS[enc_name]
                e5_kwargs = vars(E5TrainArgs()) if enc.tune_encoder else None
                t0 = time.time()
                model = model_cls(problem_type=dataset.task_type, device=device, verbose=False,
                                  dataset=dataset.dataset_id, image_folder=dataset.image_folder,
                                  tune_e5=enc.tune_encoder, e5_train_kwargs=e5_kwargs, e5_model_name=enc.encoder_name)
                model.fit(split.x_train.copy(), split.y_train.copy())
                y_true = np.asarray(transform_preprocess_y(y=split.y_test.copy(), scaler=model.target_transformer))
                y_pred = np.asarray(model.predict(split.x_test.copy()))
                res = {
                    "model": args.model, "dataset": ds_name, "encoder": enc_name, "fold": fold, "size": args.size,
                    **calculate_metric(y_true=y_true, y_pred=y_pred, d_output=model.d_output),
                    **course_metric(y_true, y_pred, model.is_cls),
                    "n_text_cols": len(model.text_transformers),
                    "n_text_features": int(sum(t.n_components for t in model.text_transformers.values())),
                    "n_train": len(split.y_train), "n_test": len(split.y_test),
                    "runtime_s": time.time() - t0,
                }
                with open(out_path, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"{ds_name} fold {fold} {enc_name:18s} {res['course_metric']} {res['course_score']:.4f} "
                      f"({res['n_text_features']} text feats, {res['runtime_s']:.0f}s)")


def summarize(args):
    import pandas as pd
    from scipy.stats import wilcoxon

    rows = [json.load(open(p)) for p in glob.glob(os.path.join(args.out_dir, "*.json"))]
    if not rows:
        print(f"no results in {args.out_dir}")
        return
    df = pd.DataFrame(rows)
    for model, dm in df.groupby("model"):
        print(f"\n=== {model} ===")
        table = dm.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
        print("mean course score (AUROC / R^2) per dataset:")
        print(table.round(4).to_string())
        table.to_csv(os.path.join(args.out_dir, f"scores_{model}.csv"))

        wide = dm.pivot_table(index=["dataset", "fold"], columns="encoder", values="course_score")
        for base in args.baselines:
            if base not in wide:
                continue
            _compare(wide, base, args.out_dir, model)
    print("\nNote: with few datasets the p-values are indicative only; 6 folds per dataset are not independent "
          "(2 repeats of 3-fold CV), so treat dataset-level wins as the primary signal.")


def _compare(wide, base, out_dir, model):
    import pandas as pd
    from scipy.stats import wilcoxon
    print(f"\npaired comparison vs {base} over (dataset, fold) pairs:")
    out = []
    for enc in wide.columns:
        if enc == base:
            continue
        pair = wide[[base, enc]].dropna()
        d = pair[enc] - pair[base]
        p = wilcoxon(d).pvalue if len(d) >= 6 and (d != 0).any() else float("nan")
        per_ds = d.groupby(level="dataset").mean()
        out.append({"encoder": enc, "n_pairs": len(d), "mean_delta": d.mean(), "wins": int((d > 0).sum()),
                    "losses": int((d < 0).sum()), "wilcoxon_p": p,
                    "datasets_improved": f"{int((per_ds > 0).sum())}/{len(per_ds)}"})
    summary = pd.DataFrame(out).set_index("encoder")
    print(summary.round(4).to_string())
    summary.to_csv(os.path.join(out_dir, f"summary_{model}_vs_{base}.csv"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="light")
    r.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    r.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    r.add_argument("--folds", nargs="+", type=int, default=list(range(6)))
    r.add_argument("--size", default="10k")
    r.add_argument("--out_dir", default="runs_sae_pilot")
    s = sub.add_parser("summarize")
    s.add_argument("--out_dir", default="runs_sae_pilot")
    s.add_argument("--baselines", nargs="+", default=[BASELINE, "e5-small-pca50"])
    args = p.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
