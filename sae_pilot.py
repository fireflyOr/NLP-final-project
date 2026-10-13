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
    e5-small-tar       Joint TAR baseline (LoRA fine-tuned, slow)
    e5-small-stack     PCA-30 + out-of-fold prediction of an MLP on frozen E5 (uses y -> compare to e5-small-tar)
    e5-small-tar-stack PCA-30 + out-of-fold prediction of a LoRA-fine-tuned E5 + head (3 fine-tunes per fold, slow)

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


class _DiskEmbeddingCache:
    """Per-text cache of FROZEN E5 embeddings on disk, keyed by (column name, text).

    The frozen encoder's output depends only on the formatted input "passage: <col>: <text>", so entries are valid
    across folds, datasets and sessions. One file per column name; fine-tuned (TAR) encoders are never cached here.
    """

    def __init__(self, cache_dir: str):
        self.dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.mem: dict = {}

    def _path(self, col: str) -> str:
        return os.path.join(self.dir, hashlib.sha1(col.encode()).hexdigest()[:16] + ".pkl")

    def _load(self, col: str) -> dict:
        if col not in self.mem:
            import pickle
            p = self._path(col)
            self.mem[col] = pickle.load(open(p, "rb")) if os.path.exists(p) else {}
        return self.mem[col]

    def get(self, texts, col, compute):
        import pickle
        store = self._load(col)
        missing = list(dict.fromkeys(t for t in texts if t not in store))
        if missing:
            vecs = compute(missing)
            store.update(zip(missing, vecs))
            tmp = self._path(col) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._path(col))
        return np.stack([store[t] for t in texts]).astype(np.float32)


def _install_embedding_cache(cache_dir: str = None):
    """Memoize E5 embeddings inside the text pipeline: in memory for every model (keyed by model object, column and
    texts), and on disk for the frozen model when cache_dir is given."""
    import multabench.baselines.preprocessing.text_embeddings as te
    if getattr(te, "_pilot_cache_installed", False):
        return
    original = te.encode_texts_with_e5
    cache: dict = {}
    disk = _DiskEmbeddingCache(cache_dir) if cache_dir else None

    def is_frozen(model, device):
        from multabench.e5.constants import E5_SMALL_V2
        return model is te.get_vanilla_e5(device, model_name=E5_SMALL_V2)[0]

    def cached(texts, col_name, model, tokenizer, device, **kw):
        h = hashlib.sha1("\x1f".join(texts).encode("utf-8", "replace")).hexdigest()
        key = (id(model), col_name, len(texts), h)
        if key not in cache:
            compute = lambda ts: original(texts=ts, col_name=col_name, model=model, tokenizer=tokenizer, device=device, **kw)
            if disk is not None and is_frozen(model, device):
                cache[key] = disk.get(list(texts), str(col_name), compute)
            else:
                cache[key] = compute(texts)
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

    os.environ.setdefault("MULTABENCH_DYNAMIC_PADDING", "1")  # same results as max_length padding, much faster
    _install_embedding_cache(args.emb_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # required by the LoRA fine-tuning code
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
                e5_kwargs = dict(vars(E5TrainArgs()), batch_size=args.e5_batch_size) if enc.tune_encoder else None
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
                    "n_text_cols": len([k for k in model.text_transformers if not k.startswith("__")]),
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
    r.add_argument("--e5_batch_size", type=int, default=256, help="LoRA fine-tuning batch size (lower it on OOM)")
    r.add_argument("--emb_cache", default=None, help="directory for cached frozen E5 embeddings (e.g. on Drive)")
    r.add_argument("--out_dir", default="runs_sae_pilot")
    s = sub.add_parser("summarize")
    s.add_argument("--out_dir", default="runs_sae_pilot")
    s.add_argument("--baselines", nargs="+", default=[BASELINE, "e5-small-pca50", "e5-small-tar"])
    args = p.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
