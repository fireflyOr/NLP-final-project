"""Out-of-fold text-model predictions as extra columns for the tabular learner ("stacking").

The tabular learner keeps the usual frozen E5 -> PCA-30 columns per text column, and additionally gets the
prediction of a text-only model trained on y:

    kind="frozen":    MLP on the concatenated frozen E5 embeddings of all text columns (one example per row)
    kind="finetuned": E5 fine-tuned with LoRA exactly as in Joint TAR (one example per row x text column,
                      "col: value"), keeping its prediction head; per-row prediction = mean over text columns

Both use y, so they are judged against Joint TAR. Leakage control: the text model is cross-fitted inside the
training data. Every training row's feature comes from a model that never saw that row; any other row
(LightGBM's validation split, test rows) gets the average of the n_splits models.

Output columns: binary -> 1 logit; multiclass -> one log-probability per class; regression -> 1 prediction.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from multabench.constants import SEED

STACK_KEY = "__text_stack__"   # key of the StackEncoder inside the text_encoders dict
N_REG_BINS = 20                # regression targets are binned for the fine-tuned head, as in Joint TAR


def _row_keys(x: DataFrame, text_cols: List[str]) -> List[str]:
    """Identify rows by index plus text content, so OOF features are only reused for the exact training rows."""
    keys = []
    for idx, row in zip(x.index, x[text_cols].astype(str).itertuples(index=False)):
        keys.append(f"{idx}|" + hashlib.sha1("\x1f".join(row).encode("utf-8", "replace")).hexdigest())
    return keys


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def _train_mlp(X: np.ndarray, y: np.ndarray, task: str, d_out: int, device: torch.device, seed: int,
               max_epochs: int = 200, patience: int = 10, batch_size: int = 256) -> _MLP:
    """task: 'bin' (BCE on 1 logit), 'mul' (CE), 'reg' (MSE on standardized y). Early stopping on 10% of rows."""
    torch.manual_seed(seed)
    idx_tr, idx_va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long if task == "mul" else torch.float32, device=device)
    model = _MLP(X.shape[1], d_out).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def loss_fn(out, target):
        if task == "bin":
            return nn.functional.binary_cross_entropy_with_logits(out.squeeze(-1), target)
        if task == "mul":
            return nn.functional.cross_entropy(out, target)
        return nn.functional.mse_loss(out.squeeze(-1), target)

    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(max_epochs):
        model.train()
        perm = torch.as_tensor(idx_tr)[torch.randperm(len(idx_tr), generator=g)]
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size].to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = torch.as_tensor(idx_va, device=device)
            val = loss_fn(model(Xt[v]), yt[v]).item()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def finetuned_logits(model: Any, tokenizer: Any, texts: List[str], device: torch.device,
                     batch_size: int = 64, max_length: int = 512) -> np.ndarray:
    """Logits of a fine-tuned E5ForTuning on 'col: value' texts (prefix added as in Joint TAR's training data)."""
    from multabench.e5.constants import E5_PASSAGE_PREFIX
    model.eval()
    prefixed = [E5_PASSAGE_PREFIX + (str(t).strip() or " ") for t in texts]
    order = np.argsort([len(t) for t in prefixed])
    out: Optional[np.ndarray] = None
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer([prefixed[j] for j in idx], padding=True, truncation=True, max_length=max_length,
                        return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])["logits"]
        logits = logits.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), logits.shape[1]), dtype=np.float32)
        out[idx] = logits
    return out


class StackEncoder:
    def __init__(self, kind: str, text_cols: List[str], is_cls: bool, device: torch.device,
                 n_splits: int, e5_train_kwargs: Optional[Dict[str, Any]] = None):
        assert kind in ("frozen", "finetuned"), kind
        self.kind = kind
        self.text_cols = list(text_cols)
        self.is_cls = is_cls
        self.device = device
        self.n_splits = n_splits
        self.e5_train_kwargs = e5_train_kwargs or {}
        self.models: list = []
        self.oof: Dict[str, np.ndarray] = {}
        self.n_components: Optional[int] = None
        self.n_ensemble_rows = 0  # for sanity checks: rows that got ensemble (not OOF) predictions

    # ---------------- target handling ----------------
    def _prepare_target(self, y: Series) -> np.ndarray:
        if self.is_cls:
            self.label_encoder = LabelEncoder().fit(y.astype(str))
            self.n_classes = len(self.label_encoder.classes_)
            self.task = "bin" if self.n_classes == 2 else "mul"
            self.n_components = 1 if self.task == "bin" else self.n_classes
            return self.label_encoder.transform(y.astype(str))
        self.task = "reg"
        self.n_components = 1
        y = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
        self.y_mean, self.y_std = float(np.nanmean(y)), float(np.nanstd(y) or 1.0)
        return (y - self.y_mean) / self.y_std

    def _splitter(self, yy: np.ndarray, groups: np.ndarray):
        """Cross-fitting folds that keep rows with identical text together. Without this, a duplicated text
        (e.g. the same track listed under several genres) lets the 'out-of-fold' model see the row's twin, so the
        OOF feature is over-confident on training rows and LightGBM over-trusts it."""
        from sklearn.model_selection import StratifiedGroupKFold
        n_groups = len(np.unique(groups))
        if self.task != "reg" and np.bincount(yy.astype(int)).min() >= self.n_splits and n_groups >= self.n_splits:
            return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED).split(yy, yy, groups)
        uniq, inv = np.unique(groups, return_inverse=True)
        fold_of_group = np.random.default_rng(SEED).permutation(len(uniq)) % self.n_splits
        fold = fold_of_group[inv]
        return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(self.n_splits)]

    # ---------------- frozen: MLP on concatenated E5 embeddings ----------------
    def _frozen_features(self, x: DataFrame) -> np.ndarray:
        import multabench.baselines.preprocessing.text_embeddings as te
        from multabench.e5.constants import E5_SMALL_V2
        model, tokenizer = te.get_vanilla_e5(self.device, model_name=E5_SMALL_V2)
        embs = [te.encode_texts_with_e5(texts=x[c].astype(str).fillna("").tolist(), model=model,
                                        tokenizer=tokenizer, device=self.device, col_name=str(c))
                for c in self.text_cols]
        return np.concatenate(embs, axis=1)

    def _fit_frozen(self, feats: np.ndarray, yy: np.ndarray, seed: int):
        d_out = self.n_classes if self.task == "mul" else 1
        return _train_mlp(feats, yy, self.task, d_out, self.device, seed)

    def _predict_frozen(self, model, feats: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(torch.as_tensor(feats, dtype=torch.float32, device=self.device))
            if self.task == "mul":
                out = torch.log_softmax(out, dim=-1)
        return out.cpu().numpy()

    # ---------------- finetuned: LoRA E5 + head, as in Joint TAR ----------------
    def _expand(self, x: DataFrame) -> List[str]:
        """One 'col: value' text per (row, text column), row-major, matching Joint TAR's training format."""
        texts = []
        for _, row in x[self.text_cols].iterrows():
            for c in self.text_cols:
                v = row[c]
                texts.append(f"{c}: {str(v).strip() if pd.notna(v) else ''}")
        return texts

    def _fit_finetuned(self, x: DataFrame, yy: np.ndarray, seed: int):
        from multabench.e5.constants import E5_SMALL_V2
        from multabench.e5.e5_finetune import finetune_e5_with_lora
        from multabench.preprocessing.splits import split_to_val
        from transformers import AutoTokenizer
        if self.task == "reg":
            bins = pd.qcut(pd.Series(yy), q=N_REG_BINS, labels=False, duplicates="drop").to_numpy()
            labels = np.array([f"b{int(b):02d}" for b in bins])  # zero-padded so string order == bin order
            bin_means = pd.Series(yy).groupby(bins).mean()
        else:
            labels = np.array([f"c{int(c):04d}" for c in yy])
            bin_means = None
        x = x.copy()
        lab = Series(labels, index=x.index)
        x_tr, x_va, y_tr, y_va = split_to_val(x=x, y=lab, is_cls=True)
        n_cols = len(self.text_cols)
        tok = AutoTokenizer.from_pretrained(E5_SMALL_V2)
        model, tok, head_le = finetune_e5_with_lora(
            train_texts=self._expand(x_tr), train_y=np.repeat(y_tr.to_numpy(), n_cols),
            val_texts=self._expand(x_va), val_y=np.repeat(y_va.to_numpy(), n_cols),
            device=self.device, tokenizer=tok, model_name=E5_SMALL_V2, seed=SEED + seed,
            return_head=True, **self.e5_train_kwargs)
        return {"model": model, "tokenizer": tok, "head_classes": list(head_le.classes_), "bin_means": bin_means}

    def _predict_finetuned(self, m: dict, x: DataFrame) -> np.ndarray:
        n_cols = len(self.text_cols)
        logits = finetuned_logits(m["model"], m["tokenizer"], self._expand(x), self.device)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p = (p / p.sum(1, keepdims=True)).reshape(len(x), n_cols, -1).mean(axis=1)  # average over text columns
        if self.task == "reg":
            means = np.array([m["bin_means"].get(int(c[1:]), 0.0) for c in m["head_classes"]])
            return (p @ means)[:, None]
        # map the head's classes (only those seen in this split) onto the global class order
        full = np.full((len(x), self.n_classes), 1e-6)
        for j, c in enumerate(m["head_classes"]):
            full[:, int(c[1:])] = p[:, j]
        full = full / full.sum(1, keepdims=True)
        if self.task == "bin":
            return np.log(full[:, 1:2] / full[:, 0:1])
        return np.log(full)

    # ---------------- public API ----------------
    def fit(self, x: DataFrame, y: Series) -> "StackEncoder":
        yy = self._prepare_target(y)
        keep = ~np.isnan(yy) if self.task == "reg" else np.ones(len(yy), dtype=bool)
        x, yy = x.loc[keep], yy[keep]
        feats = self._frozen_features(x) if self.kind == "frozen" else None
        oof = np.zeros((len(x), self.n_components), dtype=np.float32)
        groups = np.array([hashlib.sha1("\x1f".join(r).encode("utf-8", "replace")).hexdigest()
                           for r in x[self.text_cols].astype(str).itertuples(index=False)])
        self.n_duplicate_rows = int(len(groups) - len(np.unique(groups)))
        if self.n_duplicate_rows:
            print(f"🧱 text stack: {self.n_duplicate_rows} training rows duplicate another row's text; kept in one fold")
        self.folds_ = [(np.asarray(tr), np.asarray(ho)) for tr, ho in self._splitter(yy, groups)]
        for k, (tr, ho) in enumerate(self.folds_):
            print(f"🧱 text stack ({self.kind}): fitting split {k + 1}/{self.n_splits} on {len(tr)} rows")
            if self.kind == "frozen":
                m = self._fit_frozen(feats[tr], yy[tr], seed=k)
                oof[ho] = self._predict_frozen(m, feats[ho])
            else:
                m = self._fit_finetuned(x.iloc[tr], yy[tr], seed=k)
                oof[ho] = self._predict_finetuned(m, x.iloc[ho])
            self.models.append(m)
        self.oof = dict(zip(_row_keys(x, self.text_cols), oof))
        return self

    def transform(self, x: DataFrame) -> DataFrame:
        keys = _row_keys(x, self.text_cols)
        out = np.zeros((len(x), self.n_components), dtype=np.float32)
        seen = np.array([k in self.oof for k in keys])
        if seen.any():
            out[seen] = np.stack([self.oof[k] for k, s in zip(keys, seen) if s])
        if (~seen).any():
            x_new = x.loc[~seen] if x.index.is_unique else x.iloc[np.where(~seen)[0]]
            if self.kind == "frozen":
                feats = self._frozen_features(x_new)
                preds = [self._predict_frozen(m, feats) for m in self.models]
            else:
                preds = [self._predict_finetuned(m, x_new) for m in self.models]
            out[~seen] = np.mean(preds, axis=0)
            self.n_ensemble_rows += int((~seen).sum())
        cols = [f"text_stack_{self.kind}_{i}" for i in range(self.n_components)]
        return DataFrame(out, index=x.index, columns=cols)
