"""Target-relevant concept filtering before PCA ("supervised principal components in an SAE basis").

Per text column, per outer fold (training rows only):
  1. frozen E5 embedding x (384)                          - same as Joint Frozen
  2. TopK SAE (256 latents, k=8) trained on this column's training embeddings, no labels
  3. relevance of each latent = max_c |corr(rank(z_i), R_c)|, where R is
       conditional (main):  residual of an out-of-fold, tabular-only LightGBM   (what the table does NOT explain)
       univariate (ablation): y minus its mean                                    (what y is, ignoring the table)
     only latents active on >= MIN_ACTIVE training rows are eligible
  4. keep S = latents whose score beats the 95th percentile of the max score over 50 row-permutations of R
     (family-wise control); if none pass, keep the top FALLBACK_K
  5. subtractive map back: x~ = x - sum_{i not in S} z_i d_i     (removes irrelevant concepts, keeps SAE residual)
  6. PCA(30) on training x~                                      - same column count as Joint Frozen

Controls (same steps, one thing changed): random latents of the same size, selection on raw E5 dimensions instead
of SAE latents, and PLS-30. Everything that uses y makes the method target-aware -> compare against Joint TAR.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import rankdata
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold

from multabench.constants import SEED
from multabench.sae.topk_sae import SAEConfig, TopKSAE, encode_numpy, train_sae

N_LATENTS, K_ACTIVE = 256, 8
MIN_ACTIVE = 20
N_PERM, PERM_Q = 50, 0.95
FALLBACK_K = 10
N_PCA = 30


# ---------------------------------------------------------------- relevance target R
def target_matrix(y: Series, is_cls: bool) -> (np.ndarray, Optional[np.ndarray]):
    """Encoded target as a matrix: one-hot (binary uses the positive class only) or standardized y."""
    if is_cls:
        classes, codes = np.unique(y.astype(str).to_numpy(), return_inverse=True)
        onehot = np.eye(len(classes))[codes]
        return (onehot[:, 1:2] if len(classes) == 2 else onehot), codes
    v = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    return ((v - np.nanmean(v)) / (np.nanstd(v) or 1.0))[:, None], None


def tabular_frame(x: DataFrame, text_cols: List[str]) -> DataFrame:
    """Non-text columns for a tabular-only model: numbers stay numeric, everything else becomes a category."""
    out = {}
    for c in x.columns:
        if c in text_cols:
            continue
        num = pd.to_numeric(x[c], errors="coerce")
        if num.notna().mean() >= 0.9 or x[c].isna().all():
            out[str(c)] = num.astype(float)
        else:
            out[str(c)] = x[c].astype(str).astype("category")
    return DataFrame(out, index=x.index)


def relevance_residual(x: DataFrame, y: Series, text_cols: List[str], is_cls: bool, conditional: bool,
                       n_splits: int = 5) -> np.ndarray:
    """R = target matrix minus a baseline prediction: the OOF tabular-only LightGBM (conditional) or the mean."""
    T, codes = target_matrix(y, is_cls)
    tab = tabular_frame(x, text_cols)
    if not conditional or tab.shape[1] == 0:
        return T - T.mean(axis=0, keepdims=True)
    from lightgbm import LGBMClassifier, LGBMRegressor
    pred = np.zeros_like(T)
    strat = is_cls and np.bincount(codes).min() >= n_splits
    splitter = (StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(tab, codes) if strat
                else KFold(n_splits, shuffle=True, random_state=SEED).split(tab))
    for tr, ho in splitter:
        if is_cls:
            m = LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=SEED)
            m.fit(tab.iloc[tr], codes[tr])
            p = np.zeros((len(ho), T.shape[1] if T.shape[1] > 1 else 2))
            p[:, m.classes_] = m.predict_proba(tab.iloc[ho])
            pred[ho] = p[:, 1:2] if T.shape[1] == 1 else p
        else:
            m = LGBMRegressor(n_estimators=200, learning_rate=0.05, verbose=-1, random_state=SEED)
            m.fit(tab.iloc[tr], T[tr, 0])
            pred[ho, 0] = m.predict(tab.iloc[ho])
    return T - pred


# ---------------------------------------------------------------- scoring + permutation selection
def _std(r: np.ndarray) -> np.ndarray:
    r = r - r.mean(axis=0, keepdims=True)
    sd = r.std(axis=0, keepdims=True)
    return np.divide(r, sd, out=np.zeros_like(r, dtype=float), where=sd > 0)


def relevance_scores(feats: np.ndarray, R: np.ndarray, eligible: np.ndarray, n_perm: int = N_PERM,
                    seed: int = SEED):
    """score_i = max_c |corr(rank(feat_i), R_c)|; null = max over eligible features under row permutations of R.

    Features are rank-transformed (robust to the SAE's heavy-tailed, mostly-zero activations); R is NOT ranked: a
    calibrated residual is uncorrelated with every function of the tabular features, and ranking it would break that,
    letting concepts that merely duplicate the table look relevant."""
    F = _std(rankdata(feats, axis=0))
    Rr = _std(R.astype(float))
    n = len(F)
    score = np.abs(F.T @ Rr / n).max(axis=1)
    score[~eligible] = 0.0
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    per_feat = np.zeros((n_perm, F.shape[1]))
    for b in range(n_perm):
        s = np.abs(F.T @ Rr[rng.permutation(n)] / n).max(axis=1)
        per_feat[b] = s
        null[b] = s[eligible].max() if eligible.any() else 0.0
    relevance_scores.null_median = np.median(per_feat, axis=0)   # per-feature median under "no relation"
    return score, float(np.quantile(null, PERM_Q))


def select(score: np.ndarray, thr: float, eligible: np.ndarray) -> np.ndarray:
    keep = np.where((score > thr) & eligible)[0]
    if len(keep) == 0:
        order = np.argsort(-score)
        keep = order[: min(FALLBACK_K, int(eligible.sum()) or FALLBACK_K)]
    return np.sort(keep)


# ---------------------------------------------------------------- per-column transforms
class SAERelevanceTransform:
    """kind: 'sae' (relevance-selected latents) or 'random' (same number of random latents).
    mode: 'sub' -> x - sum_{i not in S} z_i d_i   (keeps the SAE's reconstruction error)
          'add' -> b_dec + sum_{i in S} z_i d_i   (kept concepts only)"""

    def __init__(self, R: np.ndarray, kind: str = "sae", mode: str = "sub", n_random: Optional[int] = None,
                 seed: int = SEED, rule: str = "keep_proven", n_latents: int = N_LATENTS, k: int = K_ACTIVE):
        """rule: 'keep_proven'      keep only latents that pass the family-wise permutation test (v1, aggressive)
                 'drop_irrelevant'  remove only eligible latents scoring below their own null median (v2, gentle);
                                    everything uncertain, and rare latents, are kept"""
        self.R, self.kind, self.mode, self.n_random, self.seed = R, kind, mode, n_random, seed
        self.rule, self.n_latents, self.k = rule, n_latents, k
        self.diag: Dict = {}

    def fit(self, emb: np.ndarray, texts: Optional[List[str]] = None) -> "SAERelevanceTransform":
        L = self.n_latents
        cfg = SAEConfig(d_in=emb.shape[1], n_latents=L, k=self.k, k_aux=L, dead_after=max(len(emb) // 2, 1000))
        steps_per_epoch = max(1, len(emb) // 256)
        self.sae = train_sae(emb, cfg, device="cpu", epochs=max(5, min(200, 3000 // steps_per_epoch)),
                             batch_size=256, lr=1e-3, seed=self.seed, log_every=10**9)
        z = encode_numpy(self.sae, emb)
        eligible = (z > 0).sum(axis=0) >= MIN_ACTIVE
        score, thr = relevance_scores(z, self.R, eligible)
        null_med = relevance_scores.null_median
        if self.rule == "drop_irrelevant":
            drop = np.where(eligible & (score < null_med))[0]
            if self.kind == "random":
                drop = np.sort(np.random.default_rng(self.seed).choice(np.where(eligible)[0], len(drop), replace=False))
            keep = np.setdiff1d(np.arange(L), drop)
        else:
            keep = select(score, thr, eligible)
            if self.kind == "random":
                pool = np.where(eligible)[0] if eligible.any() else np.arange(L)
                keep = np.sort(np.random.default_rng(self.seed).choice(pool, min(self.n_random or len(keep), len(pool)), replace=False))
        self.keep_idx = keep
        self.drop = np.setdiff1d(np.arange(L), keep)
        self.W_dec = self.sae.W_dec.detach().cpu().numpy()
        self.b_dec = self.sae.b_dec.detach().cpu().numpy()
        x_t = self._filter(emb, z)
        kept_part = z[:, keep] @ self.W_dec[keep]
        resid = emb - (z @ self.W_dec + self.b_dec)
        tot = float(((emb - emb.mean(0)) ** 2).sum())
        var_share = lambda a: round(float(((a - a.mean(0)) ** 2).sum()) / tot, 4)
        self.pca = PCA(n_components=min(N_PCA, *x_t.shape), random_state=SEED).fit(x_t)
        self.n_components = self.pca.n_components_
        removed = np.linalg.norm(emb - x_t, axis=1).mean() / np.linalg.norm(emb - emb.mean(0), axis=1).mean()
        self.diag = {"rule": self.rule, "sae": f"{L}/k{self.k}", "n_eligible": int(eligible.sum()),
                     "n_kept": int(len(keep)), "n_removed": int(len(self.drop)), "threshold": round(thr, 4),
                     "passed_test": int(((score > thr) & eligible).sum()), "removed_norm_ratio": round(float(removed), 4),
                     "var_share_kept": var_share(kept_part), "var_share_sae_error": var_share(resid), "mode": self.mode,
                     "top": self._top(keep, score, z, texts)}
        self.keep = keep
        return self

    def _top(self, keep, score, z, texts, n_lat: int = 5, n_txt: int = 3):
        if texts is None:
            return []
        out = []
        for i in keep[np.argsort(-score[keep])][:n_lat]:
            idx = np.argsort(-z[:, i])[:n_txt]
            out.append({"latent": int(i), "score": round(float(score[i]), 4), "n_active": int((z[:, i] > 0).sum()),
                        "examples": [str(texts[j])[:160] for j in idx]})
        return out

    def _filter(self, emb: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.mode == "add":
            return self.b_dec + z[:, self.keep_idx] @ self.W_dec[self.keep_idx]
        return emb - z[:, self.drop] @ self.W_dec[self.drop]

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.pca.transform(self._filter(emb, encode_numpy(self.sae, emb)))


class DimRelevanceTransform:
    """Control: the same relevance test on the raw E5 dimensions; keep the selected dimensions, then PCA."""

    def __init__(self, R: np.ndarray):
        self.R = R
        self.diag: Dict = {}

    def fit(self, emb: np.ndarray, texts=None) -> "DimRelevanceTransform":
        eligible = np.ones(emb.shape[1], dtype=bool)
        score, thr = relevance_scores(emb, self.R, eligible)
        self.keep = select(score, thr, eligible)
        self.pca = PCA(n_components=min(N_PCA, len(self.keep), len(emb)), random_state=SEED).fit(emb[:, self.keep])
        self.n_components = self.pca.n_components_
        self.diag = {"n_kept": int(len(self.keep)), "threshold": round(thr, 4), "passed_test": int((score > thr).sum())}
        return self

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.pca.transform(emb[:, self.keep])


class PLSTransform:
    """Control: partial least squares to 30 components, using the (univariate) target matrix."""

    def __init__(self, T: np.ndarray):
        self.T = T
        self.diag: Dict = {}

    def fit(self, emb: np.ndarray, texts=None) -> "PLSTransform":
        self.pls = PLSRegression(n_components=min(N_PCA, emb.shape[1], len(emb) - 1), scale=False).fit(emb, self.T)
        self.n_components = self.pls.n_components
        return self

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.pls.transform(emb)
