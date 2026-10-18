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


# ---------------------------------------------------------------- v3: joint (sibling-aware) group-shadow pruning
GSAE_LATENTS, GSAE_K = 2048, 32
GROUP_DIST = 0.5        # siblings: average-linkage cosine distance between decoder directions below this
N_SHADOW_REPEATS = 3


def group_siblings(W: np.ndarray, dist_threshold: float = GROUP_DIST) -> np.ndarray:
    """Cluster latents whose decoder directions point the same way (feature-splitting siblings). Returns labels."""
    if len(W) < 2:
        return np.zeros(len(W), dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=dist_threshold).fit_predict(W)


def group_shadow_importance(tab: DataFrame, z: np.ndarray, groups: np.ndarray, target: np.ndarray, is_cls: bool,
                            repeats: int = N_SHADOW_REPEATS, seed: int = SEED):
    """One LightGBM per repeat on [table | all latents | one shadow copy per group]. A group's shadow is its latents
    with the rows permuted JOINTLY (same permutation for the whole group), so it keeps the group's internal structure
    and marginal distribution but has no relation to the target, the table or other concepts.
    Returns (real, shadow): (repeats, n_groups) arrays of total split gain per group."""
    from lightgbm import LGBMClassifier, LGBMRegressor
    G = int(groups.max()) + 1 if len(groups) else 0
    rng = np.random.default_rng(seed)
    real, shadow = np.zeros((repeats, G)), np.zeros((repeats, G))
    n_tab, L = tab.shape[1], z.shape[1]
    for r in range(repeats):
        sh = np.empty_like(z)
        for g in range(G):
            cols = np.where(groups == g)[0]
            sh[:, cols] = z[np.ix_(rng.permutation(len(z)), cols)]
        X = pd.concat([tab.reset_index(drop=True),
                       DataFrame(np.hstack([z, sh]), columns=[f"z{i}" for i in range(L)] + [f"s{i}" for i in range(L)])],
                      axis=1)
        params = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3, subsample=0.8,
                      subsample_freq=1, importance_type="gain", verbose=-1, random_state=seed + r)
        m = (LGBMClassifier(**params) if is_cls else LGBMRegressor(**params)).fit(X, target)
        imp = m.booster_.feature_importance(importance_type="gain")
        real[r] = np.bincount(groups, weights=imp[n_tab:n_tab + L], minlength=G)
        shadow[r] = np.bincount(groups, weights=imp[n_tab + L:], minlength=G)
    return real, shadow


class GroupShadowSAETransform:
    """Concept pruning judged jointly with sibling concepts and the table (group-wise Boruta).

    Per text column, on the fold's training rows only:
      1. TopK SAE (2048 latents, k=32) on the column's embeddings (frozen or TAR-tuned), no labels
      2. eligible latents (active on >= MIN_ACTIVE rows) are clustered into sibling groups by decoder direction
      3. LightGBM on [table | all eligible latents | a jointly-permuted shadow copy of each group], repeated 3x
      4. drop a group if its real gain does not beat its own shadow's gain in a majority of repeats
         (rare latents are never tested and are kept)
      5. subtractive map-back x~ = x - sum_{dropped} z_i d_i, then PCA-30 (same column count as Frozen/TAR)
    kind='random' drops the same number of eligible latents chosen at random (same SAE)."""

    def __init__(self, tab: DataFrame, target: np.ndarray, is_cls: bool, kind: str = "sae",
                 n_latents: int = GSAE_LATENTS, k: int = GSAE_K, device: str = "cpu", seed: int = SEED,
                 repeats: int = N_SHADOW_REPEATS, dist_threshold: float = GROUP_DIST):
        self.tab, self.target, self.is_cls, self.kind = tab, target, is_cls, kind
        self.n_latents, self.k, self.device, self.seed = n_latents, k, str(device), seed
        self.repeats, self.dist_threshold = repeats, dist_threshold
        self.diag: Dict = {}

    def fit(self, emb: np.ndarray, texts: Optional[List[str]] = None) -> "GroupShadowSAETransform":
        assert len(emb) == len(self.tab) == len(self.target), "transform must be fit on the training rows only"
        L = self.n_latents
        cfg = SAEConfig(d_in=emb.shape[1], n_latents=L, k=self.k, k_aux=min(L, 512), dead_after=max(len(emb) // 2, 1000))
        steps_per_epoch = max(1, len(emb) // 256)
        self.sae = train_sae(emb, cfg, device=self.device, epochs=max(5, min(200, 4000 // steps_per_epoch)),
                             batch_size=256, lr=1e-3, seed=self.seed, log_every=10**9).cpu()
        z = encode_numpy(self.sae, emb)
        self.W_dec = self.sae.W_dec.detach().cpu().numpy()
        self.b_dec = self.sae.b_dec.detach().cpu().numpy()
        elig = np.where((z > 0).sum(axis=0) >= MIN_ACTIVE)[0]
        groups = group_siblings(self.W_dec[elig], self.dist_threshold)
        real, shadow = group_shadow_importance(self.tab, z[:, elig], groups, self.target, self.is_cls,
                                               self.repeats, self.seed)
        wins = (real > shadow).sum(axis=0)
        drop_groups = np.where(wins <= self.repeats // 2)[0]
        drop = elig[np.isin(groups, drop_groups)]
        if self.kind == "random":
            drop = np.random.default_rng(self.seed).choice(elig, len(drop), replace=False)
        self.drop = np.sort(drop)
        x_t = self._filter(emb, z)
        self.pca = PCA(n_components=min(N_PCA, *x_t.shape), random_state=SEED).fit(x_t)
        self.n_components = self.pca.n_components_
        tot = float(((emb - emb.mean(0)) ** 2).sum())
        removed_part = z[:, self.drop] @ self.W_dec[self.drop]
        resid = emb - (z @ self.W_dec + self.b_dec)
        G = int(groups.max()) + 1 if len(groups) else 0
        sizes = np.bincount(groups, minlength=G)
        self.diag = {"rule": "group_shadow", "kind": self.kind, "sae": f"{L}/k{self.k}", "n_eligible": int(len(elig)),
                     "n_groups": G, "n_multi_groups": int((sizes > 1).sum()), "n_groups_dropped": int(len(drop_groups)),
                     "n_removed": int(len(self.drop)),
                     "var_share_removed": round(float(((removed_part - removed_part.mean(0)) ** 2).sum()) / tot, 4),
                     "var_share_sae_error": round(float(((resid - resid.mean(0)) ** 2).sum()) / tot, 4),
                     "top": self._top(elig, groups, real.mean(0) - shadow.mean(0), z, texts)}
        return self

    def _top(self, elig, groups, margin, z, texts, n_grp: int = 5, n_txt: int = 3):
        if texts is None or len(margin) == 0:
            return []
        out = []
        for g in np.argsort(-margin)[:n_grp]:
            members = elig[groups == g]
            act = z[:, members].sum(axis=1)
            out.append({"group": int(g), "latents": [int(i) for i in members[:10]], "gain_margin": round(float(margin[g]), 2),
                        "examples": [str(texts[j])[:160] for j in np.argsort(-act)[:n_txt]]})
        return out

    def _filter(self, emb: np.ndarray, z: np.ndarray) -> np.ndarray:
        return emb - z[:, self.drop] @ self.W_dec[self.drop]

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.pca.transform(self._filter(emb, encode_numpy(self.sae, emb)))


# ---------------------------------------------------------------- v4: held-out, harm-ranked, validated amount
CV_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75)
CV_TOL = 0.001          # a larger removal must lower the held-out loss by more than 0.1% to be preferred


def _lgbm(is_cls: bool, seed: int, **kw):
    from lightgbm import LGBMClassifier, LGBMRegressor
    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                  verbose=-1, random_state=seed, **kw)
    return LGBMClassifier(**params) if is_cls else LGBMRegressor(**params)


def _loss(model, X, target, is_cls: bool) -> float:
    """Held-out log loss (classes unseen in training get probability 1e-6) or MSE."""
    if is_cls:
        p = np.clip(model.predict_proba(X), 1e-6, 1.0)
        col = {c: j for j, c in enumerate(model.classes_)}
        pt = np.array([p[i, col[t]] if t in col else 1e-6 for i, t in enumerate(target)])
        return float(-np.mean(np.log(pt)))
    return float(np.mean((model.predict(X) - target) ** 2))


def _tab_numeric(tab: DataFrame):
    """Table as a float matrix; category columns become integer codes (NaN for missing), returned as indices."""
    cols, cat_idx = [], []
    for j, c in enumerate(tab.columns):
        if isinstance(tab[c].dtype, pd.CategoricalDtype):
            codes = tab[c].cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            cols.append(codes)
            cat_idx.append(j)
        else:
            cols.append(tab[c].to_numpy(dtype=float))
    return (np.column_stack(cols) if cols else np.zeros((len(tab), 0))), cat_idx


class HeldOutPruneSAETransform:
    """Concept pruning decided on rows the fine-tuned encoder never trained on.

    `val_mask` marks TAR's own early-stopping rows: TAR's LoRA never took a gradient step on them, so concept
    activations there are not memorized. Per text column:
      1. TopK SAE (2048, k=32) on all training-fold embeddings (unsupervised)
      2. sibling groups of eligible latents (as in v3)
      3. LightGBM on fit rows over [table | all eligible latents]; group importance = rise in held-out loss when
         the group's columns are permuted jointly on the held-out rows (joint: all other concepts stay in place)
      4. harm ranking: a group is harmful when it takes up variance (which PCA-30 spends components on) but
         carries no held-out signal; score = variance share / max(importance, eps), largest first
      5. amount: remove the top {0, 10, 25, 50, 75}% of eligible concepts by that ranking; for each, PCA-30 on fit
         rows -> LightGBM [table | PCA] -> held-out loss; pick the lowest (0% = plain TAR, so it can opt out)
      6. subtractive map-back of the chosen set, PCA-30 on all training rows
    Test rows are never used."""

    def __init__(self, tab: DataFrame, target: np.ndarray, is_cls: bool, val_mask: np.ndarray,
                 n_latents: int = GSAE_LATENTS, k: int = GSAE_K, device: str = "cpu", seed: int = SEED,
                 fractions=CV_FRACTIONS, dist_threshold: float = GROUP_DIST, n_perm: int = 2):
        self.tab, self.target, self.is_cls = tab.reset_index(drop=True), np.asarray(target), is_cls
        self.val = np.asarray(val_mask, dtype=bool)
        self.n_latents, self.k, self.device, self.seed = n_latents, k, str(device), seed
        self.fractions, self.dist_threshold, self.n_perm = fractions, dist_threshold, n_perm
        self.diag: Dict = {}

    def fit(self, emb: np.ndarray, texts: Optional[List[str]] = None) -> "HeldOutPruneSAETransform":
        assert len(emb) == len(self.tab) == len(self.target) == len(self.val)
        fit_r, val_r = ~self.val, self.val
        L = self.n_latents
        cfg = SAEConfig(d_in=emb.shape[1], n_latents=L, k=self.k, k_aux=min(L, 512), dead_after=max(len(emb) // 2, 1000))
        steps_per_epoch = max(1, len(emb) // 256)
        self.sae = train_sae(emb, cfg, device=self.device, epochs=max(5, min(200, 4000 // steps_per_epoch)),
                             batch_size=256, lr=1e-3, seed=self.seed, log_every=10**9).cpu()
        z = encode_numpy(self.sae, emb)
        self.W_dec = self.sae.W_dec.detach().cpu().numpy()
        self.b_dec = self.sae.b_dec.detach().cpu().numpy()
        elig = np.where((z[fit_r] > 0).sum(axis=0) >= MIN_ACTIVE)[0]
        groups = group_siblings(self.W_dec[elig], self.dist_threshold)
        G = int(groups.max()) + 1 if len(groups) else 0

        # 3. joint held-out permutation importance per group (numpy for speed; categorical table columns as codes)
        tab_np, cat_idx = _tab_numeric(self.tab)
        X = np.hstack([tab_np, z[:, elig]])
        m = _lgbm(self.is_cls, self.seed, colsample_bytree=0.3).fit(X[fit_r], self.target[fit_r],
                                                                     categorical_feature=cat_idx or "auto")
        Xv = X[val_r]
        base = _loss(m, Xv, self.target[val_r], self.is_cls)
        rng = np.random.default_rng(self.seed)
        imp = np.zeros(G)
        n_tab = tab_np.shape[1]
        for g in range(G):
            cols_g = n_tab + np.where(groups == g)[0]
            if not Xv[:, cols_g].any():
                continue                                  # never active on held-out rows: no evidence either way
            for _ in range(self.n_perm):
                Xp = Xv.copy()
                Xp[:, cols_g] = Xv[rng.permutation(len(Xv))][:, cols_g]
                imp[g] += (_loss(m, Xp, self.target[val_r], self.is_cls) - base) / self.n_perm

        # 4. harm ranking: variance taken per unit of held-out signal
        # decoder rows are unit-norm, so the variance contributed by latent i is just var(z_i)
        tot = float(((emb[fit_r] - emb[fit_r].mean(0)) ** 2).sum())
        contrib_var = z[fit_r][:, elig].var(axis=0) * fit_r.sum() / tot
        gvar = np.bincount(groups, weights=contrib_var, minlength=G)
        pos = imp[imp > 0]
        eps = 1e-3 * (np.median(pos) if len(pos) else 1e-6)
        harm = gvar / (np.maximum(imp, 0) + eps)
        order = np.argsort(-harm)                          # most harmful group first
        cum = np.cumsum(np.bincount(groups, minlength=G)[order])

        # 5. choose the amount on held-out rows
        results = []
        for f in self.fractions:
            n_groups = int(np.searchsorted(cum, f * len(elig), side="right")) if f > 0 else 0
            drop = np.sort(elig[np.isin(groups, order[:n_groups])])
            x_t = emb - z[:, drop] @ self.W_dec[drop]
            pca = PCA(n_components=min(N_PCA, x_t.shape[1]), random_state=SEED).fit(x_t[fit_r])
            P = np.hstack([tab_np, pca.transform(x_t)])
            mm = _lgbm(self.is_cls, self.seed).fit(P[fit_r], self.target[fit_r], categorical_feature=cat_idx or "auto")
            results.append((f, _loss(mm, P[val_r], self.target[val_r], self.is_cls), drop))
        loss0 = results[0][1]
        best = results[0]
        for f, l, d in results[1:]:
            if l < min(best[1], loss0 * (1 - CV_TOL)):
                best = (f, l, d)
        self.drop = best[2]
        x_t = self._filter(emb, z)
        self.pca = PCA(n_components=min(N_PCA, *x_t.shape), random_state=SEED).fit(x_t)
        self.n_components = self.pca.n_components_
        removed = z[:, self.drop] @ self.W_dec[self.drop]
        self.diag = {"rule": "heldout_cv", "sae": f"{L}/k{self.k}", "n_eligible": int(len(elig)), "n_groups": G,
                     "n_val_rows": int(val_r.sum()), "chosen_fraction": best[0], "n_removed": int(len(self.drop)),
                     "val_loss": {str(f): round(l, 5) for f, l, _ in results},
                     "frac_groups_no_signal": round(float((imp <= 0).mean()), 3) if G else None,
                     "var_share_removed": round(float(((removed - removed.mean(0)) ** 2).sum())
                                                / float(((emb - emb.mean(0)) ** 2).sum()), 4)}
        return self

    def _filter(self, emb: np.ndarray, z: np.ndarray) -> np.ndarray:
        return emb - z[:, self.drop] @ self.W_dec[self.drop]

    def transform(self, emb: np.ndarray) -> np.ndarray:
        return self.pca.transform(self._filter(emb, encode_numpy(self.sae, emb)))
