"""TabSTAR + SAE concept pruning on the fine-tuned text-cell embeddings.

Per outer fold:
  split   outer-train -> fit rows (90%) + val rows (10%, max 1000), with TabSTAR's own splitter. Test rows are only
          used for final predictions.
  phase 1 standard TabSTAR LoRA fine-tuning on the fit rows (early stopping / checkpoint averaging on val rows).
          -> variant "tabstar": plain TabSTAR predictions.
  embed   the fine-tuned E5 embeds every cell of fit/val/test rows once (CLS vectors, as TabSTAR does); cached.
  prune   per free-text column: TopK SAE (512 latents, k=32) trained on the FIT rows' cell embeddings (no labels);
          relevance of each latent to y measured on the FIT rows only (conditional on an OOF tabular-only model);
          remove latents scoring below their own permutation median ("drop_irrelevant"); subtractive map-back
          e~ = e - sum_{removed} z_i d_i, applied identically to fit/val/test embeddings.
  phase 2 E5 frozen; numerical-fusion + interaction encoder + heads re-trained from the SAME phase-1 weights on the
          cached (pruned or unpruned) fit-row embeddings; early stopping on val rows. Three arms:
            "tabstar-p2"     no pruning   (isolates the effect of the extra training)
            "tabstar-prune"  relevance pruning
            "tabstar-rand"   the same number of latents removed at random

Leakage: nothing is fitted on test rows; val rows are used only for early stopping (as in TabSTAR itself), never
for SAE training or latent selection. Relevance is scored on embeddings of rows the encoder was fine-tuned on,
which inflates scores; under the drop-below-noise rule that biases toward KEEPING latents (the safe direction).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

from multabench.constants import SEED
from multabench.sae import relevance as rel
from multabench.sae.topk_sae import encode_numpy

P2_LR, P2_WD, P2_EPOCHS, P2_PATIENCE, P2_BATCH = 5e-5, 0.01, 20, 3, 128
N_LATENTS, K_ACTIVE = 512, 32


# ------------------------------------------------------------------ column bookkeeping
def free_text_columns(x: DataFrame) -> List[str]:
    """MulTaBench's own rule for which columns are free text (vs categorical / numerical)."""
    from multabench.baselines.preprocessing.dates import fit_date_encoders, transform_date_features
    from multabench.baselines.preprocessing.feature_types import classify_semantic_features, transform_feature_types
    from multabench.baselines.preprocessing.numerical_detection import detect_numerical_features
    from multabench.baselines.preprocessing.sparse import densify_objects
    x = x.copy()
    x, _ = densify_objects(x=x, y=None)
    x = transform_date_features(x=x, date_transformers=fit_date_encoders(x=x))
    num = detect_numerical_features(x)
    x = transform_feature_types(x=x, numerical_features=num, image_features=set())
    return sorted(classify_semantic_features(x=x, numerical_features=num).text_features)


def x_txt_columns(verbalizer, X: DataFrame) -> List[str]:
    """Column order of TabSTARVerbalizer.transform(X).x_txt (replicates its steps without the values)."""
    from tabstar.preprocessing.dates import transform_date_features
    from tabstar.preprocessing.feat_types import transform_feature_types
    from tabstar.preprocessing.sparse import densify_objects
    from tabstar.preprocessing.texts import replace_column_names
    from tabstar.preprocessing.verbalize import prepend_target_tokens
    x = X.copy()
    x, _ = densify_objects(x=x, y=None)
    x = transform_date_features(x=x, date_transformers=verbalizer.date_transformers)
    x, _ = replace_column_names(x=x, y=None)
    num_cols = sorted(verbalizer.numerical_transformers)
    x = transform_feature_types(x=x, numerical_features=set(num_cols))
    x = x.drop(columns=verbalizer.constant_columns, errors="ignore")
    x = prepend_target_tokens(x=x, y_name=verbalizer.y_name, y_values=verbalizer.y_values)
    text_cols = [c for c in x.columns if c not in num_cols]
    return text_cols + num_cols


# ------------------------------------------------------------------ embeddings and the upper network
@torch.no_grad()
def cell_embeddings(model, x_txt: np.ndarray, device, rows_per_batch: int = 256) -> torch.Tensor:
    """[n, seq, 384] CLS embeddings of every cell from the (fine-tuned) TabSTAR text encoder."""
    model.eval()
    out = []
    for i in range(0, len(x_txt), rows_per_batch):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            e = model.get_textual_embedding(x_txt[i:i + rows_per_batch])
        out.append(e.float().cpu())
    return torch.cat(out, dim=0)


def upper_forward(model, emb: torch.Tensor, x_num: torch.Tensor, d_output: int) -> torch.Tensor:
    """TabStarModel.forward without the text encoder (same modules, same order)."""
    h = model.numerical_fusion(textual_embeddings=emb, x_num=x_num)
    h = model.tabular_encoder(h)
    tokens = h[:, :d_output]
    scores = model.reg_head(tokens) if d_output == 1 else model.cls_head(tokens)
    return scores.squeeze(-1)


def _loss(scores: torch.Tensor, y: torch.Tensor, d_output: int) -> torch.Tensor:
    if d_output == 1:
        return torch.nn.functional.mse_loss(scores.view(-1), y.float())
    return torch.nn.functional.cross_entropy(scores, y.long())


def train_upper(model, E_fit, N_fit, y_fit, E_val, N_val, y_val, d_output, device, seed: int = SEED) -> float:
    """Fine-tune fusion + interaction + heads on cached embeddings; early stopping on val loss; restores best."""
    torch.manual_seed(seed)
    upper = [model.numerical_fusion, model.tabular_encoder, model.cls_head, model.reg_head]
    params = [p for m in upper for p in m.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=P2_LR, weight_decay=P2_WD)
    yv = torch.as_tensor(y_val)

    def val_loss():
        for m in upper:
            m.eval()
        with torch.no_grad():
            tot = 0.0
            for i in range(0, len(E_val), 512):
                s = upper_forward(model, E_val[i:i+512].to(device), N_val[i:i+512].to(device), d_output)
                tot += _loss(s, yv[i:i+512].to(device), d_output).item() * len(s)
        return tot / len(E_val)

    best, best_state, bad = val_loss(), copy.deepcopy([m.state_dict() for m in upper]), 0
    g = torch.Generator().manual_seed(seed)
    yf = torch.as_tensor(y_fit)
    for _ in range(P2_EPOCHS):
        for m in upper:
            m.train()
        perm = torch.randperm(len(E_fit), generator=g)
        for i in range(0, len(perm), P2_BATCH):
            b = perm[i:i + P2_BATCH]
            s = upper_forward(model, E_fit[b].to(device), N_fit[b].to(device), d_output)
            loss = _loss(s, yf[b].to(device), d_output)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        v = val_loss()
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy([m.state_dict() for m in upper]), 0
        else:
            bad += 1
            if bad >= P2_PATIENCE:
                break
    for m, sd in zip(upper, best_state):
        m.load_state_dict(sd)
    for m in upper:
        m.eval()
    return best


@torch.no_grad()
def predict_upper(model, E, N, d_output, device) -> np.ndarray:
    out = []
    for i in range(0, len(E), 512):
        s = upper_forward(model, E[i:i+512].to(device), N[i:i+512].to(device), d_output).float()
        out.append(torch.softmax(s, dim=1) if d_output > 1 else s)
    return torch.cat(out).cpu().numpy()


# ------------------------------------------------------------------ pruning
def fit_pruners(E_fit: torch.Tensor, positions: Dict[str, int], R: np.ndarray, device, seed: int = SEED):
    """One SAE per free-text column, trained and scored on FIT rows only. Returns {col: (sae, drop_rel, drop_rand, diag)}."""
    from multabench.sae.topk_sae import SAEConfig, train_sae
    pruners = {}
    for col, pos in positions.items():
        X = E_fit[:, pos, :].numpy()
        cfg = SAEConfig(d_in=X.shape[1], n_latents=N_LATENTS, k=K_ACTIVE, k_aux=N_LATENTS,
                        dead_after=max(len(X) // 2, 1000))
        steps_per_epoch = max(1, len(X) // 256)
        sae = train_sae(X, cfg, device=str(device), epochs=max(5, min(200, 3000 // steps_per_epoch)),
                        batch_size=256, lr=1e-3, seed=seed, log_every=10**9).cpu()
        z = encode_numpy(sae, X)
        eligible = (z > 0).sum(axis=0) >= rel.MIN_ACTIVE
        score, thr = rel.relevance_scores(z, R, eligible)
        drop = np.where(eligible & (score < rel.relevance_scores.null_median))[0]
        drop_rand = np.sort(np.random.default_rng(seed).choice(np.where(eligible)[0], len(drop), replace=False))
        W = sae.W_dec.detach().numpy()
        resid = X - (z @ W + sae.b_dec.detach().numpy())
        tot = float(((X - X.mean(0)) ** 2).sum())
        share = lambda a: round(float(((a - a.mean(0)) ** 2).sum()) / tot, 4)
        diag = {"n_eligible": int(eligible.sum()), "n_removed": int(len(drop)),
                "var_share_removed": share(z[:, drop] @ W[drop]), "fvu_sae": share(resid)}
        pruners[col] = (sae, drop, drop_rand, diag)
    return pruners


def apply_pruning(E: torch.Tensor, positions: Dict[str, int], pruners, which: str) -> torch.Tensor:
    """Subtract the removed latents' contributions from each free-text cell embedding (returns a copy)."""
    E = E.clone()
    for col, pos in positions.items():
        sae, drop_rel, drop_rand, _ = pruners[col]
        drop = drop_rel if which == "prune" else drop_rand
        if len(drop) == 0:
            continue
        X = E[:, pos, :].numpy()
        z = encode_numpy(sae, X)
        W = sae.W_dec.detach().numpy()
        E[:, pos, :] = torch.as_tensor(X - z[:, drop] @ W[drop])
    return E


# ------------------------------------------------------------------ one outer fold, all four variants
def run_fold(x_train: DataFrame, y_train: Series, x_test: DataFrame, y_test: Series, is_cls: bool,
             pretrain: Optional[str], device: torch.device, seed: int = SEED, output_dir: Optional[str] = None,
             time_limit: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
    from tabstar.preprocessing.splits import split_to_val
    from tabstar.preprocessing.texts import normalize_col_name
    from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

    x_fit, x_val, y_fit, y_val = split_to_val(x=x_train, y=y_train, is_cls=is_cls)
    cls = TabSTARClassifier if is_cls else TabSTARRegressor
    # output_dir MUST be unique per run: TabSTAR names its checkpoint folder by the current second, so parallel
    # jobs sharing a working directory would otherwise overwrite each other's checkpoints.
    star = cls(pretrain_dataset_or_path=pretrain, device=device, random_state=seed, is_paper_version=False,
               lora_batch=64, keep_model=False, output_dir=output_dir, time_limit=time_limit)
    star.fit(x_fit, y_fit, x_val, y_val)                                              # phase 1
    v = star.preprocessor_
    d_out = v.d_output
    # Classification: label codes + class probabilities. Regression: ORIGINAL-scale y (TabSTAR trains on clipped
    # z-scores; predictions are mapped back with its scaler so R^2 is computed on the real target).
    if is_cls:
        y_true = np.asarray(v.transform_target(y_test.copy()))
        to_out = lambda p: p
    else:
        y_true = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
        to_out = lambda z: v.target_transformer.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    preds = {"tabstar": to_out(np.asarray(star._infer(x_test)).squeeze())}

    model = star.model_.merge_and_unload()                                            # plain TabStarModel
    model.to(device)
    data = {k: v.transform(x, yy) for k, (x, yy) in
            {"fit": (x_fit, y_fit), "val": (x_val, y_val), "test": (x_test, None)}.items()}
    E = {k: cell_embeddings(model, d.x_txt, device) for k, d in data.items()}
    N = {k: torch.as_tensor(d.x_num, dtype=torch.float32) for k, d in data.items()}
    yv = {k: np.asarray(data[k].y) for k in ("fit", "val")}

    cols = x_txt_columns(v, x_fit)
    assert len(cols) == E["fit"].shape[1], (len(cols), E["fit"].shape)
    text = free_text_columns(x_fit)
    positions = {c: cols.index(normalize_col_name(c)) for c in text if normalize_col_name(c) in cols}

    R = rel.relevance_residual(x=x_fit, y=y_fit, text_cols=text, is_cls=is_cls, conditional=True)
    pruners = fit_pruners(E["fit"], positions, R, device, seed=seed)

    start = copy.deepcopy([m.state_dict() for m in (model.numerical_fusion, model.tabular_encoder,
                                                    model.cls_head, model.reg_head)])
    diag = {"n_fit": len(x_fit), "n_val": len(x_val), "n_test": len(x_test), "free_text_columns": list(positions),
            "pruning": {c: p[3] for c, p in pruners.items()}}
    for arm, which in (("tabstar-p2", None), ("tabstar-prune", "prune"), ("tabstar-rand", "rand")):
        for m, sd in zip((model.numerical_fusion, model.tabular_encoder, model.cls_head, model.reg_head), start):
            m.load_state_dict(sd)                                                     # identical starting point
        Ek = E if which is None else {k: apply_pruning(e, positions, pruners, which) for k, e in E.items()}
        diag[f"{arm}_val_loss"] = train_upper(model, Ek["fit"], N["fit"], yv["fit"], Ek["val"], N["val"], yv["val"],
                                              d_out, device, seed=seed)
        preds[arm] = to_out(predict_upper(model, Ek["test"], N["test"], d_out, device).squeeze())
    return preds, y_true, diag
