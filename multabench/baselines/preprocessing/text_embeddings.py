import os

from sklearn.decomposition import PCA

from multabench.constants import SEED

from multabench.preprocessing.discretize import discretize_numerical

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch

from multabench.e5.constants import E5_SMALL_V2, E5_TRANSFORMS, PRUNE_ENCODERS, RELEVANCE_ENCODERS, STACK_ENCODERS, TAR_STACK_ENCODERS, TF_IDF
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5

PCA_COMPONENTS = 30


class SkrubColumnEncoder:
    """Per-column text encoder using skrub.StringEncoder (TF-IDF + TruncatedSVD). CPU-only, no tokenizer needed."""

    def __init__(self, string_encoder: Any, col_name: str, n_components: int):
        self.string_encoder = string_encoder
        self.col_name = col_name
        self.n_components = n_components
        self.encoder = self  # so wrapper.encoder.transform(X) delegates to self.transform(X)

    def encode_texts(self, texts: list[str], device) -> np.ndarray:
        """Encode texts with skrub StringEncoder (device ignored — CPU-only)."""
        result = self.string_encoder.transform(pd.Series(texts, dtype=str))
        return np.asarray(result)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X  # encode_texts already returns final (N, n_components) array


class E5ColumnEncoder:
    """Per-column text encoder: holds E5 model, tokenizer (processor), and PCA (or identity). Uses passage: col_name: col_val format."""

    def __init__(self, model: Any, tokenizer: Any, encoder: Any, col_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.col_name = col_name
        self.n_components = encoder.n_components

    def encode_texts(self, texts: list[str], device: torch.device) -> np.ndarray:
        """Encode texts with this column's E5 model and tokenizer (passage: col_name: col_val)."""
        return encode_texts_with_e5(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            col_name=self.col_name,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.transform(X)


def fit_text_encoders_skrub(
    x: DataFrame,
    text_features_list: list[str],
) -> Dict[str, SkrubColumnEncoder]:
    """Fit one SkrubColumnEncoder per column using skrub.StringEncoder (TF-IDF + TruncatedSVD)."""
    from skrub import StringEncoder
    text_encoders: Dict[str, SkrubColumnEncoder] = {}
    for col in text_features_list:
        col_series = x[col].astype(str).fillna("")
        print(f"Fitting SkrubStringEncoder for column {col} with n_components={PCA_COMPONENTS} for {len(col_series)} texts")
        string_enc = StringEncoder(n_components=PCA_COMPONENTS)
        string_enc.fit(col_series)
        text_encoders[str(col)] = SkrubColumnEncoder(
            string_encoder=string_enc,
            col_name=str(col),
            n_components=PCA_COMPONENTS,
        )
    return text_encoders


def fit_text_encoders_vanilla(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    e5_model_name: str = E5_SMALL_V2,
) -> Dict[str, E5ColumnEncoder]:
    """Fit one E5ColumnEncoder per column using shared vanilla E5 + PCA per column. Uses passage: col_name: col_val format."""
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        print(f"Fitting E5ColumnEncoder for column {col} with model {e5_model_name} for {len(texts)} texts")
        col_embeddings = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=str(col))
        encoder = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        encoder.fit(col_embeddings)
        text_encoders[str(col)] = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


def fit_text_encoders_transformed(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    transform_name: str,
) -> Dict[str, E5ColumnEncoder]:
    """Frozen E5-small per column, with PCA(30) replaced by an SAE (or identity) transform fitted on train rows only."""
    from multabench.e5.constants import FOLD_SAE, PCA as PCA_MARKER
    from multabench.sae.feature_transform import FoldSAETransform, IdentityTransform, SAEFeatureTransform, load_sae_checkpoint

    checkpoint, binary, size = E5_TRANSFORMS[transform_name]
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        col_embeddings = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=str(col))
        if checkpoint is None:
            encoder = IdentityTransform().fit(col_embeddings)
        elif checkpoint == FOLD_SAE:
            encoder = FoldSAETransform(binary=binary, n_latents=size).fit(col_embeddings)
        elif checkpoint == PCA_MARKER:
            encoder = PCA(n_components=min(size, *col_embeddings.shape), random_state=SEED).fit(col_embeddings)
        else:
            encoder = SAEFeatureTransform(sae=load_sae_checkpoint(checkpoint), binary=binary).fit(col_embeddings)
        print(f"Fitted {transform_name} for column {col}: {encoder.n_components} features from {len(texts)} texts")
        text_encoders[str(col)] = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


def fit_text_encoders_relevance(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    y: Series,
    is_cls: bool,
    variant: str,
) -> Dict[str, E5ColumnEncoder]:
    """Frozen E5 per column; PCA(30) replaced by target-relevant concept filtering (or one of its controls)."""
    from multabench.sae import relevance as rel
    conditional = variant in ("sae_cond", "sae_cond_add", "sae_rand", "dim_cond", "v2_cond", "v2_rand")
    R = rel.relevance_residual(x=x, y=y, text_cols=text_features_list, is_cls=is_cls, conditional=conditional)
    T, _ = rel.target_matrix(y, is_cls)
    model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        emb = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=str(col))
        if variant.startswith("v2_"):
            enc = rel.SAERelevanceTransform(R, kind="random" if variant == "v2_rand" else "sae",
                                            rule="drop_irrelevant", n_latents=512, k=32)
        elif variant in ("sae_cond", "sae_uni"):
            enc = rel.SAERelevanceTransform(R, kind="sae")
        elif variant == "sae_cond_add":
            enc = rel.SAERelevanceTransform(R, kind="sae", mode="add")
        elif variant == "sae_rand":
            enc = rel.SAERelevanceTransform(R, kind="random")
        elif variant == "dim_cond":
            enc = rel.DimRelevanceTransform(R)
        else:
            enc = rel.PLSTransform(T)
        enc.fit(emb, texts=texts)
        print(f"Fitted relevance[{variant}] for column {col}: {enc.n_components} features; {enc.diag.get('n_kept', '-')} kept")
        text_encoders[str(col)] = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=enc, col_name=str(col))
    return text_encoders


_TUNED_MEMO: Dict[str, Any] = {}


def _tuned_key(train_texts, train_y, val_texts, val_y, model_name, kwargs) -> str:
    """Identity of a TAR fine-tuning run: its exact inputs. Only the latest model is kept (one fold at a time)."""
    import hashlib
    h = hashlib.sha1()
    for part in ("\x1f".join(train_texts), "\x1f".join(val_texts), str(list(train_y)), str(list(val_y)),
                 model_name, repr(sorted((k, repr(v)) for k, v in kwargs.items()))):
        h.update(part.encode("utf-8", "replace"))
    return h.hexdigest()


def fit_text_encoders_pruned(x, text_features_list, device, y, is_cls, tuned, kind, e5_train_kwargs, d_output):
    """Frozen or TAR-tuned E5 per column; PCA-30 replaced by sibling-aware SAE concept pruning + PCA-30."""
    from multabench.sae import relevance as rel
    if kind == "xfit":
        return fit_text_encoders_xfit(x=x, text_features_list=text_features_list, device=device, y=y, is_cls=is_cls,
                                      e5_train_kwargs=e5_train_kwargs, d_output=d_output)
    tab = rel.tabular_frame(x, text_features_list)
    T, codes = rel.target_matrix(y, is_cls)
    target = codes if is_cls else T[:, 0]
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    def factory(col, val_mask=None):
        if kind == "heldout":
            return rel.HeldOutPruneSAETransform(tab=tab, target=target, is_cls=is_cls, val_mask=val_mask, device=sae_device)
        return rel.GroupShadowSAETransform(tab=tab, target=target, is_cls=is_cls, kind=kind, device=sae_device)
    if tuned:
        encs = fit_text_encoders_tuned(x=x, text_features_list=text_features_list, device=device, y=y,
                                       e5_train_kwargs=e5_train_kwargs or {}, is_cls=is_cls, d_output=d_output,
                                       transform_factory=factory)
    else:
        model, tokenizer = get_vanilla_e5(device, model_name=E5_SMALL_V2)
        encs = {}
        for col in text_features_list:
            texts = x[col].astype(str).fillna("").tolist()
            emb = encode_texts_with_e5(texts=texts, model=model, tokenizer=tokenizer, device=device, col_name=str(col))
            encs[col] = E5ColumnEncoder(model=model, tokenizer=tokenizer, col_name=str(col),
                                        encoder=factory(col).fit(emb, texts=texts))
    for col, e in encs.items():
        d = e.encoder.diag
        if kind == "heldout":
            print(f"Fitted pruning[tar/heldout] for {col}: chose {d['chosen_fraction']:.0%} -> {d['n_removed']}/"
                  f"{d['n_eligible']} concepts removed, var removed {d['var_share_removed']}, val loss {d['val_loss']}")
            continue
        print(f"Fitted pruning[{'tar' if tuned else 'frozen'}/{kind}] for {col}: {d['n_removed']}/{d['n_eligible']} "
              f"concepts removed ({d['n_groups_dropped']}/{d['n_groups']} groups), var removed {d['var_share_removed']}")
    return encs


def fit_text_encoders_xfit(x, text_features_list, device, y, is_cls, e5_train_kwargs, d_output, n_inner: int = 3):
    """TAR + SAE pruning whose removal amount is chosen by nested cross-fitting, so every judgment is made on rows
    the judging encoder never saw.

    Inner loop (training rows only), for each of n_inner splits (A = fit part, B = held-out part):
      1. fine-tune TAR on A only (same recipe as Joint TAR)
      2. per text column: SAE + harm ranking on A's embeddings (variance taken / joint shadow-corrected gain)
      3. for each removal fraction f: drop the top-f concepts, PCA-30 per column (fit on A)
         -> LightGBM on [table | all text columns] fit on A -> loss on B   (the same model shape as the final one)
    The fraction with the lowest total held-out loss is chosen (0% = plain TAR; a larger removal must win by 0.1%).
    Outer step: the fold's usual TAR (shared with the e5-small-tar arm) -> SAE + ranking on all training rows ->
    drop the chosen fraction -> PCA-30. Test rows are never used."""
    from sklearn.model_selection import KFold, StratifiedKFold
    from multabench.sae import relevance as rel
    tab = rel.tabular_frame(x, text_features_list)
    T, codes = rel.target_matrix(y, is_cls)
    target = codes if is_cls else T[:, 0]
    tab_np, cat_idx = rel._tab_numeric(tab)
    sae_device = "cuda" if torch.cuda.is_available() else "cpu"
    fractions = rel.CV_FRACTIONS
    strat = is_cls and np.bincount(codes).min() >= n_inner
    splits = (StratifiedKFold(n_inner, shuffle=True, random_state=SEED).split(tab_np, codes) if strat
              else KFold(n_inner, shuffle=True, random_state=SEED).split(tab_np))
    losses = {f: 0.0 for f in fractions}
    for k, (tr, ho) in enumerate(splits):
        print(f"✂️ cross-fit {k + 1}/{n_inner}: TAR on {len(tr)} rows, judging on {len(ho)} unseen rows")
        model_k, tok_k = fit_text_encoders_tuned(x=x.iloc[tr], text_features_list=text_features_list, device=device,
                                                 y=y.iloc[tr], e5_train_kwargs=e5_train_kwargs or {}, is_cls=is_cls,
                                                 d_output=d_output, use_memo=False, return_model=True)
        parts_tr = {f: [tab_np[tr]] for f in fractions}
        parts_ho = {f: [tab_np[ho]] for f in fractions}
        for col in text_features_list:
            texts = x[col].astype(str).fillna("").tolist()
            emb = encode_texts_with_e5(texts=texts, model=model_k, tokenizer=tok_k, device=device, col_name=str(col))
            pr = rel.RankedSAEPruner(tab.iloc[tr], target[tr], is_cls, device=sae_device).fit(emb[tr])
            for f in fractions:
                a, b = pr.filtered(emb[tr], f), pr.filtered(emb[ho], f)
                pca = PCA(n_components=min(PCA_COMPONENTS, a.shape[1]), random_state=SEED).fit(a)
                parts_tr[f].append(pca.transform(a))
                parts_ho[f].append(pca.transform(b))
        for f in fractions:
            m = rel._lgbm(is_cls, SEED).fit(np.hstack(parts_tr[f]), target[tr], categorical_feature=cat_idx or "auto")
            losses[f] += rel._loss(m, np.hstack(parts_ho[f]), target[ho], is_cls) * len(ho) / len(x)
        del model_k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = 0.0
    for f in fractions[1:]:
        if losses[f] < min(losses[best], losses[0.0] * (1 - rel.CV_TOL)):
            best = f
    loss_str = {str(f): round(v, 5) for f, v in losses.items()}
    print(f"✂️ cross-fitted held-out loss per removal fraction: {loss_str} -> chose {best:.0%}")
    factory = lambda col, val_mask=None: rel.FixedFractionTransform(
        rel.RankedSAEPruner(tab, target, is_cls, device=sae_device), best, inner_losses=loss_str)
    encs = fit_text_encoders_tuned(x=x, text_features_list=text_features_list, device=device, y=y,
                                   e5_train_kwargs=e5_train_kwargs or {}, is_cls=is_cls, d_output=d_output,
                                   transform_factory=factory)
    for col, e in encs.items():
        d = e.encoder.diag
        print(f"Fitted pruning[tar/xfit] for {col}: {best:.0%} -> {d['n_removed']}/{d['n_eligible']} concepts removed, "
              f"var removed {d['var_share_removed']}")
    return encs


def fit_text_encoders_tuned(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    y: Series,
    e5_train_kwargs: Dict[str, Any],
    is_cls: bool,
    d_output: int,
    e5_model_name: str = E5_SMALL_V2,
    transform_factory=None,
    use_memo: bool = True,
    return_model: bool = False,
):
    """Fit a single E5 model for all text columns with passage: col_name: col_val format. Each column gets an E5ColumnEncoder sharing the same tuned model."""
    from transformers import AutoTokenizer

    from multabench.e5.e5_finetune import finetune_e5_with_lora
    from multabench.preprocessing.splits import split_to_val

    if not is_cls:
        y = discretize_numerical(y, n_bins=20)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x, y=y, is_cls=True)
    kwargs = e5_train_kwargs or {}

    # Build combined train/val: for each row, for each text col, create (col_name: col_val, label)
    train_texts: list[str] = []
    train_y: list = []
    for idx in x_tr.index:
        row_y = y_tr.loc[idx]
        for col in text_features_list:
            val = str(x_tr.loc[idx, col]).strip() if pd.notna(x_tr.loc[idx, col]) else ""
            train_texts.append(f"{col}: {val}")
            train_y.append(row_y)

    val_texts: list[str] = []
    val_y: list = []
    for idx in x_val.index:
        row_y = y_val.loc[idx]
        for col in text_features_list:
            val = str(x_val.loc[idx, col]).strip() if pd.notna(x_val.loc[idx, col]) else ""
            val_texts.append(f"{col}: {val}")
            val_y.append(row_y)

    train_y_arr = np.array(train_y)
    val_y_arr = np.array(val_y)

    key = _tuned_key(train_texts, train_y_arr, val_texts, val_y_arr, e5_model_name, kwargs)
    if use_memo and _TUNED_MEMO.get("key") == key:   # same fold, another TAR-based arm: reuse the identical fine-tuned model
        tuned_model, tuned_tokenizer = _TUNED_MEMO["model"]
        print(f"Reusing the fine-tuned E5 of this fold ({len(train_texts)} train examples)")
    else:
        if use_memo:
            _TUNED_MEMO.clear()
        tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
        print(f"Finetuning single E5 ({e5_model_name}) for {len(text_features_list)} columns with {len(train_texts)} train examples (passage: col_name: col_val)")
        tuned_model, tuned_tokenizer = finetune_e5_with_lora(
            train_texts=train_texts,
            train_y=train_y_arr,
            val_texts=val_texts,
            val_y=val_y_arr,
            device=device,
            tokenizer=tokenizer,
            model_name=e5_model_name,
            **kwargs,
        )
        tuned_model.to(device)
        if use_memo:
            _TUNED_MEMO.update(key=key, model=(tuned_model, tuned_tokenizer))
    if return_model:
        return tuned_model, tuned_tokenizer
    if transform_factory is not None:
        val_mask = x.index.isin(x_val.index)   # TAR's early-stopping rows: never trained on
        return {col: E5ColumnEncoder(model=tuned_model, tokenizer=tuned_tokenizer, col_name=str(col),
                                     encoder=transform_factory(col, val_mask).fit(
                                         encode_texts_with_e5(texts=x[col].astype(str).fillna("").tolist(),
                                                              model=tuned_model, tokenizer=tuned_tokenizer,
                                                              device=device, col_name=str(col)),
                                         texts=x[col].astype(str).fillna("").tolist()))
                for col in text_features_list}

    text_encoders: Dict[str, E5ColumnEncoder] = {}
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        col_embeddings = encode_texts_with_e5(texts=texts, model=tuned_model, tokenizer=tuned_tokenizer, device=device, col_name=str(col))
        encoder = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        encoder.fit(col_embeddings)
        text_encoders[col] = E5ColumnEncoder(model=tuned_model, tokenizer=tuned_tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


def fit_text_encoders(
    x: DataFrame,
    text_features: Set[str],
    device: torch.device,
    y: Optional[Series] = None,
    tune_e5: bool = False,
    e5_train_kwargs: Optional[Dict[str, Any]] = None,
    is_cls: bool = True,
    d_output: int = 2,
    e5_model_name: str = E5_SMALL_V2,
) -> Dict[str, E5ColumnEncoder]:
    """
    Fit one E5 model per text column (or vanilla E5 shared across columns when not tuning).
    Each column gets an E5ColumnEncoder wrapper holding model, tokenizer, and PCA.
    Returns text_encoders mapping column -> E5ColumnEncoder.
    """
    text_features_list = sorted(text_features)
    if not text_features_list:
        return {}
    if e5_model_name == TF_IDF:
        return fit_text_encoders_skrub(
            x=x,
            text_features_list=text_features_list,
        )
    if e5_model_name in PRUNE_ENCODERS:
        assert y is not None, "concept pruning needs the target"
        tuned, kind = PRUNE_ENCODERS[e5_model_name]
        return fit_text_encoders_pruned(x=x, text_features_list=text_features_list, device=device, y=y, is_cls=is_cls,
                                        tuned=tuned, kind=kind, e5_train_kwargs=e5_train_kwargs, d_output=d_output)
    if e5_model_name in RELEVANCE_ENCODERS:
        assert y is not None, "relevance filtering needs the target"
        return fit_text_encoders_relevance(x=x, text_features_list=text_features_list, device=device, y=y,
                                           is_cls=is_cls, variant=RELEVANCE_ENCODERS[e5_model_name])
    if e5_model_name in TAR_STACK_ENCODERS:
        # Joint TAR unchanged (same fine-tuned E5 as the e5-small-tar arm of this fold) + stacked text predictions
        from multabench.stack.text_stack import STACK_KEY, StackEncoder
        assert y is not None, "text stacking needs the target"
        kind, n_splits = TAR_STACK_ENCODERS[e5_model_name]
        text_encoders = fit_text_encoders_tuned(x=x, text_features_list=text_features_list, device=device, y=y,
                                                e5_train_kwargs=e5_train_kwargs or {}, is_cls=is_cls, d_output=d_output)
        text_encoders[STACK_KEY] = StackEncoder(kind=kind, text_cols=text_features_list, is_cls=is_cls, device=device,
                                                n_splits=n_splits, e5_train_kwargs=e5_train_kwargs).fit(x, y)
        return text_encoders
    if e5_model_name in STACK_ENCODERS:
        from multabench.stack.text_stack import STACK_KEY, StackEncoder
        assert y is not None, "text stacking needs the target"
        kind, n_splits = STACK_ENCODERS[e5_model_name]
        text_encoders = fit_text_encoders_vanilla(x=x, text_features_list=text_features_list, device=device,
                                                  e5_model_name=E5_SMALL_V2)
        text_encoders[STACK_KEY] = StackEncoder(kind=kind, text_cols=text_features_list, is_cls=is_cls, device=device,
                                                n_splits=n_splits, e5_train_kwargs=e5_train_kwargs).fit(x, y)
        return text_encoders
    if e5_model_name in E5_TRANSFORMS:
        assert not tune_e5, "SAE/identity transforms are frozen-only"
        return fit_text_encoders_transformed(
            x=x,
            text_features_list=text_features_list,
            device=device,
            transform_name=e5_model_name,
        )
    if tune_e5 and y is not None:
        return fit_text_encoders_tuned(
            x=x,
            text_features_list=text_features_list,
            device=device,
            y=y,
            e5_train_kwargs=e5_train_kwargs or {},
            is_cls=is_cls,
            d_output=d_output,
            e5_model_name=e5_model_name,
        )
    return fit_text_encoders_vanilla(
        x=x,
        text_features_list=text_features_list,
        device=device,
        e5_model_name=e5_model_name,
    )


def transform_text_features(
    x: DataFrame,
    text_encoders: Dict[str, E5ColumnEncoder],
    device: torch.device,
) -> DataFrame:
    from multabench.stack.text_stack import STACK_KEY
    stack_df = text_encoders[STACK_KEY].transform(x) if STACK_KEY in text_encoders else None
    for text_col, wrapper in text_encoders.items():
        if text_col == STACK_KEY:
            continue
        texts = x[text_col].astype(str).fillna("").tolist()
        embeddings = wrapper.encode_texts(texts, device)
        n_components = wrapper.n_components
        pca_vec = wrapper.encoder.transform(embeddings)
        pca_cols = [f"{text_col}_txt_pca_{i}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_vec, index=x.index, columns=pca_cols)
        cols_before = len(x.columns)
        x = x.drop(columns=[text_col])
        x = pd.concat([x, pca_df], axis=1)
        assert len(x.columns) == cols_before + n_components - 1
    if stack_df is not None:
        x = pd.concat([x, stack_df], axis=1)
    return x
