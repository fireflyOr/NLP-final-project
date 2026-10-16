"""Tests for the SAE text transform. CPU-only; E5 is replaced by a deterministic fake encoder."""
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch

from multabench.sae.feature_transform import IdentityTransform, SAEFeatureTransform
from multabench.sae.topk_sae import SAEConfig, TopKSAE, encode_numpy, random_sae, sae_stats, train_sae


def _sparse_data(n=20_000, d=64, n_atoms=512, k_true=4, seed=0):
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(n_atoms, d))
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    codes = np.zeros((n, n_atoms))
    for i in range(n):
        idx = rng.choice(n_atoms, k_true, replace=False)
        codes[i, idx] = rng.uniform(0.5, 1.5, k_true)
    return (codes @ atoms).astype(np.float32)


@pytest.fixture(scope="module")
def trained():
    x = _sparse_data()
    cfg = SAEConfig(d_in=64, n_latents=1024, k=8, dead_after=20_000)
    sae = train_sae(x, cfg, epochs=30, batch_size=1024, lr=1e-3, log_every=10_000)
    return sae, x


def test_topk_exact_sparsity_and_unit_decoder(trained):
    sae, x = trained
    z = encode_numpy(sae, x[:500])
    assert ((z > 0).sum(1) <= sae.cfg.k).all()
    norms = sae.W_dec.detach().norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_training_beats_random_control(trained):
    sae, x = trained
    s_trained = sae_stats(sae, x[-2000:])
    s_random = sae_stats(random_sae(x, sae.cfg), x[-2000:])
    assert s_trained["fvu"] < 0.35, s_trained
    assert s_trained["fvu"] < 0.5 * s_random["fvu"], (s_trained, s_random)


def test_save_load_roundtrip(trained, tmp_path):
    sae, x = trained
    path = tmp_path / "s.pt"
    sae.save(str(path), extra={"a": 1})
    sae2 = TopKSAE.load(str(path))
    np.testing.assert_allclose(encode_numpy(sae, x[:100]), encode_numpy(sae2, x[:100]), atol=1e-6)
    assert sae2.extra == {"a": 1}


def test_feature_transform_selection(trained):
    sae, x = trained
    t = SAEFeatureTransform(sae, binary=True, min_count=5, max_latents=100).fit(x[:3000])
    assert t.n_components == 100
    out = t.transform(x[3000:3500])
    assert out.shape == (500, 100)
    assert set(np.unique(out)) <= {0.0, 1.0}
    cont = SAEFeatureTransform(sae, binary=False, min_count=10**9, max_latents=None).fit(x[:300])
    assert cont.n_components == 1  # degenerate fallback keeps one latent
    assert IdentityTransform().fit(x).transform(x[:3]).shape == (3, 64)


# ---------------- end-to-end through the MulTaBench pipeline with a fake E5 ----------------

WORDS = ["great", "terrible", "fast", "slow", "cheap", "expensive", "red", "blue", "fresh", "old",
         "service", "price", "quality", "delivery", "taste", "design", "size", "color", "staff", "location"]


def _fake_embed(texts, col_name, model=None, tokenizer=None, device=None, **kw):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in str(t).split():
            seed = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            out[i] += np.random.default_rng(seed).normal(size=384)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-8)
    return out


@pytest.fixture()
def fake_e5(monkeypatch, tmp_path):
    import multabench.baselines.preprocessing.text_embeddings as te
    import multabench.sae.feature_transform as ft
    monkeypatch.setattr(te, "encode_texts_with_e5", _fake_embed)
    monkeypatch.setattr(te, "get_vanilla_e5", lambda device, model_name=None: (None, None))
    rng = np.random.default_rng(1)
    corpus = [" ".join(rng.choice(WORDS, rng.integers(3, 12))) for _ in range(20_000)]
    emb = _fake_embed(corpus, "text")
    cfg = SAEConfig(d_in=384, n_latents=512, k=8, dead_after=10_000)
    train_sae(emb, cfg, epochs=5, batch_size=1024, lr=1e-3, log_every=10_000).save(str(tmp_path / "e5-small-topk.pt"))
    random_sae(emb, cfg).save(str(tmp_path / "e5-small-random.pt"))
    monkeypatch.setattr(ft, "DEFAULT_SAE_DIR", str(tmp_path))
    monkeypatch.setattr(ft.load_sae_checkpoint, "__defaults__", (str(tmp_path),))
    ft._SAE_CACHE.clear()


def _toy_dataset(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    review = [" ".join(rng.choice(WORDS, rng.integers(4, 10))) + f" item{i}" for i in range(n)]
    price = rng.normal(size=n)
    y = ((np.array(["great" in r for r in review]) + (price > 0)) >= 1).astype(int)
    x = pd.DataFrame({"price": price, "review": review})
    return x, pd.Series(y, name="target")


@pytest.mark.parametrize("encoder", ["e5-small", "e5-small-full", "e5-small-sae", "e5-small-sae-bin", "e5-small-rsae-bin", "e5-small-fsae", "e5-small-fsae-bin", "e5-small-fsae50", "e5-small-pca50"])
def test_pipeline_runs_with_each_encoder(fake_e5, encoder):
    from multabench.baselines.lgbm import LightGBM
    from multabench.datasets.objects import SupervisedTask
    from multabench.e5.constants import TEXT_ENCODERS
    x, y = _toy_dataset()
    enc = TEXT_ENCODERS[encoder]
    model = LightGBM(problem_type=SupervisedTask.BINARY, device=torch.device("cpu"), verbose=False,
                     e5_model_name=enc.encoder_name)
    model.fit(x.iloc[:1000], y.iloc[:1000])
    assert "review" in model.text_transformers, "review should be detected as a text column"
    n_feats = model.text_transformers["review"].n_components
    expected = {"e5-small": 30, "e5-small-full": 384, "e5-small-fsae": 30, "e5-small-fsae-bin": 30, "e5-small-fsae50": 50, "e5-small-pca50": 50}
    if encoder in expected:
        assert n_feats == expected[encoder]
    else:
        assert 1 <= n_feats <= 512
    pred = model.predict(x.iloc[1000:])
    assert pred.shape == (500,) and np.isfinite(pred).all()


# ---------------- text stacking (out-of-fold text-model predictions) ----------------

def _toy_task(task: str, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    review = [" ".join(rng.choice(WORDS, rng.integers(4, 10))) + f" item{i}" for i in range(n)]
    title = [" ".join(rng.choice(WORDS, 3)) for _ in range(n)]
    price = rng.normal(size=n)
    good = np.array(["great" in r for r in review], dtype=float)
    fast = np.array(["fast" in t for t in title], dtype=float)
    if task == "bin":
        y = ((good + (price > 0)) >= 1).astype(int)
    elif task == "mul":
        y = (good + fast + (price > 0.5)).astype(int)  # 4 classes
    else:
        y = 2 * good + fast + 0.5 * price + rng.normal(scale=0.1, size=n)
    x = pd.DataFrame({"price": price, "review": review, "title": title})
    return x, pd.Series(y, name="target")


@pytest.fixture()
def fake_finetune(monkeypatch):
    """Replace LoRA fine-tuning with a stub whose 'head' scores texts by keywords."""
    from sklearn.preprocessing import LabelEncoder
    import multabench.e5.e5_finetune as ef
    import multabench.stack.text_stack as ts
    import transformers

    def finetune(**kw):
        le = LabelEncoder().fit(np.asarray(kw["train_y"]).astype(str))
        return len(le.classes_), None, le

    def logits(model, tokenizer, texts, device, **kw):
        n_cls = model
        out = np.zeros((len(texts), n_cls), dtype=np.float32)
        for i, t in enumerate(texts):
            score = ("great" in t) + ("fast" in t)
            out[i, min(score, n_cls - 1)] = 3.0
        return out

    monkeypatch.setattr(ef, "finetune_e5_with_lora", finetune)
    monkeypatch.setattr(ts, "finetuned_logits", logits)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: None))


@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
@pytest.mark.parametrize("encoder", ["e5-small-stack", "e5-small-tar-stack"])
def test_stack_pipeline(fake_e5, fake_finetune, task, encoder):
    from multabench.baselines.lgbm import LightGBM
    from multabench.datasets.objects import SupervisedTask
    from multabench.e5.constants import TEXT_ENCODERS
    from multabench.stack.text_stack import STACK_KEY
    x, y = _toy_task(task)
    problem = {"bin": SupervisedTask.BINARY, "mul": SupervisedTask.MULTICLASS, "reg": SupervisedTask.REGRESSION}[task]
    enc = TEXT_ENCODERS[encoder]
    model = LightGBM(problem_type=problem, device=torch.device("cpu"), verbose=False, tune_e5=True,
                     e5_train_kwargs={}, e5_model_name=enc.encoder_name)
    model.fit(x.iloc[:1000], y.iloc[:1000])
    stack = model.text_transformers[STACK_KEY]
    assert {"review", "title"} <= set(model.text_transformers)
    expected_cols = {"bin": 1, "mul": 4, "reg": 1}[task]
    assert stack.n_components == expected_cols
    n_val = 1000 - len(stack.oof)  # LightGBM holds out a validation split before fitting the preprocessor
    assert n_val > 0
    # training rows used OOF features; only the validation rows went through the ensemble so far
    assert stack.n_ensemble_rows == n_val
    pred = model.predict(x.iloc[1000:])
    assert stack.n_ensemble_rows == n_val + 500
    assert np.isfinite(pred).all()
    feats = stack.transform(x.iloc[1000:])
    assert feats.shape == (500, expected_cols) and np.isfinite(feats.to_numpy()).all()
    # the stacked feature should carry real signal on held-out rows
    if task == "bin":
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y.iloc[1000:], feats.iloc[:, 0]) > 0.6
    elif task == "reg":
        assert np.corrcoef(y.iloc[1000:], feats.iloc[:, 0])[0, 1] > 0.3


def test_stack_oof_differs_from_in_sample(fake_e5):
    """OOF predictions for training rows must differ from what a model trained on them would output."""
    from multabench.stack.text_stack import StackEncoder
    x, y = _toy_task("bin", n=600)
    st = StackEncoder(kind="frozen", text_cols=["review", "title"], is_cls=True, device=torch.device("cpu"),
                      n_splits=5).fit(x, y)
    oof = st.transform(x).to_numpy()
    assert st.n_ensemble_rows == 0
    st.oof = {}  # force ensemble predictions for the same rows
    ens = st.transform(x).to_numpy()
    assert not np.allclose(oof, ens)


# ---------------- target-relevant concept filtering ----------------

def _atom_data(n=6000, d=256, n_atoms=400, seed=0):
    """Sparse-dictionary embeddings: strong irrelevant atoms dominate the variance; y depends on one weak atom
    (atom 0) or on a tabular feature. Atom 1 copies the tabular feature (relevant alone, redundant given the table)."""
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(n_atoms, d)); atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    tab = rng.normal(size=n)
    Z = np.zeros((n, n_atoms))
    for i in range(n):
        Z[i, rng.choice(np.arange(2, n_atoms), 4, replace=False)] = rng.uniform(2, 4, 4)   # strong, irrelevant
    has0 = rng.random(n) < 0.3
    Z[has0, 0] = rng.uniform(1.5, 2.5, has0.sum())                                       # weaker, relevant
    has1 = tab > 0.5
    Z[has1, 1] = rng.uniform(1.5, 2.5, has1.sum())                                       # weaker, copies the table
    y = ((has0 | (tab > 1.2)) ^ (rng.random(n) < 0.05)).astype(int)
    return (Z @ atoms).astype(np.float32), atoms, tab, pd.Series(y, name="y")


def test_relevance_selects_weak_relevant_concept():
    from multabench.sae import relevance as rel
    emb, atoms, tab, y = _atom_data()
    x = pd.DataFrame({"tab": tab, "txt": [f"t{i}" for i in range(len(y))]})
    out = {}
    for conditional in (False, True):
        R = rel.relevance_residual(x=x, y=y, text_cols=["txt"], is_cls=True, conditional=conditional)
        t = rel.SAERelevanceTransform(R, kind="sae").fit(emb)
        D = t.W_dec[t.keep] / np.linalg.norm(t.W_dec[t.keep], axis=1, keepdims=True)
        out[conditional] = (np.abs(D @ atoms[0]).max(), np.abs(D @ atoms[1]).max() if len(D) else 0, t)
    # both variants find the weak relevant atom; only the univariate one also keeps the table-redundant atom
    assert out[True][0] > 0.8 and out[False][0] > 0.8, {k: v[:2] for k, v in out.items()}
    assert out[False][1] > 0.8 and out[True][1] < 0.8, {k: v[:2] for k, v in out.items()}
    t = out[True][2]
    assert {"var_share_kept", "var_share_sae_error", "n_kept", "threshold", "top"} <= set(t.diag)
    assert 1 <= t.diag["n_kept"] <= 20  # a handful of concepts, not a large fraction of the 256


@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
@pytest.mark.parametrize("encoder", ["e5-small-relsae2", "e5-small-relsae2-rand", "e5-small-relsae", "e5-small-relsae-add", "e5-small-relsae-uni", "e5-small-relsae-rand",
                                     "e5-small-reldim", "e5-small-pls"])
def test_relevance_pipeline(fake_e5, task, encoder):
    from multabench.baselines.lgbm import LightGBM
    from multabench.datasets.objects import SupervisedTask
    from multabench.e5.constants import TEXT_ENCODERS
    x, y = _toy_task(task)
    problem = {"bin": SupervisedTask.BINARY, "mul": SupervisedTask.MULTICLASS, "reg": SupervisedTask.REGRESSION}[task]
    model = LightGBM(problem_type=problem, device=torch.device("cpu"), verbose=False, tune_e5=True,
                     e5_train_kwargs={}, e5_model_name=TEXT_ENCODERS[encoder].encoder_name)
    model.fit(x.iloc[:1000], y.iloc[:1000])
    for col in ("review", "title"):
        assert 1 <= model.text_transformers[col].n_components <= 30
    pred = model.predict(x.iloc[1000:])
    assert np.isfinite(pred).all() and len(pred) == 500


def test_v2_removes_only_irrelevant():
    from multabench.sae import relevance as rel
    emb, atoms, tab, y = _atom_data()
    x = pd.DataFrame({"tab": tab, "txt": ["t"] * len(y)})
    R = rel.relevance_residual(x=x, y=y, text_cols=["txt"], is_cls=True, conditional=True)
    t = rel.SAERelevanceTransform(R, rule="drop_irrelevant", n_latents=512, k=32).fit(emb)
    D = t.W_dec[t.keep_idx] / np.linalg.norm(t.W_dec[t.keep_idx], axis=1, keepdims=True)
    assert np.abs(D @ atoms[0]).max() > 0.8                       # the relevant concept survives
    assert 0.2 * t.diag["n_eligible"] < t.diag["n_removed"] < 0.8 * t.diag["n_eligible"]  # gentle: roughly half


# ---------------- TabSTAR + SAE pruning (phase 1 mocked: tiny random TabSTAR, no download) ----------------

@pytest.fixture()
def tiny_tabstar(monkeypatch, tmp_path):
    import transformers
    from transformers import BertConfig, BertModel, BertTokenizerFast
    import tabstar.arch.arch as arch
    from tabstar.tabstar_model import BaseTabSTAR
    from peft import LoraConfig, get_peft_model
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "predictive", "feature", "value", "target", ":",
             "review", "title", "price", "y", "to", "quantile", "lower", "higher", "than", "unknown", "numerical"] + WORDS
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=384, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
    monkeypatch.setattr(arch.AutoModel, "from_pretrained", staticmethod(lambda *a, **k: BertModel(cfg)))
    monkeypatch.setattr(arch.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: tok))

    def fake_fit(self, X, y, x_val=None, y_val=None):
        self._prepare_for_train(X, y, x_val, y_val)
        torch.manual_seed(0)
        m = arch.TabStarModel(arch.TabStarConfig())
        self.model_ = get_peft_model(m, LoraConfig(r=4, target_modules=["linear1", "linear2"])).to(self.device)
    monkeypatch.setattr(BaseTabSTAR, "fit", fake_fit)


@pytest.mark.parametrize("task", ["bin", "mul", "reg"])
def test_star_prune_fold_no_leakage(tiny_tabstar, monkeypatch, task):
    import multabench.star.prune_star as ps
    seen = {}
    real_fit_pruners, real_resid = ps.fit_pruners, ps.rel.relevance_residual
    monkeypatch.setattr(ps, "fit_pruners", lambda E, *a, **k: (seen.setdefault("sae_rows", len(E)), real_fit_pruners(E, *a, **k))[1])
    monkeypatch.setattr(ps.rel, "relevance_residual", lambda x, *a, **k: (seen.setdefault("rel_rows", len(x)), real_resid(x, *a, **k))[1])
    monkeypatch.setattr(ps, "P2_EPOCHS", 2)
    x, y = _toy_task(task, n=800)
    tr, te = slice(0, 600), slice(600, None)
    preds, y_true, diag = ps.run_fold(x.iloc[tr], y.iloc[tr], x.iloc[te], y.iloc[te], is_cls=task != "reg",
                                      pretrain=None, device=torch.device("cpu"))
    # leakage invariants: SAE and relevance see only the fit rows (not val, not test)
    assert seen["sae_rows"] == diag["n_fit"] == seen["rel_rows"]
    assert diag["n_fit"] + diag["n_val"] == 600 and diag["n_test"] == 200
    assert set(diag["free_text_columns"]) == {"review", "title"}
    assert set(preds) == {"tabstar", "tabstar-p2", "tabstar-prune", "tabstar-rand"}
    for v, p in preds.items():
        assert len(p) == 200 and np.isfinite(p).all(), v
    if task == "reg":
        assert np.allclose(y_true, y.iloc[te].to_numpy())          # metric on the original target scale
    for col, d in diag["pruning"].items():
        assert 0 <= d["n_removed"] <= d["n_eligible"] <= 512


def test_star_runner_writes_results(tiny_tabstar, monkeypatch, tmp_path):
    import sys, types, json, glob
    import multabench.star.prune_star as ps
    import multabench.benchmark.load as L
    from multabench.benchmark.splits import get_split
    from multabench.datasets.objects import SupervisedTask
    monkeypatch.setattr(ps, "P2_EPOCHS", 1)
    x, y = _toy_task("bin", n=700)
    ds = types.SimpleNamespace(x=x, y=y, is_cls=True, task_type=SupervisedTask.BINARY)
    monkeypatch.setattr(L, "load_multabench_dataset", lambda d: ds)
    monkeypatch.setattr(L, "split_dataset", lambda d, fold, size: (lambda tr, te: types.SimpleNamespace(
        x_train=x.iloc[tr], y_train=y.iloc[tr], x_test=x.iloc[te], y_test=y.iloc[te]))(*get_split(y.to_numpy(), True, fold, size)))
    sys.argv = ["x", "run", "--datasets", "BIN_TEXT_FAKE_JOB_POSTING", "--folds", "0", "--out_dir", str(tmp_path)]
    import star_prune_pilot
    star_prune_pilot.main()
    files = sorted(glob.glob(str(tmp_path / "*.json")))
    assert len(files) == 4
    for f in files:
        d = json.load(open(f))
        assert d["course_metric"] == "auroc" and 0 <= d["course_score"] <= 1 and d["model"] == "star"
