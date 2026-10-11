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
