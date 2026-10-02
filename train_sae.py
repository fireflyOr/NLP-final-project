"""Train the TopK SAE on E5-small-v2 embeddings of a generic text corpus (no MulTaBench data).

    python train_sae.py --n_texts 300000 --n_latents 4096 --k 16

Writes sae_checkpoints/e5-small-topk.pt (trained) and sae_checkpoints/e5-small-random.pt
(untrained control with the same architecture and sparsity), plus the cached corpus embeddings.

Texts are embedded with the same "passage: <col>: <text>" format and mean-pool + L2 used for
benchmark columns, so the SAE is trained on the distribution it is later applied to.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch

from multabench.e5.constants import E5_SMALL_V2, format_e5_passage
from multabench.e5.e5_finetune import _mean_pool_l2, get_vanilla_e5
from multabench.sae.topk_sae import SAEConfig, random_sae, sae_stats, train_sae

# Generic, public corpora. None of them is a source of a MulTaBench dataset, but some overlap in
# domain (e.g. reviews); that is fine for leakage since no benchmark rows are used.
CORPORA = {
    "wikitext": dict(path="Salesforce/wikitext", name="wikitext-103-raw-v1", split="train", field="text"),
    "ag_news": dict(path="fancyzhx/ag_news", name=None, split="train", field="text"),
    "yelp": dict(path="Yelp/yelp_review_full", name=None, split="train", field="text"),
    "dbpedia": dict(path="fancyzhx/dbpedia_14", name=None, split="train", field="content"),
}
GENERIC_COL_NAMES = ["text", "description", "review", "title", "summary", "notes", "comment", "name"]


def load_corpus(names: List[str], n_texts: int, seed: int) -> List[str]:
    from datasets import load_dataset
    per = n_texts // len(names)
    texts: List[str] = []
    for name in names:
        spec = CORPORA[name]
        ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=50_000)
        got = 0
        for row in ds:
            t = str(row[spec["field"]]).strip()
            if len(t) < 20 or t.startswith(" ="):  # skip wikitext headings / empty lines
                continue
            # vary length: the benchmark has both short fields and long reviews
            words = t.split()
            if len(words) > 8 and random.random() < 0.5:
                words = words[: random.randint(3, min(len(words), 64))]
            texts.append(" ".join(words))
            got += 1
            if got >= per:
                break
        print(f"corpus {name}: {got} texts")
    return texts


@torch.no_grad()
def embed(texts: List[str], device: str, batch_size: int, seed: int) -> np.ndarray:
    """E5 embeddings with dynamic padding (identical to max_length padding under masked mean-pooling)."""
    rng = random.Random(seed)
    prefixed = [format_e5_passage(rng.choice(GENERIC_COL_NAMES), t) for t in texts]
    model, tok = get_vanilla_e5(torch.device(device), model_name=E5_SMALL_V2)
    order = np.argsort([len(t) for t in prefixed])  # length-sorted batches are much faster
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([prefixed[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        out[idx] = _mean_pool_l2(h.float(), enc["attention_mask"]).cpu().numpy()
        if (i // batch_size) % 200 == 0:
            print(f"embedded {i + len(idx)}/{len(texts)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", default=list(CORPORA))
    p.add_argument("--n_texts", type=int, default=300_000)
    p.add_argument("--n_latents", type=int, default=4096)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--embed_batch_size", type=int, default=256)
    p.add_argument("--out_dir", type=str, default="sae_checkpoints")
    p.add_argument("--name", type=str, default="e5-small-topk")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    emb_path = os.path.join(args.out_dir, f"corpus_emb_{args.n_texts}_{args.seed}.npy")
    if os.path.exists(emb_path):
        emb = np.load(emb_path)
        print(f"loaded cached embeddings {emb.shape}")
    else:
        emb = embed(load_corpus(args.corpora, args.n_texts, args.seed), device, args.embed_batch_size, args.seed)
        np.save(emb_path, emb)

    # hold out 5% to measure reconstruction on unseen text
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(emb))
    n_val = max(1000, len(emb) // 20)
    val, tr = emb[perm[:n_val]], emb[perm[n_val:]]

    cfg = SAEConfig(d_in=emb.shape[1], n_latents=args.n_latents, k=args.k,
                    dead_after=max(len(tr) // 2, 10_000))
    sae = train_sae(tr, cfg, device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    stats = sae_stats(sae, val)
    print(f"held-out: {stats}")
    sae.cpu().save(os.path.join(args.out_dir, f"{args.name}.pt"), extra={"args": vars(args), "val": stats})

    rnd = random_sae(tr, cfg, seed=args.seed)
    rstats = sae_stats(rnd, val)
    print(f"random control held-out: {rstats}")
    rnd.save(os.path.join(args.out_dir, "e5-small-random.pt"), extra={"args": vars(args), "val": rstats})

    with open(os.path.join(args.out_dir, f"{args.name}_stats.json"), "w") as f:
        json.dump({"trained": stats, "random": rstats, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
