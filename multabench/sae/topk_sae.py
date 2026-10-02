"""TopK sparse autoencoder over sentence embeddings (Gao et al., 2024, "Scaling and evaluating sparse autoencoders").

encode:  z = TopK(ReLU(W_enc (x - b_dec) + b_enc))        exactly k active latents per input
decode:  x_hat = W_dec z + b_dec                            decoder columns kept at unit norm
loss:    ||x - x_hat||^2  +  aux_coef * AuxK loss            AuxK revives dead latents
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_in: int = 384
    n_latents: int = 4096
    k: int = 16
    k_aux: int = 256          # dead latents used by the AuxK loss
    aux_coef: float = 1 / 32
    dead_after: int = 2_000_000  # a latent is "dead" if it hasn't fired for this many samples


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        w = torch.randn(cfg.d_in, cfg.n_latents)
        w = w / w.norm(dim=0, keepdim=True)
        self.W_dec = nn.Parameter(w.T.clone())                 # (n_latents, d_in)
        self.W_enc = nn.Parameter(w.clone())                   # (d_in, n_latents), tied init
        self.b_enc = nn.Parameter(torch.zeros(cfg.n_latents))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("samples_since_fired", torch.zeros(cfg.n_latents, dtype=torch.long))

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, vals)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        z = torch.zeros_like(pre).scatter_(-1, idx, vals)
        x_hat = self.decode(z)
        mse = (x_hat - x).pow(2).sum(-1).mean()

        if self.training:
            fired = torch.zeros(self.cfg.n_latents, dtype=torch.bool, device=x.device)
            fired[idx[vals > 0].flatten()] = True
            self.samples_since_fired += x.shape[0]
            self.samples_since_fired[fired] = 0

        aux = torch.zeros((), device=x.device)
        dead = self.samples_since_fired > self.cfg.dead_after
        n_dead = int(dead.sum())
        if self.training and n_dead > 0:
            # AuxK: reconstruct the residual using only the top dead latents.
            k_aux = min(self.cfg.k_aux, n_dead)
            dead_pre = pre.masked_fill(~dead, 0.0)
            a_vals, a_idx = dead_pre.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(pre).scatter_(-1, a_idx, a_vals)
            residual = (x - x_hat).detach()
            aux = (z_aux @ self.W_dec - residual).pow(2).sum(-1).mean()
            aux = aux / residual.pow(2).sum(-1).mean().clamp_min(1e-8)

        return {"z": z, "x_hat": x_hat, "mse": mse, "aux": aux,
                "loss": mse + self.cfg.aux_coef * aux, "n_dead": torch.tensor(n_dead)}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def remove_parallel_grad(self) -> None:
        """Project out the gradient component that would change decoder norms (keeps them at 1)."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        self.W_dec.grad -= (self.W_dec.grad * w).sum(dim=1, keepdim=True) * w

    # ---------- persistence ----------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"cfg": asdict(self.cfg), "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "TopKSAE":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        sae = cls(SAEConfig(**ckpt["cfg"]))
        sae.load_state_dict(ckpt["state_dict"])
        sae.extra = ckpt.get("extra", {})
        return sae.eval()


def train_sae(embeddings: np.ndarray, cfg: SAEConfig, device: str = "cpu", epochs: int = 10,
              batch_size: int = 4096, lr: float = 2e-4, seed: int = 0, log_every: int = 200) -> TopKSAE:
    """Train a TopK SAE on (N, d_in) embeddings. b_dec starts at the geometric-median proxy (mean)."""
    torch.manual_seed(seed)
    x_all = torch.as_tensor(embeddings, dtype=torch.float32)
    sae = TopKSAE(cfg).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(x_all.mean(0).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), eps=6.25e-10)
    n = len(x_all)
    step = 0
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            x = x_all[perm[i:i + batch_size]].to(device)
            out = sae(x)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            sae.remove_parallel_grad()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae.normalize_decoder()
            if step % log_every == 0:
                print(f"epoch {epoch} step {step}: mse {out['mse'].item():.5f} "
                      f"aux {out['aux'].item():.4f} dead {int(out['n_dead'])}/{cfg.n_latents}")
            step += 1
    return sae.eval()


def random_sae(embeddings: np.ndarray, cfg: SAEConfig, seed: int = 0) -> TopKSAE:
    """Untrained control: random unit-norm directions, same TopK sparsity, b_dec = data mean."""
    torch.manual_seed(seed)
    sae = TopKSAE(cfg)
    with torch.no_grad():
        sae.b_dec.copy_(torch.as_tensor(embeddings, dtype=torch.float32).mean(0))
    return sae.eval()


@torch.no_grad()
def sae_stats(sae: TopKSAE, embeddings: np.ndarray, batch_size: int = 8192) -> Dict[str, float]:
    """Fraction of variance unexplained and fraction of latents that never fire on these embeddings."""
    x_all = torch.as_tensor(embeddings, dtype=torch.float32)
    dev = sae.b_dec.device
    sse, counts = 0.0, torch.zeros(sae.cfg.n_latents, device=dev)
    for i in range(0, len(x_all), batch_size):
        x = x_all[i:i + batch_size].to(dev)
        z = sae.encode(x)
        sse += (sae.decode(z) - x).pow(2).sum().item()
        counts += (z > 0).sum(0)
    total = (x_all - x_all.mean(0)).pow(2).sum().item()
    return {"fvu": sse / total, "frac_never_fired": float((counts == 0).float().mean())}


@torch.no_grad()
def encode_numpy(sae: TopKSAE, embeddings: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    dev = sae.b_dec.device
    x_all = torch.as_tensor(embeddings, dtype=torch.float32)
    out = [sae.encode(x_all[i:i + batch_size].to(dev)).cpu().numpy() for i in range(0, len(x_all), batch_size)]
    return np.concatenate(out, axis=0).astype(np.float32)
