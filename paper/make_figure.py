"""Figure 1: per-dataset gain of TAR + text stacking over Joint TAR, for each learner (final run).

    python paper/make_figure.py --final results/runs_tarstack --out paper/figures/stack_gain.pdf
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from make_tables import SHORT  # noqa: E402

# reference categorical palette, fixed order (validated: CVD dE >= 8 adjacent; aqua relies on marker shape + legend)
LEARNERS = [("lgbt", "LightGBM tuned", "#2a78d6", "o", True),
            ("cat", "CatBoost", "#eb6834", "s", True),
            ("light", "LightGBM default", "#1baf7a", "^", False)]
CLIP = -0.014


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--final", default="results/runs_tarstack")
    a.add_argument("--out", default="paper/figures/stack_gain.pdf")
    args = a.parse_args()
    df = pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(args.final, "*.json"))])
    gains = {}
    for m, *_ in LEARNERS:
        t = df[df.model == m].pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
        gains[m] = t["e5-small-tar-fstack"] - t["e5-small-tar"]
    g = pd.DataFrame(gains)
    g = g.loc[g[["lgbt", "cat"]].mean(axis=1).sort_values().index]      # rows ordered by the robust learners' gain

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.5})
    fig, ax = plt.subplots(figsize=(5.5, 2.45))
    y = np.arange(len(g))
    ax.axvline(0, color="#8a8a86", lw=0.8, zorder=1)
    for k in y:
        ax.axhline(k, color="#e6e5e1", lw=0.4, zorder=0)
    offsets = {"lgbt": 0.18, "cat": 0.0, "light": -0.18}
    for m, label, color, marker, filled in LEARNERS:
        v = g[m].to_numpy()
        clipped = v < CLIP
        ax.scatter(np.where(clipped, CLIP, v), y + offsets[m], s=16, marker=marker, zorder=3, linewidths=0.9,
                   facecolors=color if filled else "white", edgecolors=color if not filled else "white", label=label)
        for k in np.where(clipped)[0]:
            ax.annotate(f"{v[k]:+.3f} (collapse)", (CLIP, k + offsets[m]), xytext=(5, -2.5),
                        textcoords="offset points", fontsize=6, color="#555551")
    ax.set_yticks(y, [SHORT.get(d, d) for d in g.index])
    ax.set_xlim(CLIP - 0.001, float(np.nanmax(g.to_numpy())) + 0.002)
    ax.set_xlabel("Gain of TAR + text stacking over Joint TAR (AUROC / $R^2$), mean over 6 folds")
    ax.tick_params(length=2, width=0.5, colors="#3d3d3a")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.6, len(g) - 0.4)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
