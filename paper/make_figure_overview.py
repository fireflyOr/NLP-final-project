"""Overview figure: mean gain over Joint TAR of every arm built on TAR, per learner, with 95% bootstrap CIs.

Each point is the mean over the 20 datasets of (arm - TAR), where each dataset's value is its mean over 6 folds.
CI: percentile bootstrap over datasets (10,000 resamples). Arms are grouped by what they do to the TAR text features.

    python paper/make_figure_overview.py --final results/runs_tarstack --out paper/figures/overview.pdf
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LEARNERS = [("lgbt", "LightGBM tuned", "#2a78d6", "o", True),
            ("cat", "CatBoost", "#eb6834", "s", True),
            ("light", "LightGBM default", "#1baf7a", "^", False)]
# (group, encoder, label), top to bottom
ARMS = [("Remove", "e5-small-tar-gsae", "SAE concept pruning"),
        ("Remove", "e5-small-tar-gsae-randvar", "Random pruning (same variance)"),
        ("Compress", "e5-small-tar-pls", "PLS, 30 components"),
        ("Compress", "e5-small-tar-bn", "Learned 30-d bottleneck"),
        ("Keep", "e5-small-tar-full", "All 384 dimensions"),
        ("Add", "e5-small-tar-fstack", "Text stacking")]
BASE = "e5-small-tar"
N_BOOT = 10_000


def gains(df, model, enc):
    t = df[df.model == model].pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    if enc not in t:
        return None
    return (t[enc] - t[BASE]).dropna().to_numpy()


def boot_ci(v, rng):
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--final", default="results/runs_tarstack")
    a.add_argument("--out", default="paper/figures/overview.pdf")
    args = a.parse_args()
    df = pd.DataFrame([json.load(open(p)) for p in glob.glob(os.path.join(args.final, "*.json"))])
    rng = np.random.default_rng(0)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.linewidth": 0.5})
    fig, ax = plt.subplots(figsize=(5.5, 2.1))
    y = np.arange(len(ARMS))[::-1]
    offsets = {"lgbt": 0.2, "cat": 0.0, "light": -0.2}
    ax.axvline(0, color="#8a8a86", lw=0.8, zorder=1)
    for i in range(1, len(ARMS)):                                  # separators between groups
        if ARMS[i][0] != ARMS[i - 1][0]:
            ax.axhline(y[i] + 0.5, color="#d0cfca", lw=0.5, zorder=0)
    for m, label, color, marker, filled in LEARNERS:
        first = True
        for k, (_, enc, _) in enumerate(ARMS):
            v = gains(df, m, enc)
            if v is None:
                continue
            lo, hi = boot_ci(v, rng)
            yy = y[k] + offsets[m]
            ax.plot([lo, hi], [yy, yy], color=color, lw=1.0, zorder=2, solid_capstyle="round")
            ax.scatter([v.mean()], [yy], s=18, marker=marker, zorder=3, linewidths=0.9,
                       facecolors=color if filled else "white", edgecolors="white" if filled else color,
                       label=label if first else None)
            first = False
    ax.set_yticks(y, [f"{lab}" for _, _, lab in ARMS])
    for k, (grp, _, _) in enumerate(ARMS):                         # group names at the far left
        if k == 0 or ARMS[k - 1][0] != grp:
            span = [j for j, r in enumerate(ARMS) if r[0] == grp]
            ax.annotate(grp, xy=(0, (y[span[0]] + y[span[-1]]) / 2), xycoords=("axes fraction", "data"),
                        xytext=(-118, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=7, fontstyle="italic", color="#555551", annotation_clip=False)
    ax.set_xlabel("Mean gain over Joint TAR across 20 datasets (AUROC / $R^2$), 95% bootstrap CI")
    ax.tick_params(length=2, width=0.5, colors="#3d3d3a")
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-0.6, len(ARMS) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.5,
              borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.30)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace(".pdf", ".png"), dpi=200)
    print("figure written to", args.out)


if __name__ == "__main__":
    main()
