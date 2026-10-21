"""Build the paper's LaTeX tables from the result JSONs.

    python paper/make_tables.py --full runs_full --xf runs_xf --out paper/tables

Writes main.tex (per-dataset scores), summary.tex (paired tests vs Joint TAR), removal.tex (how much was removed)
and xf.tex (cross-fitted experiment). Dataset-level tests use per-dataset means over folds (folds are not independent).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SHORT = {
    "BIN_TEXT_FAKE_JOB_POSTING": "Fake job postings", "BIN_TEXT_JIGSAW_TOXICITY": "Jigsaw toxicity",
    "BIN_TEXT_KICKSTARTER_FUNDING": "Kickstarter", "MUL_TEXT_DATA_SCIENTIST_SALARY": "DS salary",
    "MUL_TEXT_MICHELIN_RESTAURANTS": "Michelin", "MUL_TEXT_PRODUCT_SENTIMENT": "Product sentiment",
    "MUL_TEXT_SPOTIFY_GENRES": "Spotify genres", "MUL_TEXT_US_ACCIDENTS": "US accidents",
    "MUL_TEXT_WINE_REVIEW": "Wine reviews", "MUL_TEXT_WOMEN_CLOTHING_REVIEW": "Women's clothing",
    "REG_TEXT_BABIES_PRICES": "Baby product prices", "REG_TEXT_BOOK_PRICE": "Book prices",
    "REG_TEXT_BOOK_READABILITY": "Book readability", "REG_TEXT_MERCARI_MARKETPLACE": "Mercari",
    "REG_TEXT_MONTGOMERY_SALARIES": "Montgomery salaries", "REG_TEXT_SCIMAGOJR_IMPACT": "SCImago impact",
    "REG_TEXT_ROTTEN_TOMATOES": "Rotten Tomatoes", "REG_TEXT_VANCOUVER_SALARIES": "Vancouver salaries",
    "REG_TEXT_VIDEO_GAMES_SALES": "Video game sales", "REG_TEXT_ZOMATO_RESTAURANTS": "Zomato",
}
MAIN = [("e5-small", "Frozen"), ("e5-small-tar", "TAR"), ("e5-small-tar-gsae", "TAR+prune"),
        ("e5-small-tar-gsae-rand", "TAR+rand"), ("e5-small-pls", "PLS"), ("e5-small-stack", "Stack")]
VS_TAR = [("e5-small-tar-gsae", "TAR + concept pruning (ours)"), ("e5-small-tar-gsae-rand", "TAR + random pruning"),
          ("e5-small-gsae", "Frozen + concept pruning"), ("e5-small-pls", "Frozen + PLS"),
          ("e5-small-stack", "Frozen + stacking"), ("e5-small", "Joint Frozen")]


def load(d):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))] if d and os.path.isdir(d) else []
    df = pd.DataFrame(rows)
    return df[df.model == "light"] if len(df) else df


def task(ds):
    return {"BIN": "bin", "MUL": "mul", "REG": "reg"}[ds[:3]]


def main_table(df, out):
    m = df.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    n = df.pivot_table(index="dataset", columns="encoder", values="fold", aggfunc="count")
    cols = [(e, h) for e, h in MAIN if e in m]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Dataset & " + " & ".join(h for _, h in cols) + r" \\", r"\midrule"]
    for ds in sorted(m.index, key=lambda d: (d[:3] != "BIN", d[:3] != "MUL", SHORT.get(d, d))):
        vals = [m.loc[ds, e] if e in m else np.nan for e, _ in cols]
        best = np.nanmax(vals) if np.isfinite(vals).any() else None
        cells = []
        for (e, _), v in zip(cols, vals):
            if not np.isfinite(v):
                cells.append("--")
                continue
            s = f"{v:.3f}" + ("$^\\dagger$" if n.loc[ds, e] < 6 else "")
            cells.append(f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 5e-4 else s)
        lines.append(f"{SHORT.get(ds, ds)} ({task(ds)}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "main.tex"), "w").write("\n".join(lines) + "\n")


def summary_table(df, out, base="e5-small-tar"):
    wide = df.pivot_table(index=["dataset", "fold"], columns="encoder", values="course_score")
    means = df.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & $\Delta$ vs.\ TAR & datasets won & folds won/lost & $p$ (datasets) \\", r"\midrule"]
    for e, name in VS_TAR:
        if e not in wide or base not in wide:
            continue
        d_ds = (means[e] - means[base]).dropna()
        d_f = (wide[e] - wide[base]).dropna()
        if d_ds.empty:
            continue
        p = wilcoxon(d_ds).pvalue if len(d_ds) >= 5 and (d_ds != 0).any() else np.nan
        lines.append(f"{name} & {d_ds.mean():+.4f} & {(d_ds > 0).sum()}/{len(d_ds)} & "
                     f"{(d_f > 0).sum()}/{(d_f < 0).sum()} & {'--' if np.isnan(p) else ('$<$0.001' if p < 1e-3 else f'{p:.3f}')}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "summary.tex"), "w").write("\n".join(lines) + "\n")


def removal_table(df, out):
    rows = []
    for _, r in df[df.encoder.isin(["e5-small-tar-gsae", "e5-small-gsae"])].iterrows():
        for col, d in (r.get("diagnostics") or {}).items():
            if isinstance(d, dict) and "n_removed" in d and d.get("n_eligible"):
                rows.append({"enc": r.encoder, "frac": d["n_removed"] / d["n_eligible"],
                             "var": d.get("var_share_removed", np.nan), "groups": d.get("n_groups", np.nan),
                             "elig": d["n_eligible"]})
    if not rows:
        return
    t = pd.DataFrame(rows).groupby("enc").agg(frac=("frac", "mean"), var=("var", "mean"), elig=("elig", "mean"),
                                              groups=("groups", "mean"), n=("frac", "size"))
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Embeddings & concepts removed & variance removed & eligible concepts & sibling groups \\", r"\midrule"]
    for e, name in (("e5-small", "Frozen"), ("e5-small-tar", "TAR")):
        key = e + "-gsae" if e == "e5-small" else "e5-small-tar-gsae"
        if key in t.index:
            x = t.loc[key]
            lines.append(f"{name} & {x['frac']:.0%} & {x['var']:.0%} & {x['elig']:.0f} & {x['groups']:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "removal.tex"), "w").write("\n".join(lines).replace("%", r"\%") + "\n")


def xf_table(df, out):
    if df.empty or "e5-small-tar-gsae-xf" not in set(df.encoder):
        return
    m = df.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    chosen = {}
    for _, r in df[df.encoder == "e5-small-tar-gsae-xf"].iterrows():
        for d in (r.get("diagnostics") or {}).values():
            if isinstance(d, dict) and "chosen_fraction" in d:
                chosen.setdefault(r.dataset, []).append(d["chosen_fraction"])
                break
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Dataset & TAR & TAR + cross-fitted pruning & chosen removal (per fold) \\", r"\midrule"]
    for ds in m.index:
        fr = ", ".join(f"{int(round(100 * f))}" for f in chosen.get(ds, []))
        lines.append(f"{SHORT.get(ds, ds)} & {m.loc[ds, 'e5-small-tar']:.3f} & "
                     f"{m.loc[ds, 'e5-small-tar-gsae-xf']:.3f} & {fr}" + r"\,\% \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "xf.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--full", default="runs_full")
    a.add_argument("--xf", default="runs_xf")
    a.add_argument("--out", default="paper/tables")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    full = load(args.full)
    if len(full):
        main_table(full, args.out)
        summary_table(full, args.out)
        removal_table(full, args.out)
    xf_table(load(args.xf), args.out)
    print("tables written to", args.out)
