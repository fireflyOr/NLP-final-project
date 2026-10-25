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
VS_TAR = [("e5-small-tar-fstack", "TAR + text stacking"), ("e5-small-tar-pls", "TAR + PLS-30 (instead of PCA)"),
          ("e5-small-tar-bn", "TAR, learned 30-d bottleneck"),
          ("e5-small-tar-full", "TAR, all 384 dims"), ("late-fusion", "Late fusion (frozen text model)"),
          ("e5-small-tar-gsae-randvar", "TAR + random pruning (variance-matched)"),
          ("e5-small-tar-gsae", "TAR + concept pruning (ours)"), ("e5-small-tar-gsae-rand", "TAR + random pruning"),
          ("e5-small-gsae", "Frozen + concept pruning"), ("e5-small-pls", "Frozen + PLS"),
          ("e5-small-stack", "Frozen + stacking"), ("e5-small", "Joint Frozen")]


def load(d, model="light"):
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))] if d and os.path.isdir(d) else []
    df = pd.DataFrame(rows)
    if not len(df) or model is None:
        return df
    return df[df.model == model]


def task(ds):
    return {"BIN": "bin", "MUL": "mul", "REG": "reg"}[ds[:3]]


FINAL_MAIN = [("e5-small-tar", "TAR"), ("e5-small-tar-fstack", "+stack"), ("e5-small-tar-pls", "+PLS"),
              ("e5-small-tar-full", "384-d"), ("e5-small-tar-bn", "BN-30"), ("late-fusion", "Late"), ("e5-small-tar-gsae", "+prune"),
              ("e5-small-tar-gsae-randvar", "+rand(var)")]


def main_table(df, out, arms=MAIN):
    m = df.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    n = df.pivot_table(index="dataset", columns="encoder", values="fold", aggfunc="count")
    cols = [(e, h) for e, h in arms if e in m]
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


def family_pvalues(df_all, base="e5-small-tar"):
    """Wilcoxon p of every arm-learner comparison with the same learner's TAR (Joint Frozen excluded): the family
    the Holm correction runs over (fourteen comparisons in the final run)."""
    ps = {}
    for mdl in sorted(set(df_all.model)):
        means = df_all[df_all.model == mdl].pivot_table(index="dataset", columns="encoder", values="course_score",
                                                        aggfunc="mean")
        if base not in means:
            continue
        for e in means.columns:
            if e in (base, "e5-small"):
                continue
            d = (means[e] - means[base]).dropna()
            if len(d) >= 5 and (d != 0).any():
                ps[(mdl, e)] = wilcoxon(d).pvalue
    keys = list(ps)
    return dict(zip(keys, _holm([ps[k] for k in keys])))


def summary_table(df, out, base="e5-small-tar", holm_family=None):
    wide = df.pivot_table(index=["dataset", "fold"], columns="encoder", values="course_score")
    means = df.pivot_table(index="dataset", columns="encoder", values="course_score", aggfunc="mean")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Method & $\Delta$ vs.\ TAR & datasets won & folds won/lost & $p$ & $p_{\text{Holm}}$ \\", r"\midrule"]
    rows = []
    for e, name in VS_TAR:
        if e not in wide or base not in wide:
            continue
        d_ds = (means[e] - means[base]).dropna()
        d_f = (wide[e] - wide[base]).dropna()
        if d_ds.empty:
            continue
        p = wilcoxon(d_ds).pvalue if len(d_ds) >= 5 and (d_ds != 0).any() else np.nan
        rows.append([name, d_ds.mean(), f"{(d_ds > 0).sum()}/{len(d_ds)}", f"{(d_f > 0).sum()}/{(d_f < 0).sum()}", p])
    if holm_family is None:                       # adjust within this table
        holm = _holm([r[4] for r in rows])
    else:                                         # adjust over the whole family of comparisons
        model = df.model.iloc[0]
        holm = [holm_family.get((model, e), np.nan) for e, _ in VS_TAR
                if e in wide and base in wide and not (means[e] - means[base]).dropna().empty]
    fmt = lambda p: "--" if np.isnan(p) else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")
    for (name, d, won, folds, p), ph in zip(rows, holm):
        lines.append(f"{name} & {d:+.4f} & {won} & {folds} & {fmt(p)} & {fmt(ph)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "summary.tex"), "w").write("\n".join(lines) + "\n")


def _holm(ps):
    """Holm-Bonferroni adjusted p-values (NaNs ignored)."""
    ps = np.asarray(ps, dtype=float)
    out = np.full_like(ps, np.nan)
    idx = [i for i in np.argsort(ps) if np.isfinite(ps[i])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


def learners_table(df, out):
    """Does the picture hold for other learners? Mean difference vs the same learner's Joint TAR, datasets won, p."""
    arms = [("e5-small", "Joint Frozen"), ("e5-small-tar-fstack", "TAR + text stacking"),
            ("e5-small-tar-pls", "TAR + PLS-30"), ("e5-small-tar-bn", "TAR, learned bottleneck")]
    learners = [m for m in ("light", "lgbt", "cat") if m in set(df.model)]
    if len(learners) < 2:
        return
    lines = [r"\begin{tabular}{l" + "c" * len(learners) + "}", r"\toprule",
             "Arm ($\\Delta$ vs.\\ TAR, datasets won, $p$) & " + " & ".join(
                 {"light": "LightGBM", "lgbt": "LightGBM tuned", "cat": "CatBoost"}[m] for m in learners) + r" \\",
             r"\midrule"]
    for e, name in arms:
        cells = []
        for mdl in learners:
            means = df[df.model == mdl].pivot_table(index="dataset", columns="encoder", values="course_score",
                                                    aggfunc="mean")
            if e not in means or "e5-small-tar" not in means:
                cells.append("--")
                continue
            d = (means[e] - means["e5-small-tar"]).dropna()
            p = wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
            cells.append(f"{d.mean():+.4f}, {(d > 0).sum()}/{len(d)}, " +
                         ("--" if np.isnan(p) else ("$<$0.001" if p < 1e-3 else f"{p:.3f}")))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "learners.tex"), "w").write("\n".join(lines) + "\n")


def removal_table(df, out):
    """Concepts and variance removed per arm (TAR embeddings); for the selection, the share of removed groups that
    were never split on by either the concept or its shadow."""
    rows = []
    for _, r in df.iterrows():
        for d in (r.get("diagnostics") or {}).values():
            if isinstance(d, dict) and d.get("n_eligible") and "n_removed" in d:
                rows.append({"enc": r.encoder, "frac": d["n_removed"] / d["n_eligible"],
                             "var": d.get("var_share_removed", np.nan),
                             "unused": (d["n_groups_unused"] / d["n_groups_dropped"]
                                        if d.get("n_groups_dropped") and "n_groups_unused" in d else np.nan)})
    if not rows:
        return
    t = pd.DataFrame(rows).groupby("enc").mean()
    names = [("e5-small-tar-gsae", "Concept pruning (ours)"), ("e5-small-tar-gsae-rand", "Random, same number"),
             ("e5-small-tar-gsae-randvar", "Random, same variance"), ("e5-small-gsae", "Ours on frozen embeddings")]
    pct = lambda v: "--" if not np.isfinite(v) else f"{v:.0%}"
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Removal & concepts removed & variance removed & removed groups never used \\", r"\midrule"]
    for e, name in names:
        if e in t.index:
            x = t.loc[e]
            unused = x["unused"] if "rand" not in e else np.nan   # random arms inherit the selection's count
            lines.append(f"{name} & {pct(x['frac'])} & {pct(x['var'])} & {pct(unused)}" + r" \\")
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
    a.add_argument("--final", default="runs_tarstack", help="the final paired run (all TAR arms, all learners)")
    a.add_argument("--out", default="paper/tables")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)
    full = load(args.full)
    if len(full):
        main_table(full, args.out)
        summary_table(full, args.out)
        removal_table(full, args.out)
    xf_table(load(args.xf), args.out)
    final_all = load(args.final, model=None)
    if len(final_all):
        final = final_all[final_all.model == "light"]
        os.makedirs(os.path.join(args.out, "final"), exist_ok=True)
        summary_table(final, os.path.join(args.out, "final"), holm_family=family_pvalues(final_all))
        main_table(final, os.path.join(args.out, "final"), arms=FINAL_MAIN)
        removal_table(final, os.path.join(args.out, "final"))
        learners_table(final_all, os.path.join(args.out, "final"))
    print("tables written to", args.out)
