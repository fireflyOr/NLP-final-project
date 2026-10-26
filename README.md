# Add, Don't Remove: Where Target Supervision Helps in Text–Tabular Learning

Final project, NLP course 097215 (Technion), Track 1 (algorithm). Or Davidovich.
Paper: [`paper/main.pdf`](paper/main.pdf) (LaTeX source in `paper/`).

This repository is the [MulTaBench](https://github.com/alanarazi7/MulTaBench) code at commit `cf189b7` plus the
project's additions. MulTaBench's own README is kept as [`README_MULTABENCH.md`](README_MULTABENCH.md).

## What the project does

MulTaBench's Joint TAR baseline fine-tunes E5-small on the target, compresses each text column to 30 PCA components
and gives them to a tree model with the table. We test three ways to use the target further on top of TAR:

- **Remove**: SAE concept pruning. A TopK SAE splits each column's embeddings into concepts; a concept is removed
  if a LightGBM over the table, all concepts and shuffled "shadow" copies does not use it more than its shadow.
  Controls remove random concepts (same number, or same variance); a cross-fitted variant chooses how much to remove
  on rows the encoder never saw.
- **Reweight / re-learn the compression**: PLS-30 instead of PCA-30, all 384 dimensions, and a 30-unit bottleneck
  trained into TAR's fine-tune.
- **Add**: text stacking. Out-of-fold predictions of a text-only MLP on frozen E5 embeddings are added as extra
  columns next to TAR's.

Result: only adding helps. TAR + text stacking beats Joint TAR with tuned LightGBM (+0.0035, 16/20 datasets,
Holm-adjusted p = 0.007) and CatBoost (+0.0036, 17/20, p = 0.004); removal and reweighting do not.

## Run the method on your own data

```bash
python run_csv.py --train train.csv --test test.csv --target price --out predictions.csv
python run_csv.py --train data.csv --target label --test-size 0.2 --compare     # also prints Joint TAR's score
```

Column types are detected automatically and everything is fitted on the training rows only; no trained weights
are needed (the method starts from the public `intfloat/e5-small-v2`). A GPU is strongly recommended. Options and a
check that reproduces a paper run exactly are in [`REPRODUCE.md`](REPRODUCE.md).

## Reproducing the paper

Setup, the SLURM scripts behind every result, and the commands that rebuild the tables and figure are in
[`REPRODUCE.md`](REPRODUCE.md). All result files are in `results/` (one JSON per dataset × fold × arm × learner).

## Code map (project additions)

| File | What it does |
|---|---|
| `multabench/sae/topk_sae.py` | TopK sparse autoencoder (Gao et al., 2024) |
| `multabench/sae/relevance.py` | concept pruning (`GroupShadowSAETransform`), random controls, cross-fitted ranking (`RankedSAEPruner`), PLS / full-dimension transforms; earlier relevance-filter variants |
| `multabench/stack/text_stack.py` | text stacking: out-of-fold text-model predictions as extra columns |
| `multabench/stack/late_fusion.py` | late-fusion control (table model + text model, linear meta-learner) |
| `multabench/baselines/lgbm_tuned.py` | LightGBM tuned on the validation split (12 configurations) |
| `multabench/baselines/preprocessing/text_embeddings.py` | wires every arm into MulTaBench's text pipeline; saves/restores each fold's TAR fine-tune so all arms share it |
| `multabench/e5/e5_finetune.py` | TAR fine-tuning additions: bottleneck, saving/loading LoRA adapters, dynamic padding |
| `multabench/e5/constants.py` | names of all arms (`--encoders`) |
| `sae_pilot.py` | experiment driver: runs arms × datasets × folds, writes result JSONs, summarizes |
| `run_csv.py`, `export_fold.py` | the method on any CSV; export a benchmark fold as CSV |
| `paper/make_tables.py`, `paper/make_figure_overview.py` | tables and figure of the paper from `results/` |
| `*.sbatch` | the SLURM jobs that produced the results |
| `tests/test_sae.py`, `tests/test_run_csv.py` | tests, including leakage checks |

Early pilots, not used in the paper: `train_sae.py` (an SAE trained on a generic corpus, for the frozen
`e5-small-sae` arms), `star_prune_pilot.py` with `multabench/star/` (pruning inside TabSTAR; infeasible on our
hardware), and the frozen-SAE / relevance-filter arms in `multabench/sae/feature_transform.py` and `relevance.py`.

## Known benchmark behavior

MulTaBench reads date columns from the Kaggle CSVs as strings, so some dates are treated as text columns (an open
upstream issue). This applies identically to every arm and baseline, so the paired comparisons are unaffected.
