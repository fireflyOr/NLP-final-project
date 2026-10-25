# Reproducing the results

Python 3.10, Linux, an NVIDIA GPU (CUDA 12.x driver). Dependencies are pinned in `pyproject.toml` / `uv.lock`
to the exact versions the reported experiments ran with.

```bash
uv sync                      # creates .venv with the locked versions (torch 2.8.0 + CUDA 12.6)
uv run pytest tests/test_sae.py -q        # 73 tests, CPU only (~1 h)
```

Datasets are downloaded from Kaggle by `kagglehub` on first use (see `.env.example`).

## Runs (SLURM scripts used for the paper)

| Script | What it runs | Output |
|---|---|---|
| `full.sbatch` | main run: frozen-embedding variants and concept pruning, 20 datasets x 6 folds, LightGBM | `runs_full/` |
| `xf.sbatch` | nested cross-fitted removal amount, 5 datasets | `runs_xf/` |
| `tarstack.sbatch` | final run: every TAR-based arm, one saved TAR fine-tune per fold (`tar_ckpt/`) | `runs_tarstack/` |
| `learners.sbatch` | same arms with tuned LightGBM, CatBoost, and late fusion (restores the saved TARs) | `runs_tarstack/` |

A single dataset/fold can be run directly, e.g.

```bash
uv run python sae_pilot.py run --datasets BIN_TEXT_FAKE_JOB_POSTING --folds 0 \
    --encoders e5-small-tar e5-small-tar-fstack --out_dir runs_demo
```

Arms (`--encoders`): `e5-small` (Joint Frozen), `e5-small-tar` (Joint TAR), `e5-small-tar-fstack` (TAR + text
stacking), `e5-small-tar-pls`, `e5-small-tar-full`, `e5-small-tar-gsae` (concept pruning), `e5-small-tar-gsae-randvar`
/ `-rand` (random controls), `e5-small-tar-gsae-xf` (cross-fitted). Learners (`--model`): `light`, `lgbt` (tuned),
`cat`, `late` (late fusion).

## Run the method on your own CSV

`run_csv.py` runs the method (Joint TAR + text stacking, then tuned LightGBM) on any table with free-text columns.
Column types are detected automatically; everything is fitted on the training rows only. A GPU is strongly recommended.

```bash
python run_csv.py --train train.csv --test test.csv --target price --out predictions.csv
python run_csv.py --train data.csv --target label --test-size 0.2 --out predictions.csv   # hold out 20%
```

Options: `--task` (auto / binary / multiclass / regression; needed when an integer target could be classes or a
count), `--learner` (lgbt / cat / light), `--encoder` (`e5-small-tar` and `e5-small` run the Joint TAR and Joint
Frozen baselines for comparison), `--drop-cols` (e.g. row ids). The output has `prediction` (regression) or
`predicted_label` and one `proba_<class>` column per class; if the test rows contain the target, AUROC or R^2 is
printed. No trained weights are needed: the method starts from the public `intfloat/e5-small-v2` and trains everything
on the given training rows.

`--compare` also runs the Joint TAR baseline on the same fine-tuned E5 and prints both scores.

To check it against the experiments, export a benchmark fold and rerun it with the experiment's saved fine-tune and
frozen-embedding cache (copies, so the originals are untouched; array index 16 = Rotten Tomatoes):

```bash
python export_fold.py --dataset REG_TEXT_ROTTEN_TOMATOES --fold 0 --out-dir csv_check
mkdir -p csv_check/tar && cp tar_ckpt/16/tar_*.pt csv_check/tar/ && cp -r emb_cache/REG_TEXT_ROTTEN_TOMATOES csv_check/emb
MULTABENCH_TAR_DIR=csv_check/tar python run_csv.py --train csv_check/train.csv --test csv_check/test.csv \
    --target target --compare --emb-cache csv_check/emb --out csv_check/pred.csv
# reproduces results/runs_tarstack/lgbt_REG_TEXT_ROTTEN_TOMATOES_{e5-small-tar,e5-small-tar-fstack}_0.json exactly:
# r2 0.5130 (Joint TAR) and 0.5233 (TAR + stacking), with the same tuned LightGBM settings
```

Without the cache, the frozen embeddings are recomputed and differ in the last digits; on this fold that changes the
tuned LightGBM's choice and the stacking score to 0.5112. Single folds are this sensitive, so compare methods over many
datasets and folds, not on one.

## Results and tables

All result files used in the paper are in `results/` (one JSON per dataset x fold x arm x learner).

```bash
uv run python paper/make_tables.py --full results/runs_full --xf results/runs_xf \
    --final results/runs_tarstack --out paper/tables
uv run python sae_pilot.py summarize --out_dir results/runs_tarstack --baselines e5-small-tar
```
