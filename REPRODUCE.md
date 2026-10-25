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

## Results and tables

All result files used in the paper are in `results/` (one JSON per dataset x fold x arm x learner).

```bash
uv run python paper/make_tables.py --full results/runs_full --xf results/runs_xf \
    --final results/runs_tarstack --out paper/tables
uv run python sae_pilot.py summarize --out_dir results/runs_tarstack --baselines e5-small-tar
```
