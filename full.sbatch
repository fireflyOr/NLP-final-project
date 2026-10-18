#!/bin/bash
#SBATCH --job-name=mtb_full
#SBATCH --array=0-19%5
#SBATCH --partition=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=full_%A_%a.out
# All 20 MulTaBench text datasets x 6 folds, LightGBM. One array task per dataset; finished runs are skipped,
# so resubmitting resumes. Order matters: e5-small-tar first, then its pruned arms reuse the same fine-tuned E5.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MULTABENCH_DYNAMIC_PADDING=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1

# the virtualenv must be built on a compute node (the login node's python is different); one task builds it
(
  flock 9
  if [ ! -f .venv/.ok ]; then
    rm -rf .venv && python3 -m venv .venv && source .venv/bin/activate
    pip install -q --upgrade pip
    pip install -q torch lightgbm scikit-learn pandas scipy transformers peft skrub kagglehub python-dotenv "autogluon.core" pytest
    pip uninstall -y -q torchao 2>/dev/null || true
    touch .venv/.ok
  fi
) 9>.venv.lock
source .venv/bin/activate

DS=(BIN_TEXT_FAKE_JOB_POSTING BIN_TEXT_JIGSAW_TOXICITY BIN_TEXT_KICKSTARTER_FUNDING MUL_TEXT_DATA_SCIENTIST_SALARY
    MUL_TEXT_MICHELIN_RESTAURANTS MUL_TEXT_PRODUCT_SENTIMENT MUL_TEXT_SPOTIFY_GENRES MUL_TEXT_US_ACCIDENTS
    MUL_TEXT_WINE_REVIEW MUL_TEXT_WOMEN_CLOTHING_REVIEW REG_TEXT_BABIES_PRICES REG_TEXT_BOOK_PRICE
    REG_TEXT_BOOK_READABILITY REG_TEXT_MERCARI_MARKETPLACE REG_TEXT_MONTGOMERY_SALARIES REG_TEXT_SCIMAGOJR_IMPACT
    REG_TEXT_ROTTEN_TOMATOES REG_TEXT_VANCOUVER_SALARIES REG_TEXT_VIDEO_GAMES_SALES REG_TEXT_ZOMATO_RESTAURANTS)
ds=${DS[$SLURM_ARRAY_TASK_ID]}
echo "== $ds on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PYTHONPATH=. python sae_pilot.py run --out_dir runs_full --emb_cache emb_cache/$ds --datasets $ds \
  --encoders e5-small e5-small-tar e5-small-tar-gsae e5-small-tar-gsae-rand \
             e5-small-gsae e5-small-stack e5-small-pls e5-small-relsae2 e5-small-relsae2-rand
