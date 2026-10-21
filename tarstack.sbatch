#!/bin/bash
#SBATCH --job-name=mtb_tstack
#SBATCH --array=0-19%4
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=tstack_%A_%a.out
# Final pre-declared test: Joint TAR vs Joint TAR + out-of-fold text stacking (same TAR fine-tune per fold).
# All 20 MulTaBench text datasets x 6 folds, LightGBM. One array task per dataset; finished runs are skipped,
# so resubmitting resumes. Order matters: e5-small-tar first, then its pruned arms reuse the same fine-tuned E5.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MULTABENCH_DYNAMIC_PADDING=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1

# the environment is built ONCE beforehand (see setup in the instructions); array tasks only use it
[ -f .venv_cu126/.ok ] || { echo "missing .venv_cu126 - run the setup srun first"; exit 1; }
source .venv_cu126/bin/activate
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU on $(hostname)"; exit 1; }

DS=(BIN_TEXT_FAKE_JOB_POSTING BIN_TEXT_JIGSAW_TOXICITY BIN_TEXT_KICKSTARTER_FUNDING MUL_TEXT_DATA_SCIENTIST_SALARY
    MUL_TEXT_MICHELIN_RESTAURANTS MUL_TEXT_PRODUCT_SENTIMENT MUL_TEXT_SPOTIFY_GENRES MUL_TEXT_US_ACCIDENTS
    MUL_TEXT_WINE_REVIEW MUL_TEXT_WOMEN_CLOTHING_REVIEW REG_TEXT_BABIES_PRICES REG_TEXT_BOOK_PRICE
    REG_TEXT_BOOK_READABILITY REG_TEXT_MERCARI_MARKETPLACE REG_TEXT_MONTGOMERY_SALARIES REG_TEXT_SCIMAGOJR_IMPACT
    REG_TEXT_ROTTEN_TOMATOES REG_TEXT_VANCOUVER_SALARIES REG_TEXT_VIDEO_GAMES_SALES REG_TEXT_ZOMATO_RESTAURANTS)
ds=${DS[$SLURM_ARRAY_TASK_ID]}
echo "== $ds on $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

PYTHONPATH=. python sae_pilot.py run --out_dir runs_tarstack --emb_cache emb_cache/$ds --datasets $ds \
  --encoders e5-small-tar e5-small-tar-fstack e5-small-tar-gsae e5-small-tar-gsae-randvar e5-small-tar-gsae-rand
