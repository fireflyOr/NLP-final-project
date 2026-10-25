#!/bin/bash
#SBATCH --job-name=mtb_bn
#SBATCH --array=0-19%4
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=bn_%A_%a.out
# Learned compression (TAR recipe with a 30-unit bottleneck instead of PCA-30), all three learners.
# The bottleneck fine-tune is saved to tar_ckpt/<i>/ by the first learner and restored by the others.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MULTABENCH_TAR_DIR=tar_ckpt/${SLURM_ARRAY_TASK_ID} MULTABENCH_DYNAMIC_PADDING=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1

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

for learner in light lgbt cat; do
  PYTHONPATH=. python sae_pilot.py run --model $learner --out_dir runs_tarstack --emb_cache emb_cache/$ds --datasets $ds \
    --encoders e5-small-tar-bn
done
