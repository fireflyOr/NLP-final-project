#!/bin/bash
#SBATCH --job-name=mtb_xf
#SBATCH --array=0-4%1
#SBATCH --partition=public
#SBATCH --gres=gpu:L40:1
#SBATCH --exclude=tdk-bm4,euler2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=xf_%A_%a.out
# Focused experiment: TAR vs TAR + pruning with the removal amount chosen by nested cross-fitting.
# Both arms run in the same process, so they share the fold's TAR fine-tune (clean paired comparison).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MULTABENCH_DYNAMIC_PADDING=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
[ -f .venv_full/.ok ] || { echo "missing .venv_full"; exit 1; }
source .venv_full/bin/activate
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "NO GPU on $(hostname)"; exit 1; }
DS=(BIN_TEXT_KICKSTARTER_FUNDING REG_TEXT_BABIES_PRICES BIN_TEXT_FAKE_JOB_POSTING MUL_TEXT_DATA_SCIENTIST_SALARY
    BIN_TEXT_JIGSAW_TOXICITY)
ds=${DS[$SLURM_ARRAY_TASK_ID]}
echo "== $ds on $(hostname)"
PYTHONPATH=. python sae_pilot.py run --out_dir runs_xf --emb_cache emb_cache/$ds --datasets $ds \
  --encoders e5-small-tar e5-small-tar-gsae-xf
