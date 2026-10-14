#!/usr/bin/env bash
# Run the relevance-filtering pilot (and later the full benchmark) on a lab GPU machine.
#
#   bash run_lab.sh setup            # once: virtualenv + dependencies
#   bash run_lab.sh pilot            # 3 datasets x 6 folds, all frozen variants (~1-3 GPU hours, mostly embedding)
#   bash run_lab.sh full ENC...      # all 20 text datasets for the chosen encoders, e.g.: full e5-small e5-small-relsae
#   bash run_lab.sh summary          # tables: vs our Frozen, and gain vs the official TAR gain
#
# Results: runs_lab/*.json (one per dataset x fold x encoder; finished runs are skipped, so re-running resumes).
# Frozen E5 embeddings are cached in emb_cache/ and reused across folds, encoders and runs.
# Run inside tmux/screen so it survives a disconnect:  tmux new -s mtb  ...  (detach: Ctrl-b d)
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-runs_lab}
CACHE=${CACHE:-emb_cache}
export MULTABENCH_DYNAMIC_PADDING=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PILOT_DATASETS="BIN_TEXT_FAKE_JOB_POSTING MUL_TEXT_WINE_REVIEW REG_TEXT_BOOK_READABILITY"
PILOT_ENCODERS="e5-small e5-small-stack e5-small-relsae e5-small-relsae-add e5-small-relsae-uni e5-small-relsae-rand e5-small-reldim e5-small-pls"
ALL_TEXT="BIN_TEXT_FAKE_JOB_POSTING BIN_TEXT_JIGSAW_TOXICITY BIN_TEXT_KICKSTARTER_FUNDING MUL_TEXT_DATA_SCIENTIST_SALARY
MUL_TEXT_MICHELIN_RESTAURANTS MUL_TEXT_PRODUCT_SENTIMENT MUL_TEXT_SPOTIFY_GENRES MUL_TEXT_US_ACCIDENTS MUL_TEXT_WINE_REVIEW
MUL_TEXT_WOMEN_CLOTHING_REVIEW REG_TEXT_BABIES_PRICES REG_TEXT_BOOK_PRICE REG_TEXT_BOOK_READABILITY
REG_TEXT_MERCARI_MARKETPLACE REG_TEXT_MONTGOMERY_SALARIES REG_TEXT_SCIMAGOJR_IMPACT REG_TEXT_ROTTEN_TOMATOES
REG_TEXT_VANCOUVER_SALARIES REG_TEXT_VIDEO_GAMES_SALES REG_TEXT_ZOMATO_RESTAURANTS"

case "${1:-}" in
  setup)
    python3 -m venv .venv && source .venv/bin/activate
    pip install -q --upgrade pip
    python -c "import torch" 2>/dev/null || pip install -q torch
    pip install -q lightgbm scikit-learn pandas scipy transformers peft skrub kagglehub python-dotenv "autogluon.core" pytest
    pip uninstall -y -q torchao 2>/dev/null || true   # old torchao versions break new peft; not needed
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    PYTHONPATH=. python -m pytest tests/test_sae.py -q -k "relevance_pipeline and relsae-bin" 2>&1 | tail -1
    ;;
  pilot)
    source .venv/bin/activate
    for ds in $PILOT_DATASETS; do
      PYTHONPATH=. python sae_pilot.py run --out_dir "$OUT" --emb_cache "$CACHE" --datasets "$ds" \
        --encoders $PILOT_ENCODERS 2>&1 | tee -a "$OUT.log" | grep -E "fold [0-9] |Error|Traceback" || true
    done
    PYTHONPATH=. python sae_pilot.py summarize --out_dir "$OUT" --baselines e5-small
    ;;
  full)
    shift; source .venv/bin/activate
    ENCODERS="${*:-e5-small e5-small-relsae}"
    for ds in $ALL_TEXT; do
      PYTHONPATH=. python sae_pilot.py run --out_dir "$OUT" --emb_cache "$CACHE" --datasets "$ds" \
        --encoders $ENCODERS 2>&1 | tee -a "$OUT.log" | grep -E "fold [0-9] |Error|Traceback" || true
    done
    PYTHONPATH=. python sae_pilot.py summarize --out_dir "$OUT" --baselines e5-small
    ;;
  summary)
    source .venv/bin/activate
    PYTHONPATH=. python sae_pilot.py summarize --out_dir "$OUT" --baselines e5-small
    ;;
  *) sed -n 2,13p "$0" ;;
esac
