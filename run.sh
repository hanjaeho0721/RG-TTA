#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATASET="${DATASET:-fairface}"
ATTRIBUTE="${ATTRIBUTE:-race}"
GALLERY_SPLIT="${GALLERY_SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"

PROMPT_CSV="${PROMPT_CSV:-./data/prompt_templates.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
RUN_NAME="${RUN_NAME:-rlcf_example}"

POLICY_CLIP="${POLICY_CLIP:-ViT-B/16}"
REWARD_CLIP="${REWARD_CLIP:-ViT-L/14}"

python -u -m RLCF_Debiasing.run_tta_experiment \
  --dataset "$DATASET" \
  --attribute "$ATTRIBUTE" \
  --gallery_split "$GALLERY_SPLIT" \
  --device "$DEVICE" \
  --prompt_csv "$PROMPT_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --policy_clip_model_name "$POLICY_CLIP" \
  --reward_clip_model_name "$REWARD_CLIP" \
  --reward_mode "${REWARD_MODE:-clip_plus_debias}" \
  --debias_lambda "${DEBIAS_LAMBDA:-1.0}" \
  --debias_score_base "${DEBIAS_SCORE_BASE:-instance_popularity}" \
  --debias_score_trace "${DEBIAS_SCORE_TRACE:-none}" \
  --subspace_mode "${SUBSPACE_MODE:-test}" \
  --subspace_top_r "${SUBSPACE_TOP_R:-30}" \
  --tta_steps "${TTA_STEPS:-30}" \
  --sample_k "${SAMPLE_K:-16}" \
  --lr "${LR:-1e-4}" \
  --wandb_mode "${WANDB_MODE:-disabled}"
