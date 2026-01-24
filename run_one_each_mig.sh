#!/bin/bash
set -euo pipefail

# ----------------------------
# User-configurable paths
# ----------------------------
REPO_DIR="/sfs/gpfs/tardis/home/jrg4wx/projects/Concept_Reweighting/SubpopBench"
DATA_DIR="/scratch/jrg4wx/subpop_bench/data"
OUT_DIR="${REPO_DIR}/output"

# Stage-1 ERM checkpoint folder (used by DFR/CRT/JTT/CR depending on implementation)
STAGE1_FOLDER="vanilla_attrNo"
STAGE1_ALGO="ERM"

# One-run tag
TAG="one_each_mig_20260114_011741"
BASE_OUTFOLDER="${TAG}_attrNo"

cd "$REPO_DIR"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate subpop_bench
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export TORCH_USE_NVML=0
export CUDA_MODULE_LOADING=LAZY



echo "Repo:   $REPO_DIR"
echo "Data:   $DATA_DIR"
echo "Out:    $OUT_DIR/$BASE_OUTFOLDER"
echo "GPU:    $(nvidia-smi -L || true)"
echo

# ----------------------------
# Common args
# ----------------------------
COMMON_ARGS=(
  --dataset Waterbirds
  --train_attr no
  --data_dir "$DATA_DIR"
  --output_dir "$OUT_DIR"
  --hparams_seed 0
  --seed 0
)

# ----------------------------
# 1) ERM
# ----------------------------
# echo "=== [1/6] ERM ==="
# python subpopbench/train.py \
#   "${COMMON_ARGS[@]}" \
#   --algorithm ERM \
#   --output_folder_name "${BASE_OUTFOLDER}/ERM"

# # ----------------------------
# # 2) GroupDRO
# # ----------------------------
# echo "=== [2/6] GroupDRO ==="
# python subpopbench/train.py \
#   "${COMMON_ARGS[@]}" \
#   --algorithm GroupDRO \
#   --output_folder_name "${BASE_OUTFOLDER}/GroupDRO"

# ----------------------------
# 3) DFR (2-stage; trains on val split internally)
# # ----------------------------
# echo "=== [3/6] DFR ==="
# python subpopbench/train.py \
#   "${COMMON_ARGS[@]}" \
#   --algorithm DFR \
#   --output_folder_name "${BASE_OUTFOLDER}/DFR" \
#   --stage1_folder "$STAGE1_FOLDER" \
#   --stage1_algo "$STAGE1_ALGO"

# # ----------------------------
# # 4) CRT (2-stage variant)
# # ----------------------------
# echo "=== [4/6] CRT ==="
# python subpopbench/train.py \
#   "${COMMON_ARGS[@]}" \
#   --algorithm CRT \
#   --output_folder_name "${BASE_OUTFOLDER}/CRT" \
#   --stage1_folder "$STAGE1_FOLDER" \
#   --stage1_algo "$STAGE1_ALGO"

# ----------------------------
# 5) JTT (often 2-stage; keeping stage1 args doesn't hurt even if ignored)
# ----------------------------
echo "=== [5/6] JTT ==="
python subpopbench/train.py \
  "${COMMON_ARGS[@]}" \
  --algorithm JTT \
  --output_folder_name "${BASE_OUTFOLDER}/JTT" \
  --stage1_folder "$STAGE1_FOLDER" \
  --stage1_algo "$STAGE1_ALGO"

# ----------------------------
# 6) CR (concept + resid tuned to your strong run)
# ----------------------------
# echo "=== [6/6] CR (concept + resid tuned) ==="
# python subpopbench/train.py \
#   "${COMMON_ARGS[@]}" \
#   --algorithm CR \
#   --output_folder_name "${BASE_OUTFOLDER}/CR" \
#   --stage1_folder "$STAGE1_FOLDER" \
#   --stage1_algo "$STAGE1_ALGO" \
#   --hparams '{
#     "cr_use_concepts": true,
#     "cr_use_resid": true,
#     "cr_feature_mode": "concat_plus_resid",

#     "cr_feat_dropout": 0.5,
#     "cr_reg": 1e-5,

#     "cr_concept_block_channels": ["environment_context", "imaging_artifacts"],

#     "cr_concept_meta_path": "artifacts/concept_scores/waterbirds_binary_meta.json",
#     "cr_concept_path_va": "artifacts/concept_scores/waterbirds_binary_va.pt",
#     "cr_concept_path_te": "artifacts/concept_scores/waterbirds_binary_te.pt",

#     "cr_resid_dim": 64,
#     "cr_resid_path_va": "artifacts/resid_cache/wb_resid_va_pca64.pt",
#     "cr_resid_path_te": "artifacts/resid_cache/wb_resid_te_pca64.pt",

#     "stage1_hparams_seed": 0,
#     "stage1_seed": 0
#   }'


echo
echo "All runs finished."
echo "Results folder: $OUT_DIR/$BASE_OUTFOLDER"
echo

# ----------------------------
# Quick summary helper
# Prints the last reported val/test avg & worst from out.txt if present
# ----------------------------
echo "=== QUICK SUMMARY (from out.txt tail) ==="
for ALG in ERM GroupDRO DFR CRT JTT CR; do
  RUN_DIR=$(ls -d "$OUT_DIR/$BASE_OUTFOLDER/$ALG"/Waterbirds_"$ALG"_hparams0_seed0 2>/dev/null | head -n 1 || true)
  echo "--- $ALG ---"
  if [[ -n "$RUN_DIR" && -f "$RUN_DIR/out.txt" ]]; then
    tail -n 25 "$RUN_DIR/out.txt" | sed -n '1,25p'
  else
    echo "No out.txt found at: $RUN_DIR"
  fi
  echo
done

echo "Tip: for clean tables, use collect_results on $OUT_DIR/$BASE_OUTFOLDER"
