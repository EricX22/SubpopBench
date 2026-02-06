#!/bin/bash
set -euo pipefail

# =========================
# User knobs (EDIT THESE)
# =========================
REPO_DIR="/sfs/gpfs/tardis/home/jrg4wx/projects/Concept_Reweighting/SubpopBench"
DATA_DIR="/scratch/jrg4wx/subpop_bench/data"
OUT_DIR="/scratch/jrg4wx/subpop_bench/output"

DATASET="${DATASET:-MetaShift}"            # e.g. MetaShift or CheXpertNoFinding
TRAIN_ATTR="${TRAIN_ATTR:-no}"             # no for unknown attributes
TAG="${TAG:-${DATASET}_full_$(date +%Y%m%d)_attrNo}"

# Algorithms to run
ALGS=(ERM GroupDRO DFR CRT JTT CR)

# Sweep size
HPARAMS_SEEDS="$(seq 0 15)"                # 0..15
SEEDS=(0)                                  # add more if you want trials

# Stage-1 (used by 2-stage algos + CR residuals)
STAGE1_ALGO="ERM"
STAGE1_FOLDER="${STAGE1_FOLDER:-${TAG}_stage1}"
STAGE1_HPSEED="${STAGE1_HPSEED:-0}"
STAGE1_SEED="${STAGE1_SEED:-0}"

# -------------------------
# CR settings
# -------------------------
CR_USE_CONCEPTS="${CR_USE_CONCEPTS:-1}"
CR_USE_RESID="${CR_USE_RESID:-1}"
CR_RESID_DIM="${CR_RESID_DIM:-64}"

CR_TASK_ID="${CR_TASK_ID:-${DATASET,,}_attr${TRAIN_ATTR}}"
CR_MODALITY="${CR_MODALITY:-natural_image}"

LABEL0="${LABEL0:-not_target}"
LABEL1="${LABEL1:-target}"

# Build CR_LABELS_JSON automatically unless user already set it.
if [ -z "${CR_LABELS_JSON:-}" ]; then
  CR_LABELS_JSON="$(printf '[{"name":"%s"},{"name":"%s"}]' "$LABEL0" "$LABEL1")"
fi
export CR_LABELS_JSON


# Cache/runtime controls (safer defaults)
CLIP_BATCH="${CLIP_BATCH:-64}"
CLIP_WORKERS="${CLIP_WORKERS:-2}"
FEAT_BATCH="${FEAT_BATCH:-128}"
FEAT_WORKERS="${FEAT_WORKERS:-4}"
RIDGE_LAM="${RIDGE_LAM:-0.001}"
PCA_K="${PCA_K:-64}"

# Slurm controls
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-6}"
SBATCH_ACCT="${SBATCH_ACCT:--A zhangmlgroup}"
SBATCH_PART="${SBATCH_PART:--p gpu}"
SBATCH_GRES="${SBATCH_GRES:---gres=gpu:1}"
SBATCH_CPU="${SBATCH_CPU:--c 4}"
SBATCH_MEM="${SBATCH_MEM:---mem=24G}"
SBATCH_TIME="${SBATCH_TIME:--t 0-12:00:00}"

# =========================
# Derived paths
# =========================
mkdir -p "${REPO_DIR}/slurm" "${OUT_DIR}" "${OUT_DIR}/${TAG}"
cd "${REPO_DIR}"

# Where stage-1 ERM run will live
STAGE1_RUN_DIR="${OUT_DIR}/${STAGE1_FOLDER}_${TRAIN_ATTR}/${DATASET}_${STAGE1_ALGO}_hparams${STAGE1_HPSEED}_seed${STAGE1_SEED}"
STAGE1_CKPT="${STAGE1_RUN_DIR}/model.pkl"

# CR artifact paths (kept inside repo artifacts/)
BANK_PATH="artifacts/concept_banks/${CR_TASK_ID}.json"
META_PATH="artifacts/concept_scores/${CR_TASK_ID}_meta.json"
C_TR="artifacts/concept_scores/${CR_TASK_ID}_tr.pt"
C_VA="artifacts/concept_scores/${CR_TASK_ID}_va.pt"
C_TE="artifacts/concept_scores/${CR_TASK_ID}_te.pt"
F_VA="artifacts/feat_cache/${CR_TASK_ID}_feat_va.pt"
F_TE="artifacts/feat_cache/${CR_TASK_ID}_feat_te.pt"
R_VA="artifacts/resid_cache/${CR_TASK_ID}_resid_va_pca${PCA_K}.pt"
R_TE="artifacts/resid_cache/${CR_TASK_ID}_resid_te_pca${PCA_K}.pt"
R_META="artifacts/resid_cache/${CR_TASK_ID}_resid_meta_pca${PCA_K}.pt"

in_flight () { squeue -u "$USER" -h -n "${TAG}" | wc -l; }

wait_for_slot () {
  while [ "$(in_flight)" -ge "${MAX_IN_FLIGHT}" ]; do
    echo "[throttle] in-flight=$(in_flight) >= ${MAX_IN_FLIGHT}, sleeping 30s..."
    sleep 30
  done
}

run_dir () {
  local algo="$1" hp="$2" seed="$3"
  echo "${OUT_DIR}/${TAG}_${TRAIN_ATTR}/${DATASET}_${algo}_hparams${hp}_seed${seed}"
}

is_done () {
  local d="$1"
  [ -f "${d}/done" ] || [ -f "${d}/final_results.pkl" ]
}

# =========================
# Helpers: JSON-safe CR labels + CR hparams
# =========================
CR_LABELS_JSON="$(printf '[{"name":"%s"},{"name":"%s"}]' "${LABEL0}" "${LABEL1}")"

# Make a JSON string literal for labels (double-encoding so it survives --hparams '...')
LABELS_ESCAPED="$(python - <<'PY'
import os, json
obj = json.loads(os.environ["CR_LABELS_JSON"])
print(json.dumps(json.dumps(obj)))
PY
)"

make_cr_hparams_json () {
  # IMPORTANT: include explicit paths so we never accidentally fall back to Waterbirds defaults
  printf '{"train_attr":"%s","cr_task_id":"%s","cr_use_concepts":true,"cr_use_resid":true,"cr_resid_dim":%s,"cr_modality":"%s","cr_labels_json":%s,"cr_concept_meta_path":"%s","cr_concept_path_tr":"%s","cr_concept_path_va":"%s","cr_concept_path_te":"%s","cr_resid_path_va":"%s","cr_resid_path_te":"%s"}' \
    "${TRAIN_ATTR}" "${CR_TASK_ID}" "${CR_RESID_DIM}" "${CR_MODALITY}" "${LABELS_ESCAPED}" \
    "${META_PATH}" "${C_TR}" "${C_VA}" "${C_TE}" "${R_VA}" "${R_TE}"
}

CR_HPARAMS_JSON="$(make_cr_hparams_json)"

# =========================
# 1) Submit Stage-1 ERM (hp0 seed0) if missing
# =========================
stage1_jobid=""

if [ -f "${STAGE1_CKPT}" ]; then
  echo "[stage1] Found stage1 ckpt: ${STAGE1_CKPT}"
else
  echo "[stage1] Missing stage1 ckpt. Submitting Stage-1 ERM hp${STAGE1_HPSEED} seed${STAGE1_SEED} -> ${STAGE1_RUN_DIR}"
  wait_for_slot
  stage1_jobid="$(
    sbatch -J "${TAG}" \
      ${SBATCH_ACCT} ${SBATCH_PART} ${SBATCH_GRES} ${SBATCH_CPU} ${SBATCH_MEM} ${SBATCH_TIME} \
      -o "${REPO_DIR}/slurm/${TAG}_${DATASET}_STAGE1_ERM_hp${STAGE1_HPSEED}_seed${STAGE1_SEED}_%j.out" \
      -e "${REPO_DIR}/slurm/${TAG}_${DATASET}_STAGE1_ERM_hp${STAGE1_HPSEED}_seed${STAGE1_SEED}_%j.err" \
      --parsable \
      --wrap "
        set -euo pipefail
        cd '${REPO_DIR}'
        source '$HOME/anaconda3/etc/profile.d/conda.sh'
        conda activate subpop_bench

        export PYTORCH_NVML_BASED_CUDA_CHECK=0
        export TORCH_USE_NVML=0
        export CUDA_MODULE_LOADING=LAZY

        python -m subpopbench.train \
          --dataset '${DATASET}' \
          --algorithm '${STAGE1_ALGO}' \
          --train_attr '${TRAIN_ATTR}' \
          --data_dir '${DATA_DIR}' \
          --output_dir '${OUT_DIR}' \
          --output_folder_name '${STAGE1_FOLDER}' \
          --hparams_seed '${STAGE1_HPSEED}' \
          --seed '${STAGE1_SEED}'
      " | awk '{print $1}'
  )"
  echo "[stage1] Submitted stage1 jobid=${stage1_jobid}"
fi

# =========================
# 2) Submit CR preflight caches job (depends on stage1 if we just submitted it)
# =========================
# Preflight requires OpenAI key (for concept bank) + GPU (CLIP caching)
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[FATAL] OPENAI_API_KEY is not set but CR concept bank generation requires it."
  echo "        export OPENAI_API_KEY=... and re-run."
  exit 1
fi

preflight_dep=()
if [ -n "${stage1_jobid}" ]; then
  preflight_dep=(--dependency="afterok:${stage1_jobid}")
fi

echo "[preflight] Submitting CR cache build job for task_id=${CR_TASK_ID}"
wait_for_slot
preflight_jobid="$(
  sbatch -J "${TAG}" \
    ${SBATCH_ACCT} ${SBATCH_PART} ${SBATCH_GRES} ${SBATCH_CPU} ${SBATCH_MEM} ${SBATCH_TIME} \
    "${preflight_dep[@]}" \
    -o "${REPO_DIR}/slurm/${TAG}_${DATASET}_CR_PREFLIGHT_%j.out" \
    -e "${REPO_DIR}/slurm/${TAG}_${DATASET}_CR_PREFLIGHT_%j.err" \
    --parsable \
    --wrap "
      set -euo pipefail
      cd '${REPO_DIR}'
      source '$HOME/anaconda3/etc/profile.d/conda.sh'
      conda activate subpop_bench

      export PYTORCH_NVML_BASED_CUDA_CHECK=0
      export TORCH_USE_NVML=0
      export CUDA_MODULE_LOADING=LAZY

      mkdir -p artifacts/concept_banks artifacts/concept_scores artifacts/feat_cache artifacts/resid_cache

      # 2.1 Generate concept bank (LLM)
      python -m subpopbench.scripts.generate_concept_bank \
        --task-id '${CR_TASK_ID}' \
        --modality '${CR_MODALITY}' \
        --labels '${CR_LABELS_JSON}' \
        --out '${BANK_PATH}' \
        --model 'gpt-4o-mini' \
        --temperature 0.2

      # 2.2 Compile concept meta (applies channel inclusion/exclusion defaults)
      python -m subpopbench.scripts.compile_concept_meta \
        --bank '${BANK_PATH}' \
        --out '${META_PATH}'

      # 2.3 Cache CLIP concepts for tr/va/te
      python -m subpopbench.scripts.cache_clip_concepts \
        --dataset '${DATASET}' --data-dir '${DATA_DIR}' \
        --split tr --meta '${META_PATH}' --out '${C_TR}' \
        --batch-size '${CLIP_BATCH}' --num-workers '${CLIP_WORKERS}' --device cuda --fp16

      python -m subpopbench.scripts.cache_clip_concepts \
        --dataset '${DATASET}' --data-dir '${DATA_DIR}' \
        --split va --meta '${META_PATH}' --out '${C_VA}' \
        --batch-size '${CLIP_BATCH}' --num-workers '${CLIP_WORKERS}' --device cuda --fp16

      python -m subpopbench.scripts.cache_clip_concepts \
        --dataset '${DATASET}' --data-dir '${DATA_DIR}' \
        --split te --meta '${META_PATH}' --out '${C_TE}' \
        --batch-size '${CLIP_BATCH}' --num-workers '${CLIP_WORKERS}' --device cuda --fp16

      # 2.4 Cache stage1 features (va/te) using stage1 checkpoint
      python -m subpopbench.scripts.cache_stage1_feats \
        --dataset '${DATASET}' --data_dir '${DATA_DIR}' \
        --split va --train_attr '${TRAIN_ATTR}' --image_arch resnet_sup_in1k \
        --stage1_ckpt '${STAGE1_CKPT}' \
        --out '${F_VA}'

      python -m subpopbench.scripts.cache_stage1_feats \
        --dataset '${DATASET}' --data_dir '${DATA_DIR}' \
        --split te --train_attr '${TRAIN_ATTR}' --image_arch resnet_sup_in1k \
        --stage1_ckpt '${STAGE1_CKPT}' \
        --out '${F_TE}'

      # 2.5 Fit residuals (ridge on VA), PCA -> k dims
      python -m subpopbench.scripts.fit_residuals_pca \
        --concept_meta '${META_PATH}' \
        --concept_va   '${C_VA}' \
        --concept_te   '${C_TE}' \
        --feat_va      '${F_VA}' \
        --feat_te      '${F_TE}' \
        --ridge_lam '${RIDGE_LAM}' --pca_k '${PCA_K}' \
        --out_va   '${R_VA}' \
        --out_te   '${R_TE}' \
        --out_meta '${R_META}'

      echo '[preflight] DONE'
    " | awk '{print $1}'
)"
echo "[preflight] Submitted preflight jobid=${preflight_jobid}"

# =========================
# 3) Submit training jobs for all ALGS
# =========================
submit_train () {
  local algo="$1" hp="$2" seed="$3"
  local d; d="$(run_dir "$algo" "$hp" "$seed")"

  if is_done "$d"; then
    echo "[skip done] ${d}"
    return
  fi

  mkdir -p "$d"
  wait_for_slot

  local dep_args=()
  local extra_args=()
  local hparams_arg=""

  # 2-stage methods depend on stage1 if stage1 was submitted now
  if [[ "$algo" == "DFR" || "$algo" == "CRT" || "$algo" == "JTT" ]]; then
    if [ -n "${stage1_jobid}" ]; then
      dep_args=(--dependency="afterok:${stage1_jobid}")
    fi
    extra_args+=(--stage1_folder "${STAGE1_FOLDER}" --stage1_algo "${STAGE1_ALGO}" --pretrained "${STAGE1_CKPT}")
  fi

  # CR depends on preflight caches and uses explicit CR hparams
  if [[ "$algo" == "CR" ]]; then
    dep_args=(--dependency="afterok:${preflight_jobid}")
    extra_args+=(--stage1_folder "${STAGE1_FOLDER}" --stage1_algo "${STAGE1_ALGO}" --pretrained "${STAGE1_CKPT}")
    hparams_arg="${CR_HPARAMS_JSON}"
  fi

  # Other algos: allow normal SubpopBench hparams registry; no custom hparams needed
  if [[ "$algo" != "CR" ]]; then
    hparams_arg=""   # use registry sampling by --hparams_seed
  fi

  # Build sbatch wrap command
  local wrap_cmd="
    set -euo pipefail
    cd '${REPO_DIR}'
    source '$HOME/anaconda3/etc/profile.d/conda.sh'
    conda activate subpop_bench

    export PYTORCH_NVML_BASED_CUDA_CHECK=0
    export TORCH_USE_NVML=0
    export CUDA_MODULE_LOADING=LAZY

    python -m subpopbench.train \
      --dataset '${DATASET}' \
      --algorithm '${algo}' \
      --train_attr '${TRAIN_ATTR}' \
      --data_dir '${DATA_DIR}' \
      --output_dir '${OUT_DIR}' \
      --output_folder_name '${TAG}' \
      --hparams_seed '${hp}' \
      --seed '${seed}' \
  "

  # Optional: pass custom CR hparams json
  if [ -n "${hparams_arg}" ]; then
    wrap_cmd="${wrap_cmd} --hparams '${hparams_arg}' "
  fi

  # Optional: pass extra args
  if [ "${#extra_args[@]}" -gt 0 ]; then
    # shellcheck disable=SC2145
    wrap_cmd="${wrap_cmd} ${extra_args[*]} "
  fi

  sbatch -J "${TAG}" \
    ${SBATCH_ACCT} ${SBATCH_PART} ${SBATCH_GRES} ${SBATCH_CPU} ${SBATCH_MEM} ${SBATCH_TIME} \
    "${dep_args[@]}" \
    -o "${REPO_DIR}/slurm/${TAG}_${DATASET}_${algo}_hp${hp}_seed${seed}_%j.out" \
    -e "${REPO_DIR}/slurm/${TAG}_${DATASET}_${algo}_hp${hp}_seed${seed}_%j.err" \
    --wrap "${wrap_cmd}"
}

echo "[submit] Launching sweeps for: ${ALGS[*]}"
for algo in "${ALGS[@]}"; do
  for hp in ${HPARAMS_SEEDS}; do
    for seed in "${SEEDS[@]}"; do
      submit_train "${algo}" "${hp}" "${seed}"
    done
  done
done

echo
echo "============================================================"
echo "Submitted all jobs under TAG=${TAG} (train_attr=${TRAIN_ATTR})"
echo "Stage1 ckpt expected at: ${STAGE1_CKPT}"
echo "CR preflight jobid: ${preflight_jobid}"
echo
echo "Monitor:"
echo "  squeue -u $USER | head"
echo "Summarize:"
echo "  python -m subpopbench.scripts.summarize_progress --root ${OUT_DIR}/${TAG}_${TRAIN_ATTR}"
echo "============================================================"
