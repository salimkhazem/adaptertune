#!/usr/bin/env bash
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASETS=(cifar10 cifar100 svhn pets food101)
MODELS=(vit_small vit_base deit_tiny)
METHODS=(head_only full_finetune adapter_tune)
NUM_SEEDS="${NUM_SEEDS:-3}"

# Auto batch sizes per model (override via env if desired).
BASE_BS=${BASE_BS:-2048}
BASE_LR=${BASE_LR:-1e-3}
BATCH_VIT_SMALL=${BATCH_VIT_SMALL:-1024}
BATCH_VIT_BASE=${BATCH_VIT_BASE:-1024}
BATCH_DEIT_TINY=${BATCH_DEIT_TINY:-1024}

RESULTS_DIR="${ROOT}/results"
LOG_DIR="${RESULTS_DIR}/launcher_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

get_batch_size() {
  case "$1" in
    vit_small) echo "${BATCH_VIT_SMALL}" ;;
    vit_base) echo "${BATCH_VIT_BASE}" ;;
    deit_tiny) echo "${BATCH_DEIT_TINY}" ;;
    *) echo "${BASE_BS}" ;;
  esac
}

scale_lr() {
  local bs="$1"
  python - <<PY
bs=${bs}
base_bs=${BASE_BS}
base_lr=${BASE_LR}
print(base_lr * (bs / base_bs))
PY
}

detect_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  else
    GPU_LIST=(0)
  fi
  echo "${GPU_LIST[@]}"
}

GPU_LIST=( $(detect_gpus) )
NUM_GPUS="${#GPU_LIST[@]}"
MAX_JOBS="${MAX_JOBS:-$NUM_GPUS}"
if [ "${MAX_JOBS}" -lt 1 ]; then
  MAX_JOBS=1
fi

echo "=========================================="
echo "Full benchmark: datasets x backbones x methods x seeds"
echo "=========================================="
echo "Datasets: ${DATASETS[*]}"
echo "Backbones: ${MODELS[*]}"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${NUM_SEEDS}"
echo "GPUs: ${GPU_LIST[*]} (max parallel jobs: ${MAX_JOBS})"
echo "Logs: ${LOG_DIR}"
echo ""

launch_count=0

wait_for_slot() {
  while [ "$(jobs -pr | wc -l)" -ge "${MAX_JOBS}" ]; do
    if ! wait -n 2>/dev/null; then
      local pid
      pid="$(jobs -pr | head -n 1)"
      if [ -n "$pid" ]; then
        wait "$pid" || true
      else
        break
      fi
    fi
  done
}

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
      for seed in $(seq 0 $((NUM_SEEDS - 1))); do
        gpu="${GPU_LIST[$((launch_count % NUM_GPUS))]}"
        bs=$(get_batch_size "${model}")
        lr=$(scale_lr "${bs}")
        exp_name="${dataset}_${model}_${method}_s${seed}"
        log_file="${LOG_DIR}/${exp_name}.log"

        cmd=(
          python -m scripts.train
          --config configs/base.yaml
          --config configs/datasets/${dataset}.yaml
          --config configs/models/${model}.yaml
          --override data.batch_size=${bs}
          --override train.lr=${lr}
          --override method.name=${method}
          --override seed=${seed}
          --override logging.exp_name=${exp_name}
          --device 0
        )

        echo "[GPU ${gpu}] ${dataset} | ${model} | ${method} | seed ${seed} | bs ${bs} | lr ${lr}"
        (
          CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" >"${log_file}" 2>&1
          echo $? > "${log_file}.exit"
        ) &

        launch_count=$((launch_count + 1))
        wait_for_slot
      done
    done
  done
done

wait || true

echo ""
echo "=========================================="
echo "All runs completed. Checking for failures..."
echo "=========================================="

failed=0
for exit_file in "${LOG_DIR}"/*.exit; do
  if [ -f "${exit_file}" ]; then
    code="$(cat "${exit_file}")"
    if [ "${code}" != "0" ]; then
      echo "FAILED: ${exit_file%.exit} (exit ${code})"
      failed=$((failed + 1))
    fi
  fi
done

if [ "${failed}" -gt 0 ]; then
  echo "Total failures: ${failed}"
else
  echo "All jobs completed successfully."
fi
