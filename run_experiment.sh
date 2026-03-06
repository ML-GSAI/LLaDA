#!/bin/bash
set -euo pipefail

cd ~/UQ4DLLM
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"  # auto | llada | ar
N_EXAMPLES="${N_EXAMPLES:-200}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DATASET="${DATASET:-triviaqa}"            # triviaqa | mmlu
DATA_SPLIT="${DATA_SPLIT:-validation}"    # triviaqa: validation, mmlu: test/dev/validation
DEVICE_MAP="${DEVICE_MAP:-auto}"          # none | auto

SAFE_MODEL_TAG="$(echo "${MODEL_NAME}" | tr '/:' '__')"
SAFE_DATASET_TAG="$(echo "${DATASET}" | tr '/:' '__')"

TRAJ_JSON="results/${SAFE_MODEL_TAG}_${SAFE_DATASET_TAG}_traj_uq.json"
SEM_JSON="results/${SAFE_MODEL_TAG}_${SAFE_DATASET_TAG}_sem_entropy.json"
VC_JSON="results/${SAFE_MODEL_TAG}_${SAFE_DATASET_TAG}_verbal_conf.json"

TRAJ_LOG="results/${SAFE_MODEL_TAG}_traj.log"
SEM_LOG="results/${SAFE_MODEL_TAG}_sem.log"
VC_LOG="results/${SAFE_MODEL_TAG}_vc.log"

echo "Starting full experiment suite at $(date)"
echo "Model: ${MODEL_NAME}"
echo "Model family: ${MODEL_FAMILY}"
echo "n_examples: ${N_EXAMPLES}"
echo "trust_remote_code: ${TRUST_REMOTE_CODE}"
echo "dataset: ${DATASET} (${DATA_SPLIT})"
echo "device_map: ${DEVICE_MAP}"

TRUST_ARG=""
if [ "${TRUST_REMOTE_CODE}" = "1" ]; then
  TRUST_ARG="--trust_remote_code"
fi

# Official-like LLaDA sampling defaults for MMLU from EVAL.md:
# gen_length=3, block_length=3, no special eos confidence inference here.
TRAJ_STEPS="${TRAJ_STEPS:-64}"
TRAJ_GEN_LENGTH="${TRAJ_GEN_LENGTH:-32}"
SEM_STEPS="${SEM_STEPS:-32}"
SEM_GEN_LENGTH="${SEM_GEN_LENGTH:-32}"
VC_STEPS="${VC_STEPS:-64}"
VC_GEN_LENGTH="${VC_GEN_LENGTH:-32}"

if [ "${DATASET}" = "mmlu" ]; then
  TRAJ_STEPS="3"
  TRAJ_GEN_LENGTH="3"
  SEM_STEPS="3"
  SEM_GEN_LENGTH="3"
  VC_STEPS="3"
  VC_GEN_LENGTH="3"
  if [ "${DATA_SPLIT}" = "validation" ]; then
    DATA_SPLIT="test"
  fi
fi

python3 experiments/run_trajectory_uq.py \
  --model_name "${MODEL_NAME}" \
  --model_family "${MODEL_FAMILY}" \
  ${TRUST_ARG} \
  --device_map "${DEVICE_MAP}" \
  --dataset "${DATASET}" \
  --data_split "${DATA_SPLIT}" \
  --n_examples "${N_EXAMPLES}" \
  --steps "${TRAJ_STEPS}" \
  --gen_length "${TRAJ_GEN_LENGTH}" \
  --output "${TRAJ_JSON}" \
  2>&1 | tee "${TRAJ_LOG}"

python3 experiments/run_semantic_entropy.py \
  --model_name "${MODEL_NAME}" \
  --model_family "${MODEL_FAMILY}" \
  ${TRUST_ARG} \
  --device_map "${DEVICE_MAP}" \
  --dataset "${DATASET}" \
  --data_split "${DATA_SPLIT}" \
  --n_examples "${N_EXAMPLES}" \
  --k_samples 5 \
  --steps "${SEM_STEPS}" \
  --gen_length "${SEM_GEN_LENGTH}" \
  --temperature 1.0 \
  --traj_results "${TRAJ_JSON}" \
  --output "${SEM_JSON}" \
  2>&1 | tee "${SEM_LOG}"

python3 experiments/run_verbal_confidence.py \
  --model_name "${MODEL_NAME}" \
  --model_family "${MODEL_FAMILY}" \
  ${TRUST_ARG} \
  --device_map "${DEVICE_MAP}" \
  --dataset "${DATASET}" \
  --data_split "${DATA_SPLIT}" \
  --n_examples "${N_EXAMPLES}" \
  --steps "${VC_STEPS}" \
  --gen_length "${VC_GEN_LENGTH}" \
  --vc_gen_length "${VC_GEN_LENGTH}" \
  --output "${VC_JSON}" \
  2>&1 | tee "${VC_LOG}"

echo "All experiments done at $(date)"
echo "Trajectory result: ${TRAJ_JSON}"
echo "Semantic result:   ${SEM_JSON}"
echo "Verbal result:     ${VC_JSON}"
