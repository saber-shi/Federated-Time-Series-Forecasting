#!/usr/bin/env bash
set -euo pipefail

# Run the PWRH-only paper evaluation, model-heterogeneity sweep, and
# residual-head ablation for two routing configurations:
#   1. hard routing with server weight power 0
#   2. adaptive routing with server weight power 1
#
# Each family/variant owns a separate directory. W&B remains disabled and the
# robust runners retain local CSV metrics and raw process logs.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/experiments/H4-sequence-pattern-alignment"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
SUITE_RUN_NAME="${RUN_NAME:-pwrh_hard_vs_adaptive_power1_$(date +%Y%m%d_%H%M%S)}"
SUITE_DIR="$RESULT_ROOT/$SUITE_RUN_NAME"

GPU_ID="${GPU_ID:-0}"
read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS:-$GPU_ID}"

EVALUATION_BASE_PORT="${EVALUATION_BASE_PORT:-8400}"
HETEROGENEITY_BASE_PORT="${HETEROGENEITY_BASE_PORT:-8600}"
ABLATION_BASE_PORT="${ABLATION_BASE_PORT:-8800}"

# Restrict this list for a partial run, for example:
#   EXPERIMENT_FAMILIES="evaluation heterogeneity"
read -r -a EXPERIMENT_FAMILIES <<< \
  "${EXPERIMENT_FAMILIES:-evaluation heterogeneity ablation}"

VARIANT_NAMES=(hard adaptive_power1)
PROTOTYPE_MODES=(hard adaptive)
SERVER_WEIGHT_POWERS=(0.0 1.0)

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

contains_family() {
  local wanted="$1"
  local family
  for family in "${EXPERIMENT_FAMILIES[@]}"; do
    if [[ "$family" == "$wanted" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -e "$SUITE_DIR" ]]; then
  echo "Refusing to reuse existing suite directory: $SUITE_DIR" >&2
  echo "Set RUN_NAME to a new value and run again." >&2
  exit 1
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer; got '$GPU_ID'." >&2
  exit 1
fi
if [[ "${#GPU_IDS_ARRAY[@]}" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU ID." >&2
  exit 1
fi
for gpu_id in "${GPU_IDS_ARRAY[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS contains invalid ID '$gpu_id'." >&2
    exit 1
  fi
done
# The evaluation and ablation phases are single-GPU; pin them to the first
# entry of GPU_IDS so setting GPU_IDS alone (without also setting GPU_ID)
# consistently steers every phase away from a busy/unwanted GPU, matching
# what the heterogeneity phase already does with the full list.
GPU_ID="${GPU_IDS_ARRAY[0]}"
for family in "${EXPERIMENT_FAMILIES[@]}"; do
  case "$family" in
    evaluation|heterogeneity|ablation) ;;
    *)
      echo "Unknown experiment family '$family'. Valid values: evaluation heterogeneity ablation" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$SUITE_DIR"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

printf '%s\n' \
  "run_name=$SUITE_RUN_NAME" \
  "started_at=$(timestamp)" \
  "experiment_families=${EXPERIMENT_FAMILIES[*]}" \
  "variants=${VARIANT_NAMES[*]}" \
  "hard_prototype_mode=hard" \
  "hard_server_weight_power=0.0" \
  "adaptive_power1_prototype_mode=adaptive" \
  "adaptive_power1_server_weight_power=1.0" \
  "gpu_id=$GPU_ID" \
  "gpu_ids=${GPU_IDS_ARRAY[*]}" \
  "wandb=disabled" \
  "saved_results=CSV metrics and raw process logs" \
  "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
  > "$SUITE_DIR/run_config.txt"

printf 'variant,prototype_mode,server_weight_power\n' > "$SUITE_DIR/variant_design.csv"
for idx in "${!VARIANT_NAMES[@]}"; do
  printf '%s,%s,%s\n' \
    "${VARIANT_NAMES[$idx]}" \
    "${PROTOTYPE_MODES[$idx]}" \
    "${SERVER_WEIGHT_POWERS[$idx]}" \
    >> "$SUITE_DIR/variant_design.csv"
done

run_evaluation_variant() {
  local index="$1"
  local variant="${VARIANT_NAMES[$index]}"
  local mode="${PROTOTYPE_MODES[$index]}"
  local power="${SERVER_WEIGHT_POWERS[$index]}"
  local port="$((EVALUATION_BASE_PORT + index * 100))"

  echo "[$(timestamp)] evaluation: variant=$variant mode=$mode power=$power"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    RESULT_ROOT="$SUITE_DIR/evaluation" \
    RUN_NAME="$variant" \
    METHODS="pwrh" \
    ENABLE_WANDB=0 \
    BASE_PORT="$port" \
    PWRH_PROTOTYPE_MODE="$mode" \
    PWRH_SERVER_WEIGHT_POWER="$power" \
    bash "$SCRIPT_DIR/run_evaluation_experiments.sh"
}

run_heterogeneity_variant() {
  local index="$1"
  local variant="${VARIANT_NAMES[$index]}"
  local mode="${PROTOTYPE_MODES[$index]}"
  local power="${SERVER_WEIGHT_POWERS[$index]}"
  local port="$((HETEROGENEITY_BASE_PORT + index * 100))"

  echo "[$(timestamp)] heterogeneity: variant=$variant mode=$mode power=$power"
  RESULT_ROOT="$SUITE_DIR/model_heterogeneity" \
    RUN_NAME="$variant" \
    METHODS="pwrh" \
    GPU_IDS="${GPU_IDS_ARRAY[*]}" \
    BASE_PORT="$port" \
    PWRH_PROTOTYPE_MODE="$mode" \
    PWRH_SERVER_WEIGHT_POWER="$power" \
    bash "$SCRIPT_DIR/run_model_heterogeneity_sweep.sh"
}

run_ablation_variant() {
  local index="$1"
  local variant="${VARIANT_NAMES[$index]}"
  local mode="${PROTOTYPE_MODES[$index]}"
  local power="${SERVER_WEIGHT_POWERS[$index]}"
  local port="$((ABLATION_BASE_PORT + index * 100))"

  echo "[$(timestamp)] ablation: variant=$variant mode=$mode power=$power"
  RESULT_ROOT="$SUITE_DIR/ablation" \
    RUN_NAME="$variant" \
    GPU_ID="$GPU_ID" \
    BASE_PORT="$port" \
    PWRH_PROTOTYPE_MODE="$mode" \
    PWRH_SERVER_WEIGHT_POWER="$power" \
    bash "$SCRIPT_DIR/run_pwrh_ablation.sh"
}

for index in "${!VARIANT_NAMES[@]}"; do
  if contains_family evaluation; then
    run_evaluation_variant "$index"
  fi
  if contains_family heterogeneity; then
    run_heterogeneity_variant "$index"
  fi
  if contains_family ablation; then
    run_ablation_variant "$index"
  fi
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$SUITE_DIR/run_config.txt"

echo "PWRH hard-vs-adaptive-power1 suite complete."
echo "Results: $SUITE_DIR"
