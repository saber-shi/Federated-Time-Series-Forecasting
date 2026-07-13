#!/usr/bin/env bash
set -euo pipefail

# Paper-evaluation runner derived from run_flower_benchmark_l2_2months.sh.
# Every invocation writes to a fresh directory so results are never appended to
# or mixed with an earlier experiment.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed-l2-2months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"
METRICS_DIR="$RUN_DIR/metrics"
RAW_LOG_DIR="$RUN_DIR/raw_logs"
MODEL_SAVE_DIR="$RUN_DIR/models"
PREDICTION_SAVE_DIR="$RUN_DIR/predictions"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
BASE_PORT="${BASE_PORT:-8088}"
INCLUSIVE_BETA="${INCLUSIVE_BETA:-0.5}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
PWRH_NUM_RESIDUAL_HEADS="${PWRH_NUM_RESIDUAL_HEADS:-6}"
PWRH_TEMPERATURE="${PWRH_TEMPERATURE:-0.1}"
PWRH_INIT_SCALE="${PWRH_INIT_SCALE:-0.01}"
PWRH_SERVER_WEIGHT_POWER="${PWRH_SERVER_WEIGHT_POWER:-0.0}"
PWRH_HEAD_SCALE="${PWRH_HEAD_SCALE:-1.0}"

# Override with, for example: METHODS="plain_heterofl fedprox"
read -r -a METHODS <<< "${METHODS:-hetero_fedavg inclusive_fl plain_heterofl fedprox pwrh}"

CLIENT_CIDS=(12162-0 12163-0 12164-0 12165-0 12166-0 12167-0)
CLIENT_LAYERS=(1 1 2 2 3 3)
CLIENT_MODEL_RATES=(0.25 0.25 0.50 0.50 1.00 1.00)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to reuse existing run directory: $RUN_DIR" >&2
  echo "Set RUN_NAME to a new value and run again." >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "Dataset not found: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$METRICS_DIR" "$RAW_LOG_DIR" "$MODEL_SAVE_DIR" "$PREDICTION_SAVE_DIR"

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb)
fi

printf '%s\n' \
  "run_name=$RUN_NAME" \
  "started_at=$(timestamp)" \
  "data_path=$DATA_PATH" \
  "methods=${METHODS[*]}" \
  "rounds=$ROUNDS" \
  "epochs=$EPOCHS" \
  "batch_size=$BATCH_SIZE" \
  "prediction_steps=$PREDICTION_STEPS" \
  "clients=${CLIENT_CIDS[*]}" \
  "client_layers=${CLIENT_LAYERS[*]}" \
  "client_model_rates=${CLIENT_MODEL_RATES[*]}" \
  "forecast_accuracy_definition=min(1, max(0, 1 - NRMSE))" \
  "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
  > "$RUN_DIR/run_config.txt"

SERVER_PID=""
CLIENT_PIDS=()

cleanup() {
  local exit_code=$?
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  for pid in "${CLIENT_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

run_method() {
  local method="$1"
  local port="$2"
  local server_args=()
  local client_args=()

  case "$method" in
    hetero_fedavg)
      server_args=(--hetero_fedavg)
      client_args=(--hetero_fedavg)
      ;;
    inclusive_fl)
      server_args=(--inclusive_fl --inclusive_beta "$INCLUSIVE_BETA")
      client_args=(--inclusive_fl)
      ;;
    plain_heterofl)
      ;;
    fedprox)
      client_args=(--fedprox_mu "$FEDPROX_MU")
      ;;
    pwrh)
      server_args=(
        --pattern_weighted_residual_heads
        --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS"
        --residual_head_temperature "$PWRH_TEMPERATURE"
        --residual_head_init_scale "$PWRH_INIT_SCALE"
        --residual_head_server_weight_power "$PWRH_SERVER_WEIGHT_POWER"
        --residual_head_weight_log_path "$METRICS_DIR/pwrh_residual_head_weights.csv"
      )
      client_args=(
        --pattern_weighted_residual_heads
        --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS"
        --residual_head_temperature "$PWRH_TEMPERATURE"
        --residual_head_init_scale "$PWRH_INIT_SCALE"
        --residual_head_scale "$PWRH_HEAD_SCALE"
        --spc_pattern_source y_hist
      )
      ;;
    *)
      echo "Unknown method '$method'. Valid methods: hetero_fedavg inclusive_fl plain_heterofl fedprox pwrh" >&2
      return 2
      ;;
  esac

  echo "[$(timestamp)] Starting $method on port $port"
  python3 "$ROOT_DIR/server-hetero.py" \
    --server_address "127.0.0.1:$port" \
    --rounds "$ROUNDS" \
    --min_fit_clients "$NUM_CLIENTS" \
    --min_evaluate_clients "$NUM_CLIENTS" \
    --min_available_clients "$NUM_CLIENTS" \
    --model_name lstm \
    --input_dim 9 \
    --out_dim 4 \
    --global_num_layers 3 \
    "${server_args[@]}" \
    "${WANDB_ARGS[@]}" \
    --metrics_log_path "$METRICS_DIR/${method}_server_metrics.csv" \
    > "$RAW_LOG_DIR/${method}_server.log" 2>&1 &
  SERVER_PID=$!
  sleep 3

  CLIENT_PIDS=()
  for idx in "${!CLIENT_CIDS[@]}"; do
    local cid="${CLIENT_CIDS[$idx]}"
    local layers="${CLIENT_LAYERS[$idx]}"
    local model_rate="${CLIENT_MODEL_RATES[$idx]}"
    local rate_args=()
    if [[ "$method" == "hetero_fedavg" || "$method" == "inclusive_fl" ]]; then
      rate_args=(--model_rate "$model_rate")
    fi

    python3 "$ROOT_DIR/client-hetero.py" \
      --server_address "127.0.0.1:$port" \
      --cid "$cid" \
      --data_path "$DATA_PATH" \
      --model_name lstm \
      --prediction_steps "$PREDICTION_STEPS" \
      --local_num_layers "$layers" \
      --global_num_layers 3 \
      "${rate_args[@]}" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      "${client_args[@]}" \
      "${WANDB_ARGS[@]}" \
      --metrics_log_path "$METRICS_DIR/${method}_client_${cid}_L${layers}_metrics.csv" \
      --model_save_path "$MODEL_SAVE_DIR/${method}_{cid}_{model_name}_final.pt" \
      --prediction_save_path "$PREDICTION_SAVE_DIR/${method}_{cid}_{model_name}_last_{num_lags}.csv" \
      > "$RAW_LOG_DIR/${method}_client_${cid}_L${layers}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
  done

  for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid"
  done
  CLIENT_PIDS=()
  wait "$SERVER_PID"
  SERVER_PID=""
  echo "[$(timestamp)] Completed $method"
}

for idx in "${!METHODS[@]}"; do
  run_method "${METHODS[$idx]}" "$((BASE_PORT + idx))"
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "Evaluation experiments complete."
echo "Results: $RUN_DIR"
echo "Per-round CSV metrics: $METRICS_DIR"
