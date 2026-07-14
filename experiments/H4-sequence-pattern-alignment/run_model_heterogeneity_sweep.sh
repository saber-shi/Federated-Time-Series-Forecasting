#!/usr/bin/env bash
set -euo pipefail

# Robustness sweep over the proportion of large (three-layer) clients.
# With 10 clients, MODEL_BETAS=(0.0 0.2 ... 1.0) corresponds exactly to
# 0, 2, 4, 6, 8, and 10 three-layer clients; all remaining clients use one layer.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-10stations-3months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-model_heterogeneity_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
BASE_PORT="${BASE_PORT:-8100}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-3}"

# InclusiveFL's transfer coefficient is distinct from the model-heterogeneity beta.
INCLUSIVE_TRANSFER_BETA="${INCLUSIVE_TRANSFER_BETA:-0.5}"
FEDPROX_MU="${FEDPROX_MU:-0.01}"
PWRH_NUM_RESIDUAL_HEADS="${PWRH_NUM_RESIDUAL_HEADS:-6}"
PWRH_TEMPERATURE="${PWRH_TEMPERATURE:-0.1}"
PWRH_INIT_SCALE="${PWRH_INIT_SCALE:-0.01}"
PWRH_SERVER_WEIGHT_POWER="${PWRH_SERVER_WEIGHT_POWER:-0.0}"
PWRH_HEAD_SCALE="${PWRH_HEAD_SCALE:-1.0}"

# Both lists may be restricted for smoke tests through environment variables.
read -r -a METHODS <<< "${METHODS:-hetero_fedavg inclusive_fl plain_heterofl fedprox pwrh}"
read -r -a MODEL_BETAS <<< "${MODEL_BETAS:-0.0 0.2 0.4 0.6 0.8 1.0}"

# Fixed order makes assignments reproducible and nested across beta values.
CLIENT_CIDS=(
  12162-0 12163-0 12164-0 12165-0 12166-0
  12167-0 12168-0 12169-0 12170-0 12171-0
)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

num_three_layer_clients() {
  case "$1" in
    0|0.0) echo 0 ;;
    0.2) echo 2 ;;
    0.4) echo 4 ;;
    0.6) echo 6 ;;
    0.8) echo 8 ;;
    1|1.0) echo 10 ;;
    *)
      echo "Unsupported model beta '$1'. Valid values: 0, 0.2, 0.4, 0.6, 0.8, 1" >&2
      return 2
      ;;
  esac
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
if [[ "$NUM_CLIENTS" -ne 10 ]]; then
  echo "This sweep requires exactly 10 configured clients; found $NUM_CLIENTS." >&2
  exit 1
fi

python3 - "$DATA_PATH" "${CLIENT_CIDS[@]}" <<'PY'
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta

data_path, *expected_clients = sys.argv[1:]
timestamps = defaultdict(list)
with open(data_path, newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"District", "time"}
    missing_columns = sorted(required - set(reader.fieldnames or []))
    if missing_columns:
        raise SystemExit(f"Dataset {data_path!r} is missing columns: {missing_columns}")
    for row in reader:
        timestamps[row["District"]].append(datetime.fromisoformat(row["time"]))

missing = sorted(set(expected_clients) - set(timestamps))
if missing:
    raise SystemExit(f"Dataset {data_path!r} is missing configured clients: {missing}")

unexpected = sorted(set(timestamps) - set(expected_clients))
if unexpected:
    raise SystemExit(f"Dataset {data_path!r} contains unexpected clients: {unexpected}")

for cid in expected_clients:
    values = sorted(timestamps[cid])
    if len(values) != 4368:
        raise SystemExit(f"Client {cid} has {len(values)} records; expected 4368 for Jan-Mar 2024.")
    if values[0] != datetime(2024, 1, 1) or values[-1] != datetime(2024, 3, 31, 23, 30):
        raise SystemExit(f"Client {cid} does not cover exactly 2024-01-01 through 2024-03-31.")
    if any(current - previous != timedelta(minutes=30) for previous, current in zip(values, values[1:])):
        raise SystemExit(f"Client {cid} contains a missing, duplicate, or irregular time interval.")
PY

# Validate beta values before creating any result files.
for beta in "${MODEL_BETAS[@]}"; do
  num_three_layer_clients "$beta" >/dev/null
done

mkdir -p "$RUN_DIR"

WANDB_ARGS=()
if [[ "$ENABLE_WANDB" == "1" ]]; then
  WANDB_ARGS=(--wandb)
fi

printf '%s\n' \
  "run_name=$RUN_NAME" \
  "started_at=$(timestamp)" \
  "data_path=$DATA_PATH" \
  "data_months_per_client=3" \
  "methods=${METHODS[*]}" \
  "model_betas=${MODEL_BETAS[*]}" \
  "rounds=$ROUNDS" \
  "epochs=$EPOCHS" \
  "batch_size=$BATCH_SIZE" \
  "prediction_steps=$PREDICTION_STEPS" \
  "clients=${CLIENT_CIDS[*]}" \
  "assignment_rule=first 10*beta clients use 3 layers; remaining clients use 1 layer" \
  "layer_1_model_rate=0.25 for heterogeneous FedAvg and InclusiveFL" \
  "layer_3_model_rate=1.0 for heterogeneous FedAvg and InclusiveFL" \
  "inclusive_transfer_beta=$INCLUSIVE_TRANSFER_BETA" \
  "forecast_accuracy_definition=min(1, max(0, 1 - NRMSE))" \
  "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
  > "$RUN_DIR/run_config.txt"

printf 'model_beta,num_three_layer_clients,num_one_layer_clients\n' > "$RUN_DIR/sweep_design.csv"
for beta in "${MODEL_BETAS[@]}"; do
  num_three="$(num_three_layer_clients "$beta")"
  printf '%s,%d,%d\n' "$beta" "$num_three" "$((NUM_CLIENTS - num_three))" \
    >> "$RUN_DIR/sweep_design.csv"
done

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
  local model_beta="$1"
  local num_three="$2"
  local method="$3"
  local port="$4"
  local beta_dir="$RUN_DIR/beta_${model_beta}"
  local metrics_dir="$beta_dir/metrics"
  local raw_log_dir="$beta_dir/raw_logs"
  local model_save_dir="$beta_dir/models"
  local prediction_save_dir="$beta_dir/predictions"
  local server_args=()
  local client_args=()

  mkdir -p "$metrics_dir" "$raw_log_dir" "$model_save_dir" "$prediction_save_dir"

  case "$method" in
    hetero_fedavg)
      server_args=(--hetero_fedavg)
      client_args=(--hetero_fedavg)
      ;;
    inclusive_fl)
      server_args=(--inclusive_fl --inclusive_beta "$INCLUSIVE_TRANSFER_BETA")
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
        --residual_head_weight_log_path "$metrics_dir/pwrh_residual_head_weights.csv"
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

  echo "[$(timestamp)] beta=$model_beta ($num_three/10 three-layer), method=$method, port=$port"
  echo "  server log: $raw_log_dir/${method}_server.log"

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
    --metrics_log_path "$metrics_dir/${method}_server_metrics.csv" \
    > "$raw_log_dir/${method}_server.log" 2>&1 &
  SERVER_PID=$!
  sleep "$SERVER_STARTUP_SECONDS"

  CLIENT_PIDS=()
  for idx in "${!CLIENT_CIDS[@]}"; do
    local cid="${CLIENT_CIDS[$idx]}"
    local layers=1
    local model_rate=0.25
    local rate_args=()

    if (( idx < num_three )); then
      layers=3
      model_rate=1.0
    fi
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
      --metrics_log_path "$metrics_dir/${method}_client_${cid}_L${layers}_metrics.csv" \
      --model_save_path "$model_save_dir/${method}_{cid}_{model_name}_final.pt" \
      --prediction_save_path "$prediction_save_dir/${method}_{cid}_{model_name}_last_{num_lags}.csv" \
      > "$raw_log_dir/${method}_client_${cid}_L${layers}.log" 2>&1 &
    CLIENT_PIDS+=("$!")
  done

  for pid in "${CLIENT_PIDS[@]}"; do
    wait "$pid"
  done
  CLIENT_PIDS=()
  wait "$SERVER_PID"
  SERVER_PID=""
  echo "[$(timestamp)] completed beta=$model_beta, method=$method"
}

experiment_index=0
for model_beta in "${MODEL_BETAS[@]}"; do
  num_three="$(num_three_layer_clients "$model_beta")"
  beta_dir="$RUN_DIR/beta_${model_beta}"
  mkdir -p "$beta_dir"

  printf 'cid,local_num_layers,model_rate\n' > "$beta_dir/client_assignment.csv"
  for idx in "${!CLIENT_CIDS[@]}"; do
    layers=1
    model_rate=0.25
    if (( idx < num_three )); then
      layers=3
      model_rate=1.0
    fi
    printf '%s,%d,%s\n' "${CLIENT_CIDS[$idx]}" "$layers" "$model_rate" \
      >> "$beta_dir/client_assignment.csv"
  done

  for method in "${METHODS[@]}"; do
    run_method "$model_beta" "$num_three" "$method" "$((BASE_PORT + experiment_index))"
    experiment_index=$((experiment_index + 1))
  done
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "Model-heterogeneity sweep complete."
echo "Results: $RUN_DIR"
