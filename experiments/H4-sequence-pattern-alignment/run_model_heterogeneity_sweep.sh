#!/usr/bin/env bash
set -euo pipefail

# Robustness sweep over the proportion of large (three-layer) clients.
# With 6 clients, MODEL_BETAS=(0 1/3 2/3 1) corresponds exactly to 0, 2, 4,
# and 6 three-layer clients; all remaining clients use one layer.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-10stations-3months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-model_heterogeneity_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
BASE_PORT="${BASE_PORT:-8100}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-3}"

# Total attempts per (beta, method) run, including the first. A run only
# retries after a genuine failure (a client/server that did not exit
# cleanly), not after the benign post-shutdown gRPC teardown crash handled
# in run_method. Set to 1 to disable retries.
RUN_METHOD_MAX_ATTEMPTS="${RUN_METHOD_MAX_ATTEMPTS:-2}"

# Clients are assigned round-robin across these physical GPU IDs. By default,
# all 6 clients share GPU 0.
read -r -a GPU_IDS <<< "${GPU_IDS:-0}"
NUM_GPUS="${#GPU_IDS[@]}"

# This sweep writes local CSV metrics and raw process logs. Keep W&B disabled
# even when the surrounding shell environment has W&B configured.
export WANDB_MODE=disabled
export WANDB_DISABLED=true

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
read -r -a MODEL_BETAS <<< "${MODEL_BETAS:-0 1/3 2/3 1}"

# Fixed order makes assignments reproducible and nested across beta values.
CLIENT_CIDS=(
  12162-0 12163-0 12164-0 12165-0 12166-0
  12167-0
)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

num_three_layer_clients() {
  case "$1" in
    0) echo 0 ;;
    1/3) echo 2 ;;
    2/3) echo 4 ;;
    1) echo 6 ;;
    *)
      echo "Unsupported model beta '$1'. Valid values: 0, 1/3, 2/3, 1" >&2
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
if [[ "$NUM_CLIENTS" -ne 6 ]]; then
  echo "This sweep requires exactly 6 configured clients; found $NUM_CLIENTS." >&2
  exit 1
fi
if [[ "$NUM_GPUS" -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU ID." >&2
  exit 1
fi
for gpu_id in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU ID '$gpu_id'. GPU_IDS must be a space-separated list of non-negative integers." >&2
    exit 1
  fi
done

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
  "gpu_ids=${GPU_IDS[*]}" \
  "gpu_assignment=round-robin by client order" \
  "wandb=disabled" \
  "saved_results=CSV metrics and raw process logs" \
  "clients=${CLIENT_CIDS[*]}" \
  "assignment_rule=first 6*beta clients use 3 layers; remaining clients use 1 layer" \
  "layer_1_model_rate=0.25 for heterogeneous FedAvg and InclusiveFL" \
  "layer_3_model_rate=1.0 for heterogeneous FedAvg and InclusiveFL" \
  "inclusive_transfer_beta=$INCLUSIVE_TRANSFER_BETA" \
  "run_method_max_attempts=$RUN_METHOD_MAX_ATTEMPTS" \
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
CLIENT_LOGS=()

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

# Both server and client metrics CSVs are opened in append mode, so a failed
# attempt's partial rows must be moved out of the way before a retry, or the
# retry's rows would silently mix with stale ones in the same file. Moves
# (not deletes) so a failure can still be inspected after the fact.
archive_failed_method_outputs() {
  local beta_dir="$1"
  local method="$2"
  local attempt="$3"
  local archive_dir="$beta_dir/failed_attempts/${method}_attempt${attempt}"

  mkdir -p "$archive_dir"
  shopt -s nullglob
  local f
  for f in "$beta_dir/metrics/${method}_"* "$beta_dir/raw_logs/${method}_"*; do
    mv "$f" "$archive_dir/"
  done
  shopt -u nullglob
}

run_method() {
  local model_beta="$1"
  local num_three="$2"
  local method="$3"
  local port="$4"
  local beta_label="${model_beta//\//_}"
  local beta_dir="$RUN_DIR/beta_${beta_label}"
  local metrics_dir="$beta_dir/metrics"
  local raw_log_dir="$beta_dir/raw_logs"
  local server_args=()
  local client_args=()

  mkdir -p "$metrics_dir" "$raw_log_dir"

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

  echo "[$(timestamp)] beta=$model_beta ($num_three/$NUM_CLIENTS three-layer), method=$method, port=$port"
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
    local gpu_id="${GPU_IDS[$((idx % NUM_GPUS))]}"

    if (( idx < num_three )); then
      layers=3
      model_rate=1.0
    fi
    if [[ "$method" == "hetero_fedavg" || "$method" == "inclusive_fl" ]]; then
      rate_args=(--model_rate "$model_rate")
    fi

    local client_log="$raw_log_dir/${method}_client_${cid}_L${layers}.log"
    echo "  client=$cid layers=$layers gpu=$gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 "$ROOT_DIR/client-hetero.py" \
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
      --require_cuda \
      --no_save_artifacts \
      "${client_args[@]}" \
      --metrics_log_path "$metrics_dir/${method}_client_${cid}_L${layers}_metrics.csv" \
      > "$client_log" 2>&1 &
    CLIENT_PIDS+=("$!")
    CLIENT_LOGS+=("$client_log")
  done

  # A client process can occasionally SIGABRT during gRPC's C-core thread
  # teardown (a known grpcio race: "Check failed: state_ == FAILED" in
  # thd.h) after it has already finished training and logged its clean
  # disconnect. That crash is cosmetic, but under `set -e` a naive `wait`
  # would otherwise kill the whole multi-hour sweep over it. Only treat a
  # nonzero exit as a real failure if the client's own log shows it did not
  # get to disconnect cleanly.
  local run_failed=0
  for i in "${!CLIENT_PIDS[@]}"; do
    local pid="${CLIENT_PIDS[$i]}"
    local client_log="${CLIENT_LOGS[$i]}"
    local status=0
    wait "$pid" || status=$?
    if [[ "$status" -ne 0 ]]; then
      if grep -q "Disconnect and shut down" "$client_log"; then
        echo "[$(timestamp)] WARNING: client pid $pid exited $status after a clean disconnect (benign post-shutdown crash); log: $client_log" >&2
      else
        echo "[$(timestamp)] ERROR: client pid $pid exited $status before disconnecting cleanly; log: $client_log" >&2
        run_failed=1
      fi
    fi
  done
  CLIENT_PIDS=()
  CLIENT_LOGS=()

  local server_status=0
  wait "$SERVER_PID" || server_status=$?
  SERVER_PID=""
  if [[ "$server_status" -ne 0 ]]; then
    echo "[$(timestamp)] ERROR: server exited $server_status; log: $raw_log_dir/${method}_server.log" >&2
    run_failed=1
  fi

  if [[ "$run_failed" -ne 0 ]]; then
    echo "[$(timestamp)] FAILED beta=$model_beta, method=$method" >&2
    return 1
  fi
  echo "[$(timestamp)] completed beta=$model_beta, method=$method"
}

experiment_index=0
for model_beta in "${MODEL_BETAS[@]}"; do
  num_three="$(num_three_layer_clients "$model_beta")"
  beta_label="${model_beta//\//_}"
  beta_dir="$RUN_DIR/beta_${beta_label}"
  mkdir -p "$beta_dir"

  printf 'cid,local_num_layers,model_rate,gpu_id\n' > "$beta_dir/client_assignment.csv"
  for idx in "${!CLIENT_CIDS[@]}"; do
    layers=1
    model_rate=0.25
    if (( idx < num_three )); then
      layers=3
      model_rate=1.0
    fi
    gpu_id="${GPU_IDS[$((idx % NUM_GPUS))]}"
    printf '%s,%d,%s,%s\n' "${CLIENT_CIDS[$idx]}" "$layers" "$model_rate" "$gpu_id" \
      >> "$beta_dir/client_assignment.csv"
  done

  for method in "${METHODS[@]}"; do
    attempt=1
    while true; do
      # Offset the port per attempt so a retry doesn't race the previous
      # attempt's server/clients for the same socket while they exit.
      port="$((BASE_PORT + experiment_index + (attempt - 1) * 1000))"
      if run_method "$model_beta" "$num_three" "$method" "$port"; then
        break
      fi
      if (( attempt >= RUN_METHOD_MAX_ATTEMPTS )); then
        echo "[$(timestamp)] giving up on beta=$model_beta, method=$method after $attempt attempt(s)" >&2
        exit 1
      fi
      echo "[$(timestamp)] beta=$model_beta, method=$method failed on attempt $attempt; archiving partial output and retrying" >&2
      archive_failed_method_outputs "$beta_dir" "$method" "$attempt"
      attempt=$((attempt + 1))
    done
    experiment_index=$((experiment_index + 1))
  done
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "Model-heterogeneity sweep complete."
echo "Results: $RUN_DIR"
