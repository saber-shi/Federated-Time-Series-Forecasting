#!/usr/bin/env bash
set -euo pipefail

# Single-factor ablation of the plain-HeteroFL client model_rate (submodel
# width). All 6 clients use the same model_rate within a run, and
# local_num_layers is held fixed across the sweep, so any performance
# difference between runs is attributable to submodel width alone, not
# depth heterogeneity (that axis is already covered by
# run_model_heterogeneity_sweep.sh).

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed-l2-2months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-plain_heterofl_model_rate_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
BASE_PORT="${BASE_PORT:-8400}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-3}"
GPU_ID="${GPU_ID:-0}"

# Total attempts per model_rate value, including the first. A run only
# retries after a genuine failure (a client/server that did not exit
# cleanly), not after the benign post-shutdown gRPC teardown crash handled
# in run_configuration. Set to 1 to disable retries.
RUN_CONFIG_MAX_ATTEMPTS="${RUN_CONFIG_MAX_ATTEMPTS:-2}"

# Held fixed across the sweep so only submodel width (model_rate) varies.
GLOBAL_NUM_LAYERS="${GLOBAL_NUM_LAYERS:-3}"
CLIENT_LOCAL_NUM_LAYERS="${CLIENT_LOCAL_NUM_LAYERS:-3}"

# Override to restrict or extend the sweep. 0.125/0.25/0.5/0.75/1.0 match the
# standard HeteroFL complexity levels.
read -r -a MODEL_RATE_VALUES <<< "${MODEL_RATE_VALUES:-0.125 0.25 0.5 0.75 1.0}"

CLIENT_CIDS=(12162-0 12163-0 12164-0 12165-0 12166-0 12167-0)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

# W&B must remain off; results are written locally as CSV metrics and raw logs.
export WANDB_MODE=disabled
export WANDB_DISABLED=true

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

rate_label() {
  local label="$1"
  label="${label//./p}"
  printf '%s\n' "$label"
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
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer; got '$GPU_ID'." >&2
  exit 1
fi
if [[ "$CLIENT_LOCAL_NUM_LAYERS" -gt "$GLOBAL_NUM_LAYERS" ]]; then
  echo "CLIENT_LOCAL_NUM_LAYERS cannot exceed GLOBAL_NUM_LAYERS." >&2
  exit 1
fi
if [[ "${#MODEL_RATE_VALUES[@]}" -eq 0 ]]; then
  echo "MODEL_RATE_VALUES must contain at least one value." >&2
  exit 1
fi
for rate in "${MODEL_RATE_VALUES[@]}"; do
  if [[ ! "$rate" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
    echo "Model rates must be in (0, 1]; got '$rate'." >&2
    exit 1
  fi
  python3 - "$rate" <<'PY'
import sys
rate = float(sys.argv[1])
if not (0.0 < rate <= 1.0):
    raise SystemExit(f"Model rate {rate} is outside (0, 1].")
PY
done

mkdir -p "$RUN_DIR"

printf '%s\n' \
  "run_name=$RUN_NAME" \
  "started_at=$(timestamp)" \
  "data_path=$DATA_PATH" \
  "ablation=single factor: plain-HeteroFL client model_rate (submodel width)" \
  "model_rate_values=${MODEL_RATE_VALUES[*]}" \
  "global_num_layers=$GLOBAL_NUM_LAYERS" \
  "client_local_num_layers=$CLIENT_LOCAL_NUM_LAYERS" \
  "run_config_max_attempts=$RUN_CONFIG_MAX_ATTEMPTS" \
  "rounds=$ROUNDS" \
  "epochs=$EPOCHS" \
  "batch_size=$BATCH_SIZE" \
  "prediction_steps=$PREDICTION_STEPS" \
  "gpu_id=$GPU_ID" \
  "clients=${CLIENT_CIDS[*]}" \
  "wandb=disabled" \
  "saved_results=CSV metrics and raw process logs" \
  "forecast_accuracy_definition=min(1, max(0, 1 - NRMSE))" \
  "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
  > "$RUN_DIR/run_config.txt"

printf 'run_index,model_rate,port,result_directory\n' > "$RUN_DIR/ablation_design.csv"

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

# Server and client metrics CSVs are opened in append mode, so a failed
# attempt's partial output must be moved out of the way before a retry, or
# the retry's rows would silently mix with stale ones in the same files.
# Each model_rate value owns a dedicated directory, so archiving means
# moving the whole thing aside (not deleting it, so a failure can still be
# inspected after the fact).
archive_failed_variant() {
  local variant_dir="$1"
  local attempt="$2"
  local archive_dir="$RUN_DIR/failed_attempts/$(basename "$variant_dir")_attempt${attempt}"

  if [[ -e "$variant_dir" ]]; then
    mkdir -p "$(dirname "$archive_dir")"
    mv "$variant_dir" "$archive_dir"
  fi
}

run_configuration() {
  local model_rate="$1"
  local port="$2"
  local rate_slug
  rate_slug="$(rate_label "$model_rate")"
  local variant="rate_${rate_slug}"
  local variant_dir="$RUN_DIR/$variant"
  local metrics_dir="$variant_dir/metrics"
  local raw_log_dir="$variant_dir/raw_logs"

  mkdir -p "$metrics_dir" "$raw_log_dir"
  printf '%s\n' \
    "model_rate=$model_rate" \
    "port=$port" \
    > "$variant_dir/config.txt"

  echo "[$(timestamp)] Starting $variant on GPU $GPU_ID, port $port"

  python3 "$ROOT_DIR/server-hetero.py" \
    --server_address "127.0.0.1:$port" \
    --rounds "$ROUNDS" \
    --min_fit_clients "$NUM_CLIENTS" \
    --min_evaluate_clients "$NUM_CLIENTS" \
    --min_available_clients "$NUM_CLIENTS" \
    --model_name lstm \
    --input_dim 9 \
    --out_dim 4 \
    --global_num_layers "$GLOBAL_NUM_LAYERS" \
    --metrics_log_path "$metrics_dir/server_metrics.csv" \
    > "$raw_log_dir/server.log" 2>&1 &
  SERVER_PID=$!
  sleep "$SERVER_STARTUP_SECONDS"

  CLIENT_PIDS=()
  CLIENT_LOGS=()
  for cid in "${CLIENT_CIDS[@]}"; do
    local client_log="$raw_log_dir/client_${cid}.log"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "$ROOT_DIR/client-hetero.py" \
      --server_address "127.0.0.1:$port" \
      --cid "$cid" \
      --data_path "$DATA_PATH" \
      --model_name lstm \
      --prediction_steps "$PREDICTION_STEPS" \
      --local_num_layers "$CLIENT_LOCAL_NUM_LAYERS" \
      --global_num_layers "$GLOBAL_NUM_LAYERS" \
      --model_rate "$model_rate" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --require_cuda \
      --no_save_artifacts \
      --metrics_log_path "$metrics_dir/client_${cid}_metrics.csv" \
      > "$client_log" 2>&1 &
    CLIENT_PIDS+=("$!")
    CLIENT_LOGS+=("$client_log")
  done

  # A client process can occasionally SIGABRT during gRPC's C-core thread
  # teardown (a known grpcio race: "Check failed: state_ == FAILED" in
  # thd.h) after it has already finished training and logged its clean
  # disconnect. That crash is cosmetic, but under `set -e` a naive `wait`
  # would otherwise kill the whole multi-hour ablation over it. Only treat a
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
    echo "[$(timestamp)] ERROR: server exited $server_status; log: $raw_log_dir/server.log" >&2
    run_failed=1
  fi

  if [[ "$run_failed" -ne 0 ]]; then
    echo "[$(timestamp)] FAILED $variant" >&2
    return 1
  fi
  echo "[$(timestamp)] Completed $variant"
}

run_index=0
for model_rate in "${MODEL_RATE_VALUES[@]}"; do
  rate_slug="$(rate_label "$model_rate")"
  variant="rate_${rate_slug}"
  printf '%d,%s,%d,%s\n' \
    "$run_index" "$model_rate" "$((BASE_PORT + run_index))" "$variant" \
    >> "$RUN_DIR/ablation_design.csv"

  attempt=1
  while true; do
    # Offset the port per attempt so a retry doesn't race the previous
    # attempt's server/clients for the same socket while they exit.
    port="$((BASE_PORT + run_index + (attempt - 1) * 1000))"
    if run_configuration "$model_rate" "$port"; then
      break
    fi
    if (( attempt >= RUN_CONFIG_MAX_ATTEMPTS )); then
      echo "[$(timestamp)] giving up on $variant after $attempt attempt(s)" >&2
      exit 1
    fi
    echo "[$(timestamp)] $variant failed on attempt $attempt; archiving partial output and retrying" >&2
    archive_failed_variant "$RUN_DIR/$variant" "$attempt"
    attempt=$((attempt + 1))
  done
  run_index="$((run_index + 1))"
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "Plain-HeteroFL model_rate ablation complete."
echo "Results: $RUN_DIR"
