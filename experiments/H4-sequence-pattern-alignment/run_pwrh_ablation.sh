#!/usr/bin/env bash
set -euo pipefail

# Full-factorial ablation of the PWRH residual-head count and residual-head
# scale. Each configuration runs the same six-client heterogeneous FL setup as
# run_evaluation_experiments.sh.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed-l2-2months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-pwrh_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
BASE_PORT="${BASE_PORT:-8200}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-3}"
GPU_ID="${GPU_ID:-0}"
RESUME="${RESUME:-0}"

# Total attempts per configuration, including the first. A configuration
# only retries after a genuine failure (a client/server that did not exit
# cleanly), not after the benign post-shutdown gRPC teardown crash handled
# in run_configuration. Set to 1 to disable retries.
RUN_CONFIG_MAX_ATTEMPTS="${RUN_CONFIG_MAX_ATTEMPTS:-2}"

# Override either space-separated list to restrict or extend the sweep.
read -r -a PWRH_NUM_RESIDUAL_HEADS_VALUES <<< \
  "${PWRH_NUM_RESIDUAL_HEADS_VALUES:-1 2 4 6 8}"
read -r -a PWRH_HEAD_SCALE_VALUES <<< \
  "${PWRH_HEAD_SCALE_VALUES:-0.0 0.25 0.5 1.0 2.0}"

# PWRH parameters held fixed during this two-factor ablation.
PWRH_TEMPERATURE="${PWRH_TEMPERATURE:-0.1}"
PWRH_INIT_SCALE="${PWRH_INIT_SCALE:-0.01}"
PWRH_SERVER_WEIGHT_POWER="${PWRH_SERVER_WEIGHT_POWER:-0.0}"
PWRH_PROTOTYPE_MODE="${PWRH_PROTOTYPE_MODE:-adaptive}"
PWRH_PROTOTYPE_SEED="${PWRH_PROTOTYPE_SEED:-0}"

CLIENT_CIDS=(12162-0 12163-0 12164-0 12165-0 12166-0 12167-0)
CLIENT_LAYERS=(1 1 2 2 3 3)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

# W&B must remain off; results are written locally as CSV metrics and raw logs.
export WANDB_MODE=disabled
export WANDB_DISABLED=true

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

scale_label() {
  local label="$1"
  label="${label//./p}"
  label="${label//+}"
  label="${label//-/m}"
  printf '%s\n' "$label"
}

if [[ -e "$RUN_DIR" && "$RESUME" != "1" ]]; then
  echo "Refusing to reuse existing run directory: $RUN_DIR" >&2
  echo "Set RUN_NAME to a new value, or set RESUME=1 to continue it." >&2
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
if [[ "${#PWRH_NUM_RESIDUAL_HEADS_VALUES[@]}" -eq 0 || "${#PWRH_HEAD_SCALE_VALUES[@]}" -eq 0 ]]; then
  echo "Both ablation value lists must contain at least one value." >&2
  exit 1
fi
for num_heads in "${PWRH_NUM_RESIDUAL_HEADS_VALUES[@]}"; do
  if [[ ! "$num_heads" =~ ^[1-9][0-9]*$ ]]; then
    echo "Residual-head counts must be positive integers; got '$num_heads'." >&2
    exit 1
  fi
done
for head_scale in "${PWRH_HEAD_SCALE_VALUES[@]}"; do
  if [[ ! "$head_scale" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
    echo "Head scales must be non-negative numbers; got '$head_scale'." >&2
    exit 1
  fi
done

mkdir -p "$RUN_DIR"

if [[ "$RESUME" == "1" ]]; then
  if [[ ! -f "$RUN_DIR/run_config.txt" ]]; then
    echo "Cannot resume because run_config.txt is missing from $RUN_DIR" >&2
    exit 1
  fi
  printf 'resumed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
  if [[ ! -f "$RUN_DIR/ablation_design.csv" ]]; then
    printf 'run_index,num_residual_heads,head_scale,port,result_directory\n' \
      > "$RUN_DIR/ablation_design.csv"
  fi
else
  printf '%s\n' \
    "run_name=$RUN_NAME" \
    "started_at=$(timestamp)" \
    "data_path=$DATA_PATH" \
    "ablation=full factorial: PWRH_NUM_RESIDUAL_HEADS x PWRH_HEAD_SCALE" \
    "num_residual_heads_values=${PWRH_NUM_RESIDUAL_HEADS_VALUES[*]}" \
    "head_scale_values=${PWRH_HEAD_SCALE_VALUES[*]}" \
    "pwrh_temperature=$PWRH_TEMPERATURE" \
    "pwrh_init_scale=$PWRH_INIT_SCALE" \
    "pwrh_server_weight_power=$PWRH_SERVER_WEIGHT_POWER" \
    "pwrh_prototype_mode=$PWRH_PROTOTYPE_MODE" \
    "pwrh_prototype_seed=$PWRH_PROTOTYPE_SEED" \
    "run_config_max_attempts=$RUN_CONFIG_MAX_ATTEMPTS" \
    "rounds=$ROUNDS" \
    "epochs=$EPOCHS" \
    "batch_size=$BATCH_SIZE" \
    "prediction_steps=$PREDICTION_STEPS" \
    "gpu_id=$GPU_ID" \
    "clients=${CLIENT_CIDS[*]}" \
    "client_layers=${CLIENT_LAYERS[*]}" \
    "wandb=disabled" \
    "saved_results=CSV metrics and raw process logs" \
    "forecast_accuracy_definition=min(1, max(0, 1 - NRMSE))" \
    "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
    > "$RUN_DIR/run_config.txt"

  printf 'run_index,num_residual_heads,head_scale,port,result_directory\n' \
    > "$RUN_DIR/ablation_design.csv"
fi

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
# attempt's partial output must be moved out of the way before a retry, or
# the retry's rows would silently mix with stale ones in the same files.
# Each configuration owns a dedicated directory, so archiving means moving
# the whole thing aside (not deleting it, so a failure can still be
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

# New runs write an explicit marker. For runs created before resume support,
# infer completion by requiring a final-round validation row from the server
# and every configured client.
variant_is_complete() {
  local variant_dir="$1"
  if [[ -f "$variant_dir/COMPLETED" ]]; then
    return 0
  fi

  local metrics_dir="$variant_dir/metrics"
  if [[ ! -f "$metrics_dir/server_metrics.csv" ]] || \
     ! grep -q "^${ROUNDS},val," "$metrics_dir/server_metrics.csv"; then
    return 1
  fi

  local idx cid layers client_metrics
  for idx in "${!CLIENT_CIDS[@]}"; do
    cid="${CLIENT_CIDS[$idx]}"
    layers="${CLIENT_LAYERS[$idx]}"
    client_metrics="$metrics_dir/client_${cid}_L${layers}_metrics.csv"
    if [[ ! -f "$client_metrics" ]] || ! grep -q "^${ROUNDS},val," "$client_metrics"; then
      return 1
    fi
  done
  return 0
}

run_configuration() {
  local num_heads="$1"
  local head_scale="$2"
  local port="$3"
  local scale_slug
  scale_slug="$(scale_label "$head_scale")"
  local variant="heads_${num_heads}_scale_${scale_slug}"
  local variant_dir="$RUN_DIR/$variant"
  local metrics_dir="$variant_dir/metrics"
  local raw_log_dir="$variant_dir/raw_logs"

  mkdir -p "$metrics_dir" "$raw_log_dir"
  printf '%s\n' \
    "num_residual_heads=$num_heads" \
    "head_scale=$head_scale" \
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
    --global_num_layers 3 \
    --pattern_weighted_residual_heads \
    --num_residual_heads "$num_heads" \
    --residual_head_temperature "$PWRH_TEMPERATURE" \
    --residual_head_init_scale "$PWRH_INIT_SCALE" \
    --residual_head_server_weight_power "$PWRH_SERVER_WEIGHT_POWER" \
    --pwrh_prototype_mode "$PWRH_PROTOTYPE_MODE" \
    --pwrh_prototype_seed "$PWRH_PROTOTYPE_SEED" \
    --residual_head_weight_log_path "$metrics_dir/residual_head_weights.csv" \
    --metrics_log_path "$metrics_dir/server_metrics.csv" \
    > "$raw_log_dir/server.log" 2>&1 &
  SERVER_PID=$!
  sleep "$SERVER_STARTUP_SECONDS"

  CLIENT_PIDS=()
  CLIENT_LOGS=()
  for idx in "${!CLIENT_CIDS[@]}"; do
    local cid="${CLIENT_CIDS[$idx]}"
    local layers="${CLIENT_LAYERS[$idx]}"
    local client_log="$raw_log_dir/client_${cid}_L${layers}.log"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python3 "$ROOT_DIR/client-hetero.py" \
      --server_address "127.0.0.1:$port" \
      --cid "$cid" \
      --data_path "$DATA_PATH" \
      --model_name lstm \
      --prediction_steps "$PREDICTION_STEPS" \
      --local_num_layers "$layers" \
      --global_num_layers 3 \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --require_cuda \
      --no_save_artifacts \
      --pattern_weighted_residual_heads \
      --num_residual_heads "$num_heads" \
      --residual_head_temperature "$PWRH_TEMPERATURE" \
      --residual_head_init_scale "$PWRH_INIT_SCALE" \
      --residual_head_scale "$head_scale" \
      --spc_pattern_source y_hist \
      --metrics_log_path "$metrics_dir/client_${cid}_L${layers}_metrics.csv" \
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
  touch "$variant_dir/COMPLETED"
  echo "[$(timestamp)] Completed $variant"
}

run_index=0
for num_heads in "${PWRH_NUM_RESIDUAL_HEADS_VALUES[@]}"; do
  for head_scale in "${PWRH_HEAD_SCALE_VALUES[@]}"; do
    scale_slug="$(scale_label "$head_scale")"
    variant="heads_${num_heads}_scale_${scale_slug}"
    variant_dir="$RUN_DIR/$variant"
    if ! grep -q ",${variant}$" "$RUN_DIR/ablation_design.csv"; then
      printf '%d,%s,%s,%d,%s\n' \
        "$run_index" "$num_heads" "$head_scale" "$((BASE_PORT + run_index))" "$variant" \
        >> "$RUN_DIR/ablation_design.csv"
    fi

    if [[ "$RESUME" == "1" ]] && variant_is_complete "$variant_dir"; then
      echo "[$(timestamp)] SKIP completed $variant"
      run_index="$((run_index + 1))"
      continue
    fi
    if [[ "$RESUME" == "1" && -e "$variant_dir" ]]; then
      echo "[$(timestamp)] Archiving incomplete $variant before resuming"
      archive_failed_variant "$variant_dir" "resume_$(date +%s)"
    fi

    attempt=1
    while true; do
      # Offset the port per attempt so a retry doesn't race the previous
      # attempt's server/clients for the same socket while they exit.
      port="$((BASE_PORT + run_index + (attempt - 1) * 1000))"
      if run_configuration "$num_heads" "$head_scale" "$port"; then
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
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "PWRH ablation complete."
echo "Results: $RUN_DIR"
