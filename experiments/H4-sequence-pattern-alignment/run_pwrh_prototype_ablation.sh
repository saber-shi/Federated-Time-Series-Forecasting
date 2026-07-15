#!/usr/bin/env bash
set -euo pipefail

# Capacity-matched prototype-routing ablation for PWRH. All modes use the same
# residual heads and six-client model heterogeneity; only prototype routing and
# prototype updates differ.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed-l2-2months.csv}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/evaluation_results}"
RUN_NAME="${RUN_NAME:-pwrh_prototype_ablation_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"

ROUNDS="${ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PREDICTION_STEPS="${PREDICTION_STEPS:-4}"
BASE_PORT="${BASE_PORT:-8300}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-3}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"

# Total attempts per prototype mode, including the first. A mode only
# retries after a genuine failure (a client/server that did not exit
# cleanly), not after the benign post-shutdown gRPC teardown crash handled
# in run_mode. Set to 1 to disable retries.
RUN_MODE_MAX_ATTEMPTS="${RUN_MODE_MAX_ATTEMPTS:-2}"

# Full PWRH settings from run_evaluation_experiments.sh, held fixed so the
# prototype mechanism is the only changing factor.
PWRH_NUM_RESIDUAL_HEADS="${PWRH_NUM_RESIDUAL_HEADS:-6}"
PWRH_TEMPERATURE="${PWRH_TEMPERATURE:-0.1}"
PWRH_INIT_SCALE="${PWRH_INIT_SCALE:-0.01}"
PWRH_SERVER_WEIGHT_POWER="${PWRH_SERVER_WEIGHT_POWER:-0.0}"
PWRH_HEAD_SCALE="${PWRH_HEAD_SCALE:-1.0}"

# adaptive: current PWRH; uniform: no prototype routing; frozen: initialize
# once; random: fixed random prototypes; hard: nearest adaptive prototype.
read -r -a PROTOTYPE_MODES <<< \
  "${PROTOTYPE_MODES:-adaptive uniform frozen random hard}"

CLIENT_CIDS=(12162-0 12163-0 12164-0 12165-0 12166-0 12167-0)
CLIENT_LAYERS=(1 1 2 2 3 3)
NUM_CLIENTS="${#CLIENT_CIDS[@]}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true

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
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer; got '$GPU_ID'." >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer; got '$SEED'." >&2
  exit 1
fi
if [[ ! "$PWRH_NUM_RESIDUAL_HEADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "PWRH_NUM_RESIDUAL_HEADS must be a positive integer." >&2
  exit 1
fi
if [[ "${#PROTOTYPE_MODES[@]}" -eq 0 ]]; then
  echo "PROTOTYPE_MODES must contain at least one mode." >&2
  exit 1
fi
for mode in "${PROTOTYPE_MODES[@]}"; do
  case "$mode" in
    adaptive|uniform|frozen|random|hard) ;;
    *)
      echo "Unknown prototype mode '$mode'. Valid modes: adaptive uniform frozen random hard" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RUN_DIR"

printf '%s\n' \
  "run_name=$RUN_NAME" \
  "started_at=$(timestamp)" \
  "data_path=$DATA_PATH" \
  "prototype_modes=${PROTOTYPE_MODES[*]}" \
  "pwrh_num_residual_heads=$PWRH_NUM_RESIDUAL_HEADS" \
  "pwrh_temperature=$PWRH_TEMPERATURE" \
  "pwrh_init_scale=$PWRH_INIT_SCALE" \
  "pwrh_server_weight_power=$PWRH_SERVER_WEIGHT_POWER" \
  "pwrh_head_scale=$PWRH_HEAD_SCALE" \
  "run_mode_max_attempts=$RUN_MODE_MAX_ATTEMPTS" \
  "rounds=$ROUNDS" \
  "epochs=$EPOCHS" \
  "batch_size=$BATCH_SIZE" \
  "prediction_steps=$PREDICTION_STEPS" \
  "gpu_id=$GPU_ID" \
  "seed=$SEED" \
  "clients=${CLIENT_CIDS[*]}" \
  "client_layers=${CLIENT_LAYERS[*]}" \
  "wandb=disabled" \
  "saved_results=CSV metrics and raw process logs" \
  "forecast_accuracy_definition=min(1, max(0, 1 - NRMSE))" \
  "git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)" \
  > "$RUN_DIR/run_config.txt"

printf 'run_index,prototype_mode,port,result_directory\n' > "$RUN_DIR/ablation_design.csv"

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
# Each mode owns a dedicated directory, so archiving means moving the whole
# thing aside (not deleting it, so a failure can still be inspected after
# the fact).
archive_failed_variant() {
  local variant_dir="$1"
  local attempt="$2"
  local archive_dir="$RUN_DIR/failed_attempts/$(basename "$variant_dir")_attempt${attempt}"

  if [[ -e "$variant_dir" ]]; then
    mkdir -p "$(dirname "$archive_dir")"
    mv "$variant_dir" "$archive_dir"
  fi
}

run_mode() {
  local mode="$1"
  local port="$2"
  local variant_dir="$RUN_DIR/$mode"
  local metrics_dir="$variant_dir/metrics"
  local raw_log_dir="$variant_dir/raw_logs"

  mkdir -p "$metrics_dir" "$raw_log_dir"

  echo "[$(timestamp)] Starting prototype mode=$mode on GPU $GPU_ID, port $port"
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
    --seed "$SEED" \
    --pattern_weighted_residual_heads \
    --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS" \
    --residual_head_temperature "$PWRH_TEMPERATURE" \
    --residual_head_init_scale "$PWRH_INIT_SCALE" \
    --residual_head_server_weight_power "$PWRH_SERVER_WEIGHT_POWER" \
    --pwrh_prototype_mode "$mode" \
    --pwrh_prototype_seed "$SEED" \
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
      --seed "$SEED" \
      --require_cuda \
      --no_save_artifacts \
      --pattern_weighted_residual_heads \
      --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS" \
      --residual_head_temperature "$PWRH_TEMPERATURE" \
      --residual_head_init_scale "$PWRH_INIT_SCALE" \
      --residual_head_scale "$PWRH_HEAD_SCALE" \
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
    echo "[$(timestamp)] FAILED prototype mode=$mode" >&2
    return 1
  fi
  echo "[$(timestamp)] Completed prototype mode=$mode"
}

for idx in "${!PROTOTYPE_MODES[@]}"; do
  mode="${PROTOTYPE_MODES[$idx]}"
  printf '%d,%s,%d,%s\n' "$idx" "$mode" "$((BASE_PORT + idx))" "$mode" \
    >> "$RUN_DIR/ablation_design.csv"

  attempt=1
  while true; do
    # Offset the port per attempt so a retry doesn't race the previous
    # attempt's server/clients for the same socket while they exit.
    port="$((BASE_PORT + idx + (attempt - 1) * 1000))"
    if run_mode "$mode" "$port"; then
      break
    fi
    if (( attempt >= RUN_MODE_MAX_ATTEMPTS )); then
      echo "[$(timestamp)] giving up on prototype mode=$mode after $attempt attempt(s)" >&2
      exit 1
    fi
    echo "[$(timestamp)] prototype mode=$mode failed on attempt $attempt; archiving partial output and retrying" >&2
    archive_failed_variant "$RUN_DIR/$mode" "$attempt"
    attempt=$((attempt + 1))
  done
done

printf 'completed_at=%s\n' "$(timestamp)" >> "$RUN_DIR/run_config.txt"
trap - EXIT INT TERM

echo "PWRH prototype ablation complete."
echo "Results: $RUN_DIR"
