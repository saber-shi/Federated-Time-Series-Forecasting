#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/benchmark_logs"
DATA_PATH="$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed.csv"
PREDICTION_STEPS=4
PLAIN_PORT=8090
SPA_PORT=8091
mkdir -p "$LOG_DIR"

CLIENT_CIDS=(
  12162-0
  12163-0
  12164-0
  12165-0
  12166-0
  12167-0
)

CLIENT_LAYERS=(
  1
  1
  2
  2
  3
  3
)

# echo "Starting plain HeteroFL benchmark server..."
# python3 "$ROOT_DIR/server-hetero.py" \
#   --server_address 127.0.0.1:"$PLAIN_PORT" \
#   --rounds 50 \
#   --min_fit_clients 6 \
#   --min_evaluate_clients 6 \
#   --min_available_clients 6 \
#   --model_name lstm \
#   --input_dim 9 \
#   --out_dim 4 \
#   --global_num_layers 3 \
#   --metrics_log_path "$LOG_DIR/plain_heterofl_server_metrics.csv" &
# SERVER_PID=$!
# sleep 3

# CLIENT_PIDS=()
# for idx in "${!CLIENT_CIDS[@]}"; do
#   cid="${CLIENT_CIDS[$idx]}"
#   layers="${CLIENT_LAYERS[$idx]}"
#   python3 "$ROOT_DIR/client-hetero.py" \
#     --server_address 127.0.0.1:"$PLAIN_PORT" \
#     --cid "$cid" \
#     --data_path "$DATA_PATH" \
#     --model_name lstm \
#     --prediction_steps "$PREDICTION_STEPS" \
#     --local_num_layers "$layers" \
#     --global_num_layers 3 \
#     --epochs 5 \
#     --wandb \
#     --metrics_log_path "$LOG_DIR/plain_heterofl_client_${cid}_L${layers}_metrics.csv" \
#     --batch_size 64 &
#   CLIENT_PIDS+=($!)
# done

# for pid in "${CLIENT_PIDS[@]}"; do
#   wait "$pid"
# done
# wait $SERVER_PID

echo "Starting SPA-HFL benchmark server..."
python3 "$ROOT_DIR/server-hetero.py" \
  --server_address 127.0.0.1:"$SPA_PORT" \
  --rounds 50 \
  --min_fit_clients 6 \
  --min_evaluate_clients 6 \
  --min_available_clients 6 \
  --model_name lstm \
  --input_dim 9 \
  --out_dim 4 \
  --global_num_layers 3 \
  --spa_hfl \
  --align_dim 32 \
  --pattern_cluster_count 3 \
  --pattern_cluster_iters 10 \
  --wandb \
  --metrics_log_path "$LOG_DIR/spa_hfl_server_metrics.csv" &
SERVER_PID=$!
sleep 3

CLIENT_PIDS=()
for idx in "${!CLIENT_CIDS[@]}"; do
  cid="${CLIENT_CIDS[$idx]}"
  layers="${CLIENT_LAYERS[$idx]}"
  python3 "$ROOT_DIR/client-hetero.py" \
    --server_address 127.0.0.1:"$SPA_PORT" \
    --cid "$cid" \
    --data_path "$DATA_PATH" \
    --model_name lstm \
    --prediction_steps "$PREDICTION_STEPS" \
    --local_num_layers "$layers" \
    --global_num_layers 3 \
    --epochs 5 \
    --batch_size 64 \
    --spa_hfl \
    --align_dim 32 \
    --wandb \
    --metrics_log_path "$LOG_DIR/spa_hfl_client_${cid}_L${layers}_metrics.csv" \
    --lambda_align 0.005 \
    --lambda_cons 0 &
  CLIENT_PIDS+=($!)
done

for pid in "${CLIENT_PIDS[@]}"; do
  wait "$pid"
done
wait $SERVER_PID

echo "Benchmark complete."
echo "Plain metrics: $LOG_DIR/plain_heterofl_server_metrics.csv"
echo "SPA-HFL metrics: $LOG_DIR/spa_hfl_server_metrics.csv"
