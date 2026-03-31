#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/benchmark_logs"
mkdir -p "$LOG_DIR"

echo "Starting plain HeteroFL benchmark server..."
python3 "$ROOT_DIR/server-hetero.py" \
  --server_address 127.0.0.1:8090 \
  --rounds 50 \
  --min_fit_clients 3 \
  --min_available_clients 3 \
  --metrics_log_path "$LOG_DIR/plain_heterofl_server_metrics.csv" &
SERVER_PID=$!
sleep 3

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8090 \
  --cid 12167-0 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 1 \
  --global_num_layers 3 \
  --epochs 5 \
  --wandb \
  --metrics_log_path "$LOG_DIR/plain_heterofl_client_12167-0_L1_metrics.csv" \
  --batch_size 64 &
CLIENT1_PID=$!

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8090 \
  --cid 12167-1 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 3 \
  --global_num_layers 3 \
  --epochs 5 \
  --wandb \
  --metrics_log_path "$LOG_DIR/plain_heterofl_client_12167-1_L3_metrics.csv" \
  --batch_size 64 &
CLIENT2_PID=$!

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8090 \
  --cid 12167-2 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 2 \
  --global_num_layers 3 \
  --epochs 5 \
  --wandb \
  --metrics_log_path "$LOG_DIR/plain_heterofl_client_12167-2_L2_metrics.csv" \
  --batch_size 64 &
CLIENT3_PID=$!

wait $CLIENT1_PID
wait $CLIENT2_PID
wait $CLIENT3_PID
wait $SERVER_PID

echo "Starting SPA-HFL benchmark server..."
python3 "$ROOT_DIR/server-hetero.py" \
  --server_address 127.0.0.1:8091 \
  --rounds 50 \
  --min_fit_clients 3 \
  --min_available_clients 3 \
  --spa_hfl \
  --align_dim 32 \
  --wandb \
  --metrics_log_path "$LOG_DIR/spa_hfl_server_metrics.csv" &
SERVER_PID=$!
sleep 3

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8091 \
  --cid 12167-0 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 1 \
  --global_num_layers 3 \
  --epochs 5 \
  --batch_size 64 \
  --spa_hfl \
  --align_dim 32 \
  --wandb \
  --metrics_log_path "$LOG_DIR/spa_hfl_client_12167-0_L1_metrics.csv" \
  --lambda_align 0.1 \
  --lambda_cons 0.1 &
CLIENT1_PID=$!

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8091 \
  --cid 12167-1 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 3 \
  --global_num_layers 3 \
  --epochs 5 \
  --batch_size 64 \
  --spa_hfl \
  --align_dim 32 \
  --wandb \
  --metrics_log_path "$LOG_DIR/spa_hfl_client_12167-1_L3_metrics.csv" \
  --lambda_align 0.1 \
  --lambda_cons 0.1 &
CLIENT2_PID=$!

python3 "$ROOT_DIR/client-hetero.py" \
  --server_address 127.0.0.1:8091 \
  --cid 12167-2 \
  --data_path "$ROOT_DIR/dataset/5G-2y-bs12167.csv" \
  --model_name lstm \
  --local_num_layers 2 \
  --global_num_layers 3 \
  --epochs 5 \
  --batch_size 64 \
  --spa_hfl \
  --align_dim 32 \
  --wandb \
  --metrics_log_path "$LOG_DIR/spa_hfl_client_12167-2_L2_metrics.csv" \
  --lambda_align 0.1 \
  --lambda_cons 0.1 &
CLIENT3_PID=$!

wait $CLIENT1_PID
wait $CLIENT2_PID
wait $CLIENT3_PID
wait $SERVER_PID

echo "Benchmark complete."
echo "Plain metrics: $LOG_DIR/plain_heterofl_server_metrics.csv"
echo "SPA-HFL metrics: $LOG_DIR/spa_hfl_server_metrics.csv"
