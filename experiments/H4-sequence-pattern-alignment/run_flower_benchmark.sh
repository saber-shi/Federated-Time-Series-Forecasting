#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/benchmark_logs"
DATA_PATH="$ROOT_DIR/dataset/5G-2y-firstcell-6stations-medium-mixed.csv"
PREDICTION_STEPS=4
PLAIN_PORT=8090
SPA_PORT=8091
SPC_BASE_PORT=8092
SPC_CLUSTER_COUNTS=(2 4)
PWRH_PORT=8096
PWRH_NUM_RESIDUAL_HEADS=6
PWRH_TEMPERATURE=0.1
PWRH_INIT_SCALE=0.01
PWRH_SERVER_WEIGHT_POWER=0.0
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

echo "Starting plain HeteroFL benchmark server..."
python3 "$ROOT_DIR/server-hetero.py" \
  --server_address 127.0.0.1:"$PLAIN_PORT" \
  --rounds 50 \
  --min_fit_clients 6 \
  --min_evaluate_clients 6 \
  --min_available_clients 6 \
  --model_name lstm \
  --input_dim 9 \
  --out_dim 4 \
  --global_num_layers 3 \
  --metrics_log_path "$LOG_DIR/plain_heterofl_server_metrics.csv" &
SERVER_PID=$!
sleep 3

CLIENT_PIDS=()
for idx in "${!CLIENT_CIDS[@]}"; do
  cid="${CLIENT_CIDS[$idx]}"
  layers="${CLIENT_LAYERS[$idx]}"
  python3 "$ROOT_DIR/client-hetero.py" \
    --server_address 127.0.0.1:"$PLAIN_PORT" \
    --cid "$cid" \
    --data_path "$DATA_PATH" \
    --model_name lstm \
    --prediction_steps "$PREDICTION_STEPS" \
    --local_num_layers "$layers" \
    --global_num_layers 3 \
    --epochs 5 \
    --wandb \
    --metrics_log_path "$LOG_DIR/plain_heterofl_client_${cid}_L${layers}_metrics.csv" \
    --batch_size 64 &
  CLIENT_PIDS+=($!)
done

for pid in "${CLIENT_PIDS[@]}"; do
  wait "$pid"
done
wait $SERVER_PID

# echo "Starting SPA-HFL benchmark server..."
# python3 "$ROOT_DIR/server-hetero.py" \
#   --server_address 127.0.0.1:"$SPA_PORT" \
#   --rounds 50 \
#   --min_fit_clients 6 \
#   --min_evaluate_clients 6 \
#   --min_available_clients 6 \
#   --model_name lstm \
#   --input_dim 9 \
#   --out_dim 4 \
#   --global_num_layers 3 \
#   --spa_hfl \
#   --align_dim 32 \
#   --pattern_cluster_count 3 \
#   --pattern_cluster_iters 10 \
#   --wandb \
#   --metrics_log_path "$LOG_DIR/spa_hfl_server_metrics.csv" &
# SERVER_PID=$!
# sleep 3

# CLIENT_PIDS=()
# for idx in "${!CLIENT_CIDS[@]}"; do
#   cid="${CLIENT_CIDS[$idx]}"
#   layers="${CLIENT_LAYERS[$idx]}"
#   python3 "$ROOT_DIR/client-hetero.py" \
#     --server_address 127.0.0.1:"$SPA_PORT" \
#     --cid "$cid" \
#     --data_path "$DATA_PATH" \
#     --model_name lstm \
#     --prediction_steps "$PREDICTION_STEPS" \
#     --local_num_layers "$layers" \
#     --global_num_layers 3 \
#     --epochs 5 \
#     --batch_size 64 \
#     --spa_hfl \
#     --align_dim 32 \
#     --wandb \
#     --metrics_log_path "$LOG_DIR/spa_hfl_client_${cid}_L${layers}_metrics.csv" \
#     --lambda_align 0.005 \
#     --lambda_cons 0 &
#   CLIENT_PIDS+=($!)
# done

# for pid in "${CLIENT_PIDS[@]}"; do
#   wait "$pid"
# done
# wait $SERVER_PID

# for cluster_idx in "${!SPC_CLUSTER_COUNTS[@]}"; do
#   SPC_CLUSTER_COUNT="${SPC_CLUSTER_COUNTS[$cluster_idx]}"
#   SPC_PORT=$((SPC_BASE_PORT + cluster_idx))

#   echo "Starting SPC-HeteroFL benchmark server with ${SPC_CLUSTER_COUNT} clusters..."
#   python3 "$ROOT_DIR/server-hetero.py" \
#     --server_address 127.0.0.1:"$SPC_PORT" \
#     --rounds 50 \
#     --min_fit_clients 6 \
#     --min_evaluate_clients 6 \
#     --min_available_clients 6 \
#     --model_name lstm \
#     --input_dim 9 \
#     --out_dim 4 \
#     --global_num_layers 3 \
#     --spc \
#     --spc_cluster_count "$SPC_CLUSTER_COUNT" \
#     --spc_cluster_iters 10 \
#     --wandb \
#     --metrics_log_path "$LOG_DIR/spc_k${SPC_CLUSTER_COUNT}_server_metrics.csv" \
#     --spc_assignment_log_path "$LOG_DIR/spc_k${SPC_CLUSTER_COUNT}_assignments.csv" &
#   SERVER_PID=$!
#   sleep 3

#   CLIENT_PIDS=()
#   for idx in "${!CLIENT_CIDS[@]}"; do
#     cid="${CLIENT_CIDS[$idx]}"
#     layers="${CLIENT_LAYERS[$idx]}"
#     python3 "$ROOT_DIR/client-hetero.py" \
#       --server_address 127.0.0.1:"$SPC_PORT" \
#       --cid "$cid" \
#       --data_path "$DATA_PATH" \
#       --model_name lstm \
#       --prediction_steps "$PREDICTION_STEPS" \
#       --local_num_layers "$layers" \
#       --global_num_layers 3 \
#       --epochs 5 \
#       --batch_size 64 \
#       --spc \
#       --spc_cluster_count "$SPC_CLUSTER_COUNT" \
#       --spc_pattern_source y_hist \
#       --wandb \
#       --metrics_log_path "$LOG_DIR/spc_k${SPC_CLUSTER_COUNT}_client_${cid}_L${layers}_metrics.csv" &
#     CLIENT_PIDS+=($!)
#   done

#   for pid in "${CLIENT_PIDS[@]}"; do
#     wait "$pid"
#   done
#   wait $SERVER_PID
# done


echo "Starting Pattern-Weighted Residual Heads benchmark server..."
python3 "$ROOT_DIR/server-hetero.py" \
  --server_address 127.0.0.1:"$PWRH_PORT" \
  --rounds 50 \
  --min_fit_clients 6 \
  --min_evaluate_clients 6 \
  --min_available_clients 6 \
  --model_name lstm \
  --input_dim 9 \
  --out_dim 4 \
  --global_num_layers 3 \
  --pattern_weighted_residual_heads \
  --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS" \
  --residual_head_temperature "$PWRH_TEMPERATURE" \
  --residual_head_init_scale "$PWRH_INIT_SCALE" \
  --residual_head_server_weight_power "$PWRH_SERVER_WEIGHT_POWER" \
  --wandb \
  --metrics_log_path "$LOG_DIR/pwrh_server_metrics.csv" \
  --residual_head_weight_log_path "$LOG_DIR/pwrh_residual_head_weights.csv" &
SERVER_PID=$!
sleep 3

CLIENT_PIDS=()
for idx in "${!CLIENT_CIDS[@]}"; do
  cid="${CLIENT_CIDS[$idx]}"
  layers="${CLIENT_LAYERS[$idx]}"
  python3 "$ROOT_DIR/client-hetero.py" \
    --server_address 127.0.0.1:"$PWRH_PORT" \
    --cid "$cid" \
    --data_path "$DATA_PATH" \
    --model_name lstm \
    --prediction_steps "$PREDICTION_STEPS" \
    --local_num_layers "$layers" \
    --global_num_layers 3 \
    --epochs 5 \
    --batch_size 64 \
    --pattern_weighted_residual_heads \
    --num_residual_heads "$PWRH_NUM_RESIDUAL_HEADS" \
    --residual_head_temperature "$PWRH_TEMPERATURE" \
    --residual_head_init_scale "$PWRH_INIT_SCALE" \
    --spc_pattern_source y_hist \
    --wandb \
    --metrics_log_path "$LOG_DIR/pwrh_client_${cid}_L${layers}_metrics.csv" &
  CLIENT_PIDS+=($!)
done

for pid in "${CLIENT_PIDS[@]}"; do
  wait "$pid"
done
wait $SERVER_PID

echo "Benchmark complete."
echo "Plain metrics: $LOG_DIR/plain_heterofl_server_metrics.csv"
# echo "SPA-HFL metrics: $LOG_DIR/spa_hfl_server_metrics.csv"
# for SPC_CLUSTER_COUNT in "${SPC_CLUSTER_COUNTS[@]}"; do
#   echo "SPC-HeteroFL k=${SPC_CLUSTER_COUNT} metrics: $LOG_DIR/spc_k${SPC_CLUSTER_COUNT}_server_metrics.csv"
#   echo "SPC-HeteroFL k=${SPC_CLUSTER_COUNT} assignments: $LOG_DIR/spc_k${SPC_CLUSTER_COUNT}_assignments.csv"
# done
echo "PWRH metrics: $LOG_DIR/pwrh_server_metrics.csv"
echo "PWRH residual head weights: $LOG_DIR/pwrh_residual_head_weights.csv"
