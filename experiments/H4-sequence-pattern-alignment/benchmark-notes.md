# Flower Benchmark Notes

Use [run_flower_benchmark.sh](/Users/saber/Library/CloudStorage/OneDrive-Personal/Documents/Git/Federated-Time-Series-Forecasting-1/experiments/H4-sequence-pattern-alignment/run_flower_benchmark.sh) to run a small direct comparison between:

- plain masked HeteroFL
- SPA-HFL over the same heterogeneous LSTM setup

Outputs:

- `benchmark_logs/plain_heterofl_server_metrics.csv`
- `benchmark_logs/spa_hfl_server_metrics.csv`
- `benchmark_logs/plain_heterofl_client_12167-0_L1_metrics.csv`
- `benchmark_logs/plain_heterofl_client_12167-2_L2_metrics.csv`
- `benchmark_logs/plain_heterofl_client_12167-1_L3_metrics.csv`
- `benchmark_logs/spa_hfl_client_12167-0_L1_metrics.csv`
- `benchmark_logs/spa_hfl_client_12167-2_L2_metrics.csv`
- `benchmark_logs/spa_hfl_client_12167-1_L3_metrics.csv`

The benchmark uses three clients from the same dataset:

- `12167-0` with local depth 1
- `12167-2` with local depth 2
- `12167-1` with local depth 3
- global depth 3
- 50 federated rounds
- 5 local epochs

Logging notes:

- server CSVs use a fixed schema across both train and validation rows
- each client CSV records one row per round per split for direct per-client analysis
- SPA-only columns remain blank in plain HeteroFL logs so the CSV shape stays stable
