# Flower Benchmark Notes

Use [run_flower_benchmark.sh](/Users/saber/Library/CloudStorage/OneDrive-Personal/Documents/Git/Federated-Time-Series-Forecasting-1/experiments/H4-sequence-pattern-alignment/run_flower_benchmark.sh) to run a small direct comparison between:

- plain masked HeteroFL
- SPA-HFL over the same heterogeneous LSTM setup

Outputs:

- `benchmark_logs/plain_heterofl_server_metrics.csv`
- `benchmark_logs/spa_hfl_server_metrics.csv`

The benchmark is intentionally small:

- 3 rounds
- 2 clients
- 2 local epochs
- layer depths 1 and 3

This is meant as a smoke test and relative comparison, not a final research result.
