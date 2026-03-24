# Research Findings

## Research Question

How can federated time-series forecasting for 5G base-station traffic improve the accuracy-energy tradeoff under non-IID client distributions?

## Current Understanding

This repository already captures three connected stages of the problem: federated traffic prediction, energy-aware analysis, and bio-inspired model evaluation. That suggests the most promising research path is not a from-scratch model search, but a controlled comparison framework that measures forecast quality together with compute or operational cost.

## Key Results

No new experiments have been run in this autoresearch workspace yet. Existing repository artifacts indicate prior baselines for multiple neural architectures and federated aggregation algorithms, plus extensions for energy-aware and bio-inspired evaluation.

A first implementation path now exists for heterogeneous recurrent clients using masked aggregation over a padded supernet representation, which enables experimentation with different LSTM/RNN/GRU depths per client without rewriting the original homogeneous pipeline.

## Patterns and Insights

The codebase appears strong on implementation breadth: multiple model families, multiple aggregation rules, notebooks for centralized and federated training, and optimization utilities for battery and renewable scheduling. The missing layer is a structured research loop that records hypotheses, results, and cross-experiment conclusions in one place.

## Lessons and Constraints

- Existing project structure already has `ml/` and `dataset/`, so the autoresearch workspace should avoid redefining those as canonical code and data roots.
- A useful first experimental baseline should come from reproducing one repository-backed federated setup before comparing new architectures or energy-aware objectives.
- Energy efficiency should be treated as a first-class evaluation dimension, not only a side note after forecasting accuracy.

## Open Questions

- Which metric should anchor the optimization loop: MAE, RMSE, MAPE, energy per training round, or a combined score?
- Which existing notebook or script is the fastest reproducible baseline for a first controlled run?
- How should battery and renewable optimization be coupled to the forecasting pipeline to measure end-to-end system benefit?
- How should HeteroFL aggregation be extended beyond depth heterogeneity to also support width heterogeneity or heterogeneous head layers?

## Optimization Trajectory

The autoresearch workspace is initialized, but no new experiment trajectory has been recorded yet.
