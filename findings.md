# Research Findings

## Research Question

How can federated time-series forecasting for 5G base-station traffic improve the accuracy-energy tradeoff under non-IID client distributions?

## Current Understanding

This repository already captures three connected stages of the problem: federated traffic prediction, energy-aware analysis, and bio-inspired model evaluation. That suggests the most promising research path is not a from-scratch model search, but a controlled comparison framework that measures forecast quality together with compute or operational cost.

The literature review on model-heterogeneous federated learning suggests that direct heterogeneous aggregation is a weak fit for time-series forecasting unless it is combined with clustering, personalization, or representation alignment. Generic HeteroFL-style methods solve compute heterogeneity, but forecasting clients also differ in temporal dynamics, exogenous structure, and drift.

Among the two candidate improvements now under consideration, sequence-pattern or representation alignment has the stronger literature support for time-series FL. Gate-weighted aggregation also looks promising, but the evidence suggests it is more effective as a personalization or routing mechanism layered on top of aligned representations, rather than as a standalone fix.

## Key Results

No new experiments have been run in this autoresearch workspace yet. Existing repository artifacts indicate prior baselines for multiple neural architectures and federated aggregation algorithms, plus extensions for energy-aware and bio-inspired evaluation.

A first implementation path now exists for heterogeneous recurrent clients using masked aggregation over a padded supernet representation, which enables experimentation with different LSTM/RNN/GRU depths per client without rewriting the original homogeneous pipeline.

A concrete next-step algorithm is now specified: **SPA-HFL**, which aligns temporal sequence patterns in a shared latent space before masked heterogeneous aggregation. The design predicts that the first benefit will be improved convergence stability and reduced client disparity.

A first runnable SPA-HFL implementation is now integrated into the in-process heterogeneous training path in `main-hetero.py`. The current version uses recurrent hidden-state projection, ACF/FFT pattern summaries, a global latent centroid, and joint forecast-plus-alignment optimization.

## Patterns and Insights

The codebase appears strong on implementation breadth: multiple model families, multiple aggregation rules, notebooks for centralized and federated training, and optimization utilities for battery and renewable scheduling. The missing layer is a structured research loop that records hypotheses, results, and cross-experiment conclusions in one place.

Across the literature, time-series FL papers repeatedly avoid pure one-global-model solutions under strong heterogeneity. Instead, they use cluster-wise training, personalized fine-tuning, or representation sharing. This pattern is evidence that temporal heterogeneity is more structural than the heterogeneity targeted by early model-heterogeneous FL methods.

The newest pattern is that alignment and gating solve different subproblems: alignment improves semantic comparability across clients, while gating improves specialization after shared structure has been made meaningful. This suggests an architecture order: align first, route second.

## Lessons and Constraints

- Existing project structure already has `ml/` and `dataset/`, so the autoresearch workspace should avoid redefining those as canonical code and data roots.
- A useful first experimental baseline should come from reproducing one repository-backed federated setup before comparing new architectures or energy-aware objectives.
- Energy efficiency should be treated as a first-class evaluation dimension, not only a side note after forecasting accuracy.
- Heterogeneous depth alone is unlikely to be sufficient for strong forecasting gains; time-series papers point toward clustering, representation alignment, or personalization as the missing ingredients.
- For recurrent forecasting models, parameter overlap does not guarantee semantic overlap in hidden-state dynamics across clients.
- The implemented V1 uses one shared centroid rather than cluster-wise centroids, so it is best understood as a stabilization baseline before moving to clustered alignment or gated routing.

## Open Questions

- Which metric should anchor the optimization loop: MAE, RMSE, MAPE, energy per training round, or a combined score?
- Which existing notebook or script is the fastest reproducible baseline for a first controlled run?
- How should battery and renewable optimization be coupled to the forecasting pipeline to measure end-to-end system benefit?
- How should HeteroFL aggregation be extended beyond depth heterogeneity to also support width heterogeneity or heterogeneous head layers?
- Should this project pivot from pure HeteroFL toward clustered or personalized heterogeneous FL for time-series forecasting?
- Can we align client representations before aggregation so that recurrent layers with different depths still share temporally meaningful features?
- Is the best next experiment an alignment regularizer on recurrent hidden states, or a lightweight client-level gating network over heterogeneous submodels?
- Which minimal alignment signal should be implemented first in this codebase: hidden-state centroid alignment, ACF/FFT pattern summaries, or contrastive sequence similarity?

## Optimization Trajectory

The autoresearch workspace is initialized, but no new experiment trajectory has been recorded yet.
