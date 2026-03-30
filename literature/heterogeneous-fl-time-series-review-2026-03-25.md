# Literature Review: Model-Heterogeneous FL for Time-Series Forecasting

Date: 2026-03-25

## Question

Why do model-heterogeneous federated learning methods often fail to transfer cleanly to time-series prediction, and what adaptations does the literature use instead?

## Core Papers

### HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients

- Authors: Enmao Diao, Jie Ding, Vahid Tarokh
- Venue: ICLR 2021
- Link: https://openreview.net/forum?id=TNkPBBYFkXg
- Relevance:
  This is the canonical starting point for model-heterogeneous FL. It allows clients with different compute budgets to train subnetworks and aggregate them into one global model.
- Key observation:
  The method is built around shrinkable subnetworks and masked parameter aggregation, with evaluation on MNIST, CIFAR-10, and WikiText-2 rather than forecasting tasks.
- Why it matters here:
  It shows that architecture heterogeneity can be handled when subnetworks are nested cleanly. But that assumption is much easier for width-sliced CNN/RNN language models than for real time-series clients with different dynamics, exogenous signals, horizons, and drift.

### Federated Model Heterogeneous Matryoshka Representation Learning (FedMRL)

- Authors: Liping Yi, Han Yu, Chao Ren, Gang Wang, Xiaoguang Liu, Xiaoxiao Li
- Venue: NeurIPS 2024
- Link: https://openreview.net/forum?id=5yboFMpvHf
- Relevance:
  FedMRL explicitly argues that prior model-heterogeneous FL methods exchange too little knowledge when they rely only on training loss or raw parameter transfer.
- Key observation:
  The paper introduces an auxiliary shared homogeneous model and representation fusion.
- Why it matters here:
  For time series, this supports the idea that raw parameter averaging is usually not enough; temporal representations need alignment before aggregation.

### Fed-REACT: Federated Representation Learning for Heterogeneous Time Series Data

- Authors: Yiyue Chen, Usman Akram, Chianing Wang, Haris Vikalo
- Venue: ICLR 2025 submission
- Link: https://openreview.net/forum?id=c6hGb8IsRN
- Relevance:
  This is a time-series-specific paper that does not solve heterogeneity by direct global averaging alone.
- Key observation:
  The method first learns time-series representations locally and then performs adaptive clustering for downstream task models.
- Why it matters here:
  The paper implicitly shows that a single global model is often too rigid for heterogeneous time-series clients; representation learning plus cluster-wise learning is more stable.

### PA-CFL: Privacy-Adaptive Clustered Federated Learning for Transformer-Based Sales Forecasting on Heterogeneous Retail Data

- Authors: Yunbo Long, Liming Xu, Ge Zheng, Alexandra Brintrup
- Venue: CoRR 2025
- Link: https://openreview.net/forum?id=2R4tSY5Xsb
- Relevance:
  A forecasting-focused paper that addresses heterogeneity through clustering rather than direct model-heterogeneous aggregation.
- Key observation:
  It groups retailers into bubbles before federated training and reports large RMSE/MAE gains over vanilla FL.
- Why it matters here:
  The method suggests that forecasting clients benefit when the federation reduces heterogeneity first, instead of forcing one heterogeneous global model to fit all clients.

### Personalized Federated DARTS for Electricity Load Forecasting of Individual Buildings

- Authors: Dalin Qin, Chenxi Wang, Qingsong Wen, Weiqi Chen, Liang Sun, Yi Wang
- Venue: IEEE Transactions on Smart Grid 2023
- Link: https://openreview.net/forum?id=rnYuh1rhqi
- Relevance:
  This paper directly combines architecture heterogeneity and time-series forecasting.
- Key observation:
  Buildings are grouped by architecture, then a per-cluster federated model is trained, followed by local fine-tuning.
- Why it matters here:
  The paper does not rely on one universal HeteroFL-style global model. It uses clustered architectures and personalization, which is strong evidence that forecasting tasks need stronger locality than generic heterogeneous FL often assumes.

### Federated Foundation Models on Heterogeneous Time Series

- Authors: Shengchao Chen, Guodong Long, Jing Jiang, Chengqi Zhang
- Venue: CoRR 2024
- Link: https://openreview.net/forum?id=sivA0Hvdvt
- Relevance:
  This paper studies heterogeneity across time-series datasets directly.
- Key observation:
  It states that cross-domain time-series fusion is harder than text or image fusion because statistical heterogeneity is much stronger.
- Why it matters here:
  This supports the broader claim that generic FL methods tuned for Euclidean domains break down more easily on time-series data.

### DualEncDecoder: Federated Short-Term Load Forecasting under Heterogeneous Data

- Authors: Sasmita Harini S, Naman Srivastava, Priyanka Nihalchandani, Varun Ojha, Pandarasamy Arjunan
- Venue: FLCA 2026 workshop
- Link: https://openreview.net/forum?id=pdsrRJqNEw
- Relevance:
  A recent forecasting paper that combines a stronger forecasting backbone with adaptive sampling and clustering.
- Key observation:
  Simpler models like LSTM and GRU improve after clustering.
- Why it matters here:
  Again, the literature leans toward partitioning clients by similarity rather than asking one heterogeneous global aggregator to reconcile all temporal variation directly.

## Synthesis

The literature suggests that model-heterogeneous FL struggles on time series for five recurring reasons:

1. **Temporal dynamics are not nested the way subnetworks are nested.**
   HeteroFL assumes smaller client models are compatible subnetworks of a larger global model. In time series, two clients may need different memory depth, recurrence depth, seasonality handling, or exogenous conditioning. Those are not always clean subnetwork slices.

2. **Temporal representation semantics drift across clients.**
   In recurrent or transformer forecasting models, hidden states encode client-specific periodicities, event responses, and lag structure. Averaging parameters from clients with different temporal regimes can mix incompatible state semantics.

3. **Forecasting heterogeneity is often structural, not just statistical.**
   Clients may differ in horizon, sampling rate, covariates, missingness, or target relationships. Many generic model-heterogeneous FL methods assume identical tasks and aligned outputs, while forecasting clients often violate that assumption.

4. **Time-series data drift breaks a static shared model assumption.**
   Temporal distributions evolve over time within the same client. A fixed heterogeneous aggregation rule can already be strained by cross-client differences; adding intra-client drift makes the shared optimum even less stable.

5. **Forecasting quality depends heavily on local inductive bias.**
   Papers like Personalized Federated DARTS and PA-CFL improve results by clustering and personalization. That suggests local architectural bias is valuable and should often be preserved rather than averaged away.

## Practical Takeaway for This Repository

For this project, directly applying HeteroFL with clients that only differ in recurrent depth is a useful first experiment, but the literature suggests it may underperform unless one or more of the following are added:

- client grouping before aggregation
- representation-level alignment instead of parameter-only alignment
- local fine-tuning after aggregation
- explicit handling of temporal drift
- architecture-aware aggregation for recurrent hidden-state semantics

## Bottom Line

Model-heterogeneous FL does not fail on time series because heterogeneity exists; it fails when the method assumes that architectural compatibility is enough. In forecasting, the harder problem is that clients often differ in temporal mechanisms, not only in compute budget. The most successful time-series FL papers therefore add clustering, personalization, representation alignment, or bias-removal modules rather than relying on masked averaging alone.
