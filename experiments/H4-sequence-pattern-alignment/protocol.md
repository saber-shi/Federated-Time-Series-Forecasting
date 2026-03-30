# H4 Protocol: Sequence-Pattern Alignment for Heterogeneous Federated Forecasting

Date: 2026-03-25

## Hypothesis

If heterogeneous recurrent clients align their sequence-level temporal patterns in a shared latent space before masked HeteroFL aggregation, then federated forecasting performance will improve over plain masked aggregation because overlapping parameters will represent more comparable temporal semantics across clients.

## Motivation

The current heterogeneous setup only aligns models structurally through shared parameter slots. The literature suggests this is not enough for time-series data, because clients can have very different seasonality, trend, volatility, drift, and exogenous dependencies. The core idea is therefore to align **sequence patterns**, not only model shapes.

## Proposed Algorithm

Name: **SPA-HFL** (Sequence-Pattern Alignment for Heterogeneous Federated Learning)

### Local model structure

Each client keeps its own heterogeneous recurrent backbone:

- shallow or deep `LSTM` / `GRU` / `RNN`
- local prediction head

Add two lightweight shared components:

1. **Projection head** `g_i(.)`
   Maps the client's last hidden state or pooled temporal representation into a shared latent dimension `d_align`.

2. **Pattern statistics head**
   Computes compact pattern summaries from each input sequence and latent sequence:
   - trend summary
   - seasonal summary
   - autocorrelation-style lag summary
   - spectral-energy summary or FFT band summary

### Client forward pass

For each batch on client `i`:

1. Encode the input sequence with the local recurrent model.
2. Extract:
   - `h_last`: last hidden state
   - `H_seq`: sequence of hidden states
3. Predict target `y_hat`.
4. Project `h_last` into aligned latent:
   - `z_i = g_i(h_last)`
5. Build pattern descriptors:
   - `p_raw_i` from the input sequence
   - `p_lat_i` from `H_seq` or `z_i`

## Alignment mechanism

The server maintains a **global pattern memory bank** for each round:

- `C_trend`
- `C_season`
- `C_acf`
- `C_spec`
- optional client-cluster centroids instead of one global centroid

Each selected client receives the current centroids and optimizes:

### Local objective

`L_total = L_forecast + lambda_align * L_align + lambda_cons * L_consistency + lambda_sep * L_separation`

where:

1. `L_forecast`
   Standard prediction loss, e.g. MSE or MAE.

2. `L_align`
   Align client latent patterns to the shared pattern centroids.
   Example:
   - cosine distance between `z_i` and the matched centroid
   - MMD / CORAL loss between batch latent distribution and global latent centroid statistics

3. `L_consistency`
   Preserve meaningful correspondence between raw sequence patterns and latent patterns.
   Example:
   - match pairwise similarity matrices built from `p_raw_i` and `p_lat_i`
   - contrastive loss where temporally similar sequences are pulled together

4. `L_separation`
   Prevent collapse by keeping dissimilar temporal regimes apart.
   Example:
   - margin loss between nearest and second-nearest pattern centroids
   - supervised contrastive variant using pseudo-pattern groups

## Server update

After local training, each client uploads:

- heterogeneous model parameters with masks
- batch-aggregated latent statistics:
  - mean and covariance of `z_i`
  - mean trend descriptor
  - mean seasonal descriptor
  - mean spectral descriptor
  - sample count

The server performs two updates:

### 1. Model aggregation

Use the current masked HeteroFL aggregation:

- weighted average only over overlapping parameters
- fallback to previous global parameters for non-overlapping positions

### 2. Pattern memory update

Update global or cluster-wise centroids:

- `C <- momentum * C + (1 - momentum) * weighted_client_stats`

Optional extension:

- run lightweight clustering over uploaded pattern descriptors to create multiple centroids
- assign each client softly to one or more centroids next round

## Why this should help

Plain HeteroFL assumes overlapping parameters already mean overlapping function. In forecasting, that is often false. SPA-HFL changes the optimization target so that shared layers are encouraged to encode similar temporal primitives even when client backbones have different depths.

This should help because:

- shallow and deep clients can still share aligned low-dimensional temporal representations
- aggregation operates on parameters whose semantics are less client-specific
- the method preserves local specialization through local backbone depth and forecasting head

## Minimal practical implementation

To keep the first version simple:

1. Use `h_last` as the aligned representation.
2. Use one linear projection head to dimension `d_align = 32`.
3. Use only two pattern summaries:
   - normalized autocorrelation summary over a few lags
   - low-frequency FFT magnitude summary
4. Use one global centroid instead of clustering.
5. Use:
   - `L_align = 1 - cosine(z_i, C_z)`
   - `L_consistency = mse(sim_raw, sim_lat)`

This gives a low-risk V1.

## Stronger V2

If V1 is stable, upgrade to:

- cluster-wise pattern centroids
- contrastive alignment
- soft client-to-centroid assignment
- gated aggregation after alignment

## Evaluation Plan

Compare:

1. Homogeneous FedAvg baseline
2. HeteroFL masked aggregation baseline
3. SPA-HFL V1
4. SPA-HFL V2 with cluster centroids

Metrics:

- RMSE
- MAE
- NRMSE
- per-client variance in performance
- convergence stability across rounds
- optional energy/runtime overhead

## Success Criterion

SPA-HFL is promising if it:

- improves average forecasting error over plain HeteroFL
- reduces variance across clients
- yields smoother round-to-round convergence
- does not add prohibitive communication overhead

## Failure Modes to Watch

- alignment collapse to trivial latent representations
- over-regularization that hurts client-specific forecasting signals
- unstable centroid updates under very small client batches
- excessive communication cost from uploaded pattern summaries

## Prediction

The first gain will likely come from improved stability and reduced client disparity, before large average-RMSE gains appear. If that happens, it would strongly support the claim that temporal representation mismatch is the main bottleneck in heterogeneous time-series FL.
