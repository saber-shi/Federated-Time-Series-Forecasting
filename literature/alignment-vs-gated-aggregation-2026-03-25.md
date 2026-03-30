# Alignment vs Gated Aggregation for Heterogeneous Time-Series FL

Date: 2026-03-25

## Question

If we develop either:

1. sequence-pattern alignment across clients, or
2. a weighted gateway/gating mechanism to aggregate different models,

which is more likely to improve heterogeneous federated learning for time-series prediction?

## Literature Signals

### 1. Sequence-pattern / representation alignment

This direction is strongly supported by recent time-series FL literature.

- **Fed-REACT** groups clients after learning representations and reports strong robustness on heterogeneous time series.
  Link: https://openreview.net/forum?id=c6hGb8IsRN
- **FFTS / Federated Foundation Models on Heterogeneous Time Series** argues that direct cross-domain fusion fails because time-series heterogeneity is too strong, then adds alignment-oriented regularization on both client and server sides.
  Link: https://openreview.net/forum?id=sivA0Hvdvt
- **FedTRL** explicitly enforces domain-invariant and semantically consistent representations and reports gains over centralized and federated baselines in forecasting.
  Link: https://openreview.net/forum?id=Ea00wC36fn
- **FeDaL** learns dataset-agnostic temporal representations by removing local and global bias and evaluates against many baselines.
  Link: https://openreview.net/forum?id=HK6t5x5gJq

### 2. Weighted gateway / gated expert aggregation

This direction is promising, but the strongest evidence is mostly from broader heterogeneous FL or non-federated forecasting.

- **FedJETs** uses a gating function over experts and reports sizable personalization gains in federated settings.
  Link: https://openreview.net/forum?id=hEl2HpiH3g
- **FedEMoE** argues sparse MoE alleviates the loss of personalized knowledge caused by dense aggregation and reports strong results under model heterogeneity.
  Link: https://openreview.net/forum?id=EC1NTRLwfS
- **pFedClub** organizes heterogeneous model blocks and builds personalized candidate models, outperforming baselines.
  Link: https://openreview.net/forum?id=xW6ga9i4eA

For forecasting itself, non-federated time-series literature shows that gated expert models often help:

- **MECATS** uses heterogeneous experts and learns coherent aggregated forecasts.
  Link: https://openreview.net/forum?id=fNCVBsB-N9p
- **MFMformer**, **MoGU**, and related 2025-2026 forecasting papers show that gating/routing helps when temporal patterns differ by frequency, scale, or uncertainty.

## Synthesis

### Which one has stronger evidence?

**Sequence-pattern alignment currently has stronger direct support for heterogeneous time-series federated learning.**

Reason:
- The most relevant time-series FL papers repeatedly solve heterogeneity by aligning representations, removing domain bias, or clustering after representation learning.
- This directly targets the real failure mode in forecasting: clients encode different periodicities, regimes, and exogenous effects into incompatible latent spaces.

### Can a weighted gateway help?

**Yes, but usually as a personalization layer or routing layer, not as a plain replacement for aggregation.**

Reason:
- Gating helps when different clients or sequences really need different experts.
- It is especially attractive if your clients have different recurrent depths or different inductive biases.
- But a gateway alone does not guarantee that expert outputs are comparable or temporally aligned.
- If representations are badly misaligned, the gate may simply learn unstable routing or overfit to client identity.

## Expected Performance in Your Setting

For this repository's setting of federated time-series forecasting with heterogeneous recurrent depths:

### If you choose sequence-pattern alignment

Expected outcome:
- More likely to stabilize heterogeneous training.
- More likely to improve global transfer across clients.
- More likely to help recurrent layers share useful seasonal/trend features.

Best use case:
- Non-IID clients with similar targets but different temporal regimes.

Main risk:
- If alignment is too strong, it can wash out client-specific forecasting structure and reduce personalization.

### If you choose a weighted gateway

Expected outcome:
- More likely to improve personalization than a single masked average.
- More useful when some clients consistently benefit from different submodels or expert combinations.
- More attractive if you eventually allow heterogeneous heads, widths, or expert branches.

Best use case:
- Strong client specialization, multiple temporal regimes, or multi-scale pattern diversity.

Main risk:
- Higher complexity and overfitting.
- Gateway training may be unstable with small local datasets.
- Communication and synchronization design become more complicated in FL.

## Recommendation

If you want the **highest-probability first improvement**, start with:

1. **sequence-pattern / representation alignment**
2. then add a **lightweight gated aggregation or expert router** on top

This ordering is safer because:
- alignment fixes semantic incompatibility first
- gating can then exploit residual specialization instead of compensating for misalignment

## Concrete Research Hypothesis

For this project, a strong hypothesis is:

> Aligning client temporal representations before heterogeneous aggregation will improve stability and forecasting accuracy more reliably than using a weighted gateway alone.

And a stronger second-stage hypothesis is:

> After temporal representation alignment, a gated expert aggregation layer will further improve personalized forecasting under model heterogeneity by routing clients or samples to the most compatible shared submodels.

## Bottom Line

Both ideas are plausible.

- **Alignment** is the more evidence-backed, lower-risk choice for heterogeneous time-series FL.
- **Weighted gating** is the higher-upside but higher-complexity choice, and likely works best after alignment rather than before it.
