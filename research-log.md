# Research Log

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-03-24 | bootstrap | Installed Orchestra AI Research Skills into the Codex custom skills directory and loaded the autoresearch instructions manually in this session. |
| 2 | 2026-03-24 | bootstrap | Scanned the repository README and code layout. Identified the working research theme as federated cellular traffic forecasting with accuracy, energy, and deployment-cost constraints. |
| 3 | 2026-03-24 | bootstrap | Initialized `research-state.yaml`, `findings.md`, `literature/`, `experiments/`, `to_human/`, and `paper/`. Seeded hypotheses around energy-efficient model choice, federated optimizer choice, and operations-aware evaluation. |
| 4 | 2026-03-24 | inner-loop | Implemented a first HeteroFL-style code path with padded masked aggregation for heterogeneous recurrent clients. Added `client-hetero.py` and `server-hetero.py` to support different recurrent depths per client under a shared supernet. |
| 5 | 2026-03-24 | inner-loop | Added `main-hetero.py` as an in-process heterogeneous FL runner aligned with the repository's original `main.py` workflow. It builds heterogeneous recurrent clients with client-specific depth assignments and aggregates them with masked supernet averaging. |
| 6 | 2026-03-25 | outer-loop | Reviewed literature on model-heterogeneous federated learning and time-series FL. Main synthesis: generic HeteroFL-style aggregation addresses compute heterogeneity, but time-series forecasting usually also requires clustering, personalization, or representation alignment because temporal heterogeneity is structurally stronger. |
| 7 | 2026-03-25 | inner-loop | Designed H4 / SPA-HFL: a sequence-pattern alignment algorithm for heterogeneous federated forecasting. Added a protocol specifying local latent projection, temporal pattern summaries, alignment losses, and server-side pattern-memory updates. |
| 8 | 2026-03-29 | inner-loop | Implemented SPA-HFL V1 in the in-process heterogeneous runner. Added recurrent feature-return hooks, a reusable `src/spa_hfl.py` alignment module, and server/client logic in `main-hetero.py` for projector aggregation and centroid updates. |
| 9 | 2026-03-29 | inner-loop | Wired SPA-HFL into the Flower `client-hetero.py` and `server-hetero.py` path. Added benchmark logging to CSV and a small runnable Flower benchmark script comparing plain HeteroFL against SPA-HFL directly. |
