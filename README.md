# Advanced Online User Vector Updates for Recommendation Systems

**From Basic EMA to Production-Ready Online Learning + Short-Form Video Ranking, Observability & Offline Evaluation**

This repository provides a practical, end-to-end guide for building real-time user vector pipelines and short-form video recommendation systems.

## Contents

| Document | Description |
|----------|-------------|
| [README.md](README.md) (this file) | Core concepts: dual-vector models, online learning, Lambda Architecture |
| [PHASE_2_IMPLEMENTATION.md](PHASE_2_IMPLEMENTATION.md) | Short-form video ranking features, Flink dual-vector operator, Redis merge, Greedy MMR diversity |
| [PHASE_3_OBSERVABILITY.md](PHASE_3_OBSERVABILITY.md) | Latency budgets, Prometheus metrics, Grafana dashboards, critical alerting rules |
| [PHASE_4_OFFLINE_EVALUATION.md](PHASE_4_OFFLINE_EVALUATION.md) | Offline replay, nDCG / ILD metrics, and safe simulation of ranking & diversity changes |

---

## 1. Advanced Online Vector Update Techniques

While EMA is lightweight and easy to implement, it has two important limitations:

- It treats all embedding dimensions uniformly.
- It is slow to adapt to abrupt **concept drift** (e.g., a user shopping for running shoes who suddenly switches to kitchen appliances).

### A. Session-Aware / Dual-Vector Models (Short-Term vs. Long-Term)

High-scale platforms typically maintain **two distinct representations** instead of a single monolithic user vector:

| Vector | Computation | Purpose |
|--------|-------------|---------|
| **Long-Term Profile** \(V_{\text{long}}\) | Overnight batch jobs (matrix factorization, deep graph neural networks, etc.) | Captures stable, baseline preferences |
| **Short-Term Session** \(V_{\text{short}}\) | Real-time streaming (Apache Flink) over the last *N* actions or current session | Captures immediate intent |

**Dynamic Gating** produces the final user representation:

$$
V_{\text{user}} = g \cdot V_{\text{short}} + (1 - g) \cdot V_{\text{long}}
$$

where the gating scalar \(g \in [0,1]\) is dynamically adjusted based on:

- Session length
- Time since last activity (inactivity gaps)
- Confidence / volume of recent interactions

### B. Online Matrix Factorization / Incremental Metric Learning

Instead of simply adding or averaging item vectors, modern systems apply lightweight **online gradient updates** inside the streaming job:

- If the user likes item \(i\) (\(y = 1\)): nudge the user vector \(u\) closer to the item vector.
- If the user skips or downvotes (\(y = 0\)): push them apart.

Common losses used on-the-fly:

- Logistic loss
- Triplet loss
- Contrastive losses

These micro-updates run inside Flink and are applied as **deltas** on top of the batch baseline.

---

## 2. Lambda Architecture for Feature Stores

Real-time streams are fast but vulnerable to state corruption, network drops, and out-of-order events (common with Kafka). The **Lambda Architecture** solves this by splitting the pipeline:

```
                  ┌──► Batch Layer (Nightly Spark / Airflow)
                  │         └── Recalibrates clean base vectors
[ User Actions ] ─┤
                  └──► Speed Layer (Flink Real-Time)
                            └── Updates hot cache (Redis)
                                       │
                                       ▼
                          Merged Serving View in Redis / Feature Store
```

### Layers Explained

- **Batch Layer (Cold / Correct)**  
  Runs nightly. Re-processes the full historical log to generate high-quality base embeddings and overwrites any accumulated drift or ordering errors.

- **Speed Layer (Hot / Fast)**  
  The Flink job that consumes real-time Kafka events and applies micro-adjustments (deltas) on top of the latest batch baseline.

- **Serving Layer**  
  Redis (or Feast / Tecton) merges on read or write:

  $$
  \text{User\_Vector\_Final} = \text{Batch\_Vector} + \text{Delta\_Vector}
  $$

This design gives you both **correctness** (batch) and **freshness** (speed).

---

## 3. Integrating Real-Time Vectors into Ranking Features

Once Flink pushes the updated vector into Redis or a feature store, the ranking stage consumes it. Typical engineered features include:

| Feature | Description | Example |
|---------|-------------|---------|
| **Real-Time Cosine Similarity** | \(\cos(V_{\text{user\_live}}, V_{\text{candidate\_item}})\) | Instant topical alignment |
| **Category Velocity** | Count of interactions in a specific category over a short window (Flink sliding window) | `tech_clicks_last_5min = 4` |
| **Recency-Weighted Interaction History** | Sequence of the last 10 item IDs fed into a lightweight Transformer target-attention layer | Similar to Alibaba DIN / DIEN architectures |

These features give the ranking model (usually a deep neural net or gradient-boosted tree) strong signals about the user’s current intent while still respecting long-term preferences.

---

## Next Phases

- **[Phase 2 – Implementation](PHASE_2_IMPLEMENTATION.md)**: Short-form video ranking + Greedy MMR diversity with full code sketches.
- **[Phase 3 – Observability](PHASE_3_OBSERVABILITY.md)**: Latency budgets, Prometheus, Grafana, and alerting for the 30–50 ms pipeline.
- **[Phase 4 – Offline Evaluation](PHASE_4_OFFLINE_EVALUATION.md)**: Safe replay of historical sessions with nDCG, ILD, and diversity metrics.

---

*This repository synthesizes common production patterns used in large-scale recommender systems (session-aware dual embeddings, online metric learning, Lambda Architecture, ranking, diversity, observability, and offline evaluation).*
