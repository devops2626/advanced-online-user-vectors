# Architecture Overview

**Advanced Online User Vector Updates for Recommendation Systems**

This repository is a complete, end-to-end engineering handbook for building production-grade short-form video recommendation systems. It covers the full lifecycle from real-time vector updates to safe live experimentation, cold-start handling, and vector database selection.

---

## Document Map

| Phase | File | Focus |
|-------|------|-------|
| Core Concepts | [README.md](README.md) | Dual-vector models, online learning, Lambda Architecture |
| Phase 2 | [PHASE_2_IMPLEMENTATION.md](PHASE_2_IMPLEMENTATION.md) | Ranking features, Flink dual-vector operator, Redis merge, Greedy MMR |
| Phase 3 | [PHASE_3_OBSERVABILITY.md](PHASE_3_OBSERVABILITY.md) | Latency budgets, Prometheus, Grafana, alerting |
| Phase 4 | [PHASE_4_OFFLINE_EVALUATION.md](PHASE_4_OFFLINE_EVALUATION.md) | Offline replay, nDCG, Intra-List Distance, simulation |
| Phase 5 | [PHASE_5_AB_TESTING.md](PHASE_5_AB_TESTING.md) | Traffic bucketing, feature flags, guardrails, online stats |
| Phase 6 | [PHASE_6_COLD_START.md](PHASE_6_COLD_START.md) | New-user onboarding, new-item exploration (Thompson Sampling) |
| Phase 7 | [PHASE_7_VECTOR_DB_SELECTION.md](PHASE_7_VECTOR_DB_SELECTION.md) | Vector database comparison & integration for candidate retrieval |

---

## High-Level System Architecture

```
[ User Actions ] ──► Kafka
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Flink Speed Layer        Spark Batch Layer
   (Short-term vector)      (Long-term vector)
          │                       │
          └───────────┬───────────┘
                      ▼
              Redis / Feature Store
                      │
                      ▼
[ Client Request ] ──► Vector DB Retrieval (Pinecone / Qdrant / etc.)
                      │
                      ▼
              Feature Merge + Ranking Model
                      │
                      ▼
              Greedy MMR Diversity
                      │
                      ▼
              Final Feed (with Cold-Start slots)
```

**Supporting layers:**
- Observability (Phase 3) monitors every stage against a strict 30–50 ms budget.
- Offline Evaluation (Phase 4) safely tests changes on historical logs.
- A/B Testing (Phase 5) enables controlled live rollouts with guardrails.
- Cold-Start (Phase 6) bootstraps both new users and new videos.
- Vector DB Selection (Phase 7) provides the retrieval foundation for candidate generation.

---

## Recommended Reading Order

1. Start with the **README** for foundational concepts (dual vectors + Lambda).
2. Read **Phase 2** for the concrete ranking + diversity implementation.
3. Add **Phase 3** to understand how to keep the system fast and reliable.
4. Use **Phase 4** to evaluate changes safely offline.
5. Apply **Phase 5** to roll changes out to real users.
6. Finish with **Phase 6** to handle the inevitable new-user and new-item cases.
7. Use **Phase 7** when choosing and integrating the vector database for retrieval.

---

## Key Design Principles Embodied in This Repo

- **Freshness + Correctness** via Lambda Architecture
- **Session awareness** through dual (long/short) vectors and dynamic gating
- **Diversity by design** with Maximal Marginal Relevance
- **Measurability** at every stage (latency, ranking quality, diversity)
- **Safety** through offline simulation and online guardrails
- **Growth enablement** via explicit cold-start strategies
- **Flexible retrieval** via a well-chosen vector database

---

*This repository is intended as a practical reference for engineers building high-scale, real-time recommendation systems.*
