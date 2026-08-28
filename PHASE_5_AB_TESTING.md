# Phase 5: Online Experimentation & A/B Testing

Now that changes can be safely simulated offline (Phase 4), we need a robust framework to roll them out to live users without destabilizing the platform. This phase covers traffic bucketing, feature flags, guardrail metrics, and online statistical evaluation for short-form video feeds.

---

## 1. End-to-End Experimentation Architecture

```
[ Incoming User Request ]
         │
         ▼
[ Experiment Assignment Service ]
  (Consistent hashing: MurmurHash(user_id + experiment_id))
         │
         ├── Control Bucket  ──► Production Ranking + MMR (λ = 0.7)
         │
         └── Treatment Bucket ──► Experimental Ranking / λ / Gating
                    │
                    ▼
         [ Feature Flag Service ]  (Dynamic parameter overrides)
                    │
                    ▼
         [ Ranking + MMR Pipeline ]
                    │
                    ▼
         [ Metrics Collection ] ──► Online Success Metrics + Guardrails
                    │
                    ▼
         [ Statistical Evaluator ]  (Welch’s t-test / Bayesian)
                    │
                    ├── Continue Experiment
                    └── Early Stop + Rollback
```

---

## 2. Traffic Bucketing & Consistent Hashing

Users must stay in the same bucket across sessions and devices. Use a deterministic hash:

```python
import mmh3  # MurmurHash3

def assign_experiment_bucket(user_id: str, experiment_id: str, num_buckets: int = 100) -> int:
    """
    Returns a bucket ID in [0, num_buckets).
    Same user + experiment always maps to the same bucket.
    """
    hash_input = f"{user_id}:{experiment_id}".encode("utf-8")
    hash_value = mmh3.hash(hash_input, signed=False)
    return hash_value % num_buckets


def is_in_treatment(user_id: str, experiment_id: str, treatment_percentage: float = 0.1) -> bool:
    """
    treatment_percentage = 0.1 → 10% of users in treatment.
    """
    bucket = assign_experiment_bucket(user_id, experiment_id)
    threshold = int(treatment_percentage * 100)
    return bucket < threshold
```

**Best practices**
- Always include `experiment_id` in the hash key so the same user can be in different experiments simultaneously.
- Use at least 100 buckets for fine-grained traffic allocation.
- Log the assigned bucket on every request for auditability.

---

## 3. Feature Flags & Dynamic Parameter Control

Avoid redeploying code to change λ, α, or model versions. Use a feature-flag service (LaunchDarkly, Unleash, or a simple Redis-backed flag store).

Example configuration payload:

```json
{
  "experiment_id": "mmr_lambda_v2",
  "flags": {
    "mmr_lambda": 0.55,
    "flink_learning_rate_alpha": 0.15,
    "ranking_model_version": "two_tower_v3",
    "enable_session_gating": true
  }
}
```

Serving-layer pseudocode:

```python
def get_experiment_config(user_id: str, experiment_id: str) -> dict:
    if is_in_treatment(user_id, experiment_id):
        return feature_flag_service.get_treatment_config(experiment_id)
    return feature_flag_service.get_control_config(experiment_id)
```

This allows remote toggling of:
- MMR diversity coefficient (λ)
- Flink speed-layer learning rate (α)
- Ranking model architecture / version
- Gating behavior

---

## 4. Guardrail Metrics & Early Stopping

Never let an experiment run if it harms core user experience. Define hard guardrails that trigger automatic rollback.

### Critical Guardrails (examples)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Ranking p99 latency | > 30 ms | Immediate rollback |
| App crash rate | > 0.5% relative increase | Immediate rollback |
| Session length (median) | > 8% drop | Pause + investigate |
| Skip rate | > 10% relative increase | Pause + investigate |
| Creator diversity index | > 15% drop | Pause + investigate |

### Early-Stopping Logic (simplified)

```python
def should_early_stop(control_metrics: dict, treatment_metrics: dict) -> bool:
    # Latency guardrail
    if treatment_metrics["ranking_p99_ms"] > 30:
        return True

    # Engagement guardrail
    session_drop = (control_metrics["median_session_sec"] - treatment_metrics["median_session_sec"]) \
                   / control_metrics["median_session_sec"]
    if session_drop > 0.08:
        return True

    # Diversity guardrail
    diversity_drop = (control_metrics["creator_coverage"] - treatment_metrics["creator_coverage"]) \
                     / control_metrics["creator_coverage"]
    if diversity_drop > 0.15:
        return True

    return False
```

Integrate these checks into your metrics pipeline (every 5–15 minutes) and automatically disable the treatment flag on breach.

---

## 5. Online Statistical Evaluation

### Primary Success Metrics (short-form video)

- Session completion rate
- Average daily watch time
- Creator diversity index (unique creators per 100 impressions)
- Long-term retention (D1 / D7 return rate)

### Statistical Methods

**Frequentist (Welch’s t-test)** – simple and widely understood:

```python
from scipy import stats

def welch_ttest(control_values, treatment_values, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values, equal_var=False)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha
    }
```

**Bayesian approach** (recommended for continuous optimization):
- Maintain Beta or Normal-Inverse-Gamma posteriors over key metrics.
- Compute probability that treatment beats control.
- Optionally use multi-armed bandit allocation to gradually shift traffic toward better variants.

---

## 6. Safe Rollout Guidelines

1. **Start small** — 1–5% traffic for the first 24–48 hours.
2. **Monitor guardrails continuously** — automated checks every few minutes.
3. **Ramp gradually** — 5% → 10% → 25% → 50% only after statistical significance and no guardrail breaches.
4. **Keep a kill switch** — feature flag that can instantly force 100% control.
5. **Document everything** — experiment hypothesis, primary metric, guardrails, and decision criteria before launch.
6. **Post-experiment analysis** — always run a full offline replay (Phase 4) on the treatment traffic for deeper insight.

---

## 7. Production Gotchas

- **Sample Ratio Mismatch (SRM)** — if the observed traffic split drifts from the intended percentage, investigate hashing or filtering bugs immediately.
- **Novelty effects** — users may engage more (or less) simply because the feed looks different. Prefer longer experiments (7–14 days) for retention metrics.
- **Interaction effects** — multiple concurrent experiments can interfere. Use orthogonal bucketing or a formal experiment orchestration layer when running many tests.
- **Metric fishing** — pre-register primary and secondary metrics. Do not chase significance on dozens of metrics after the fact.

---

*This Phase-5 guide closes the loop from offline simulation to safe, measurable live experimentation for ranking, diversity, and gating changes.*
