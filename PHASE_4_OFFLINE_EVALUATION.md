# Phase 4: Offline Evaluation & Simulation for Feeds

Before rolling out changes to live production traffic, high-throughput platforms rely on offline evaluation and replay simulation. This phase provides a framework to replay historical user sessions, score alternative ranking models, and measure diversity without risking user experience.

---

## 1. The Offline Evaluation Workflow

```
[ Historical Kafka Logs (Parquet / S3) ]
                   │
                   ▼
       [ Offline Replay Engine ] ──► Simulates New Ranking Model & MMR
                   │
                   ├──► Ranking Metrics (nDCG@K, MRR)
                   └──► Diversity Metrics (Intra-List Distance, Creator Coverage)
```

- **Log Extraction**: Pull historical user action logs (impressions, clicks, watch time, skips) from your data lake (e.g., S3/BigQuery stored as Parquet files).
- **State Reconstruction**: Rebuild the user's short-term vector sequence up to timestamp *t* using the historical event stream.
- **Candidate Re-scoring**: Run the candidate items through the retrieval set and score them using the new ranking model or modified MMR parameters.
- **Metric Calculation**: Compare the simulated recommendations against what the user actually interacted with in the historical log.

---

## 2. Core Evaluation Metrics

### A. Ranking Quality

- **Normalized Discounted Cumulative Gain (nDCG@K)**  
  Evaluates the position of relevant items in the top-K recommendations, giving higher weight to items appearing earlier in the feed.

- **Mean Reciprocal Rank (MRR)**  
  Measures how high up the first clicked or liked video appeared in the feed.

### B. Diversity & Fairness

- **Intra-List Distance (ILD)**  
  Measures the average dissimilarity between all pairs of items in the recommended slate using their embedding distance.

- **Creator Coverage**  
  The proportion of unique creators represented across all generated user feeds during the evaluation window, ensuring the algorithm doesn't over-index on a tiny fraction of hyper-popular accounts.

---

## 3. Code Sketch: Offline Replay & Metric Evaluator (Python)

Below is a Python script that simulates an offline evaluation batch, re-ranking candidate items and computing nDCG@K alongside Intra-List Distance (ILD).

```python
import numpy as np


def compute_dcg(relevances, k):
    relevances = np.asfarray(relevances)[:k]
    if relevances.size == 0:
        return 0.0
    return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))


def compute_ndcg(actual_relevances, predicted_items, k=20):
    """Map predicted items to their actual relevance scores from logs."""
    relevances = [actual_relevances.get(item, 0.0) for item in predicted_items[:k]]
    dcg = compute_dcg(relevances, k)

    # Ideal DCG is sorted relevance scores descending
    ideal_relevances = sorted(actual_relevances.values(), reverse=True)
    idcg = compute_dcg(ideal_relevances, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_intra_list_distance(recommended_items):
    n = len(recommended_items)
    if n <= 1:
        return 0.0

    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            emb_i = recommended_items[i]["embedding"]
            emb_j = recommended_items[j]["embedding"]
            cosine_sim = np.dot(emb_i, emb_j) / (
                np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8
            )
            total_dist += 1.0 - cosine_sim
            count += 1

    return total_dist / count if count > 0 else 0.0


def evaluate_offline_batch(session_logs, scoring_model, top_k=20):
    ndcg_scores = []
    ild_scores = []

    for session in session_logs:
        user_vector = session["user_vector"]
        candidates = session["candidates"]
        actual_relevances = session[
            "ground_truth_relevances"
        ]  # {item_id: score (e.g., 1.0 for watch, 0.0 for skip)}

        # Score candidates using model
        for item in candidates:
            item["ml_score"] = scoring_model.predict(user_vector, item)

        # Sort and slice top K
        ranked_candidates = sorted(
            candidates, key=lambda x: x["ml_score"], reverse=True
        )
        predicted_ids = [item["item_id"] for item in ranked_candidates[:top_k]]
        top_items = ranked_candidates[:top_k]

        # Compute metrics
        ndcg = compute_ndcg(actual_relevances, predicted_ids, top_k)
        ild = compute_intra_list_distance(top_items)

        ndcg_scores.append(ndcg)
        ild_scores.append(ild)

    return {
        "mean_ndcg": np.mean(ndcg_scores),
        "mean_ild": np.mean(ild_scores),
    }
```

---

## 4. Production Gotchas in Offline Simulation

- **Feedback Loop Bias (Position Bias)**  
  Historical logs only record interactions for items that were actually shown at specific positions by the old production algorithm. Items hidden at position 100 never received clicks. Use **Inverse Propensity Scoring (IPS)** weights to unbias historical reward estimation.

- **Distribution Shift**  
  User interest vectors generated offline using static batch logs may fail to capture fast real-time session adjustments. Ensure your replay framework simulates real-time Flink state delta updates step-by-step.

---

*This Phase-4 guide completes the offline verification loop, allowing rigorous testing of ranking, MMR, and gating changes against historical data before any live rollout.*
