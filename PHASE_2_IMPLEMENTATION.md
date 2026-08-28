# Phase 2: Short-Form Video Ranking + Diversity (MMR)

**Practical implementation guide for a production short-form video feed engine.**

Short-form video is the ultimate proving ground: latency budgets are razor-thin (~30–50 ms for ranking) and user interest shifts minute-by-minute.

---

## 1. End-to-End Architecture Diagram (Lambda + Ranking + MMR)

```
[ User Action / Event ]
         │
         ├──► (Stream: Kafka) ──► [ Flink Speed Layer ] ──► Updates Short-Term Vector (Redis)
         │
         └──► (Batch: Nightly) ─► [ Spark Batch Layer ] ──► Updates Long-Term Vector (Redis)

[ Client Request ] ──► [ Retrieval Stage (ANN / Milvus) ]
                               │
                               ▼ (Top ~500 Candidates)
                       [ Feature Store / Redis Merge ]
                               │ (Fetches Merged Vectors + Sliding Window Features)
                               ▼
                       [ Ranking Model (ML Scoring / ~20ms) ]
                               │ (Top ~50 Candidates)
                               ▼
                       [ Greedy MMR Post-Processor (Diversity Check) ]
                               │ (Top 20 Final Feed)
                               ▼
                        [ Client / UI Feed Response ]
```

---

## 2. Code Sketch 1: Flink Dual-Vector Update Operator (Java)

This operator maintains both a long-term profile vector and a short-term session vector, applying a session gating mechanism whenever a user interacts with a video.

```java
public class DualVectorUpdateFunction extends KeyedProcessFunction<String, UserAction, UpdatedUserVector> {
    
    private MapState<String, float[]> longTermState;
    private MapState<String, float[]> shortTermState;
    private ValueState<Long> lastActionTimeState;
    
    private static final long SESSION_TIMEOUT_MS = 30 * 60 * 1000; // 30 mins
    private static final float ALPHA = 0.1f; // Learning rate

    @Override
    public void open(Configuration parameters) {
        longTermState = getRuntimeContext().getMapState(
            new MapStateDescriptor<>("longTerm", String.class, float[].class));
        shortTermState = getRuntimeContext().getMapState(
            new MapStateDescriptor<>("shortTerm", String.class, float[].class));
        lastActionTimeState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("lastActionTime", Long.class));
    }

    @Override
    public void processElement(UserAction action, Context ctx, Collector<UpdatedUserVector> out) throws Exception {
        String userId = action.getUserId();
        float[] itemEmbedding = action.getItemEmbedding();
        long currentTime = action.getTimestamp();

        Long lastTime = lastActionTimeState.value();
        float[] shortVec = shortTermState.get("vector");
        float[] longVec = longTermState.get("vector");

        if (shortVec == null) shortVec = new float[itemEmbedding.length];
        if (longVec == null) longVec = new float[itemEmbedding.length];

        // Reset short-term vector if session expired
        if (lastTime != null && (currentTime - lastTime > SESSION_TIMEOUT_MS)) {
            java.util.Arrays.fill(shortVec, 0.0f);
        }

        // Apply EMA update to short-term vector
        for (int i = 0; i < shortVec.length; i++) {
            shortVec[i] = (1 - ALPHA) * shortVec[i] + ALPHA * itemEmbedding[i] * action.getWeight();
        }

        shortTermState.put("vector", shortVec);
        lastActionTimeState.update(currentTime);

        // Dynamic gating blend for final serving vector
        float gatingFactor = calculateGatingFactor(currentTime, lastTime);
        float[] finalVec = new float[itemEmbedding.length];
        for (int i = 0; i < finalVec.length; i++) {
            finalVec[i] = gatingFactor * shortVec[i] + (1 - gatingFactor) * longVec[i];
        }

        out.collect(new UpdatedUserVector(userId, finalVec, currentTime));
    }

    private float calculateGatingFactor(long now, Long lastTime) {
        if (lastTime == null) return 1.0f;
        long elapsed = now - lastTime;
        // Exponential decay of short-term weight over inactivity time
        return (float) Math.exp(-elapsed / (15.0 * 60 * 1000)); // 15 min half-life scale
    }
}
```

---

## 3. Code Sketch 2: Redis Merge Logic (Python)

When building feature vectors for the ranking model, the serving layer merges the batch baseline with the real-time speed delta on the fly.

```python
import redis
import numpy as np

r = redis.Redis(host="localhost", port=6379, db=0)


def get_merged_user_vector(user_id: str, dimension: int = 128) -> np.ndarray:
    pipeline = r.pipeline()
    pipeline.get(f"user:batch_vec:{user_id}")
    pipeline.get(f"user:speed_vec:{user_id}")
    batch_bytes, speed_bytes = pipeline.execute()

    # Fallback to zeros if missing
    batch_vec = (
        np.frombuffer(batch_bytes, dtype=np.float32)
        if batch_bytes
        else np.zeros(dimension, dtype=np.float32)
    )
    speed_vec = (
        np.frombuffer(speed_bytes, dtype=np.float32)
        if speed_bytes
        else np.zeros(dimension, dtype=np.float32)
    )

    # Lambda merge: Batch baseline + dynamic real-time offset
    final_vector = batch_vec + 0.8 * speed_vec
    return final_vector
```

---

## 4. Code Sketch 3: Greedy MMR Post-Processor (Python)

To prevent echo chambers and consecutive duplicate creators/audio tracks in a short-form video feed, apply Maximal Marginal Relevance right before rendering:

```python
import numpy as np


def compute_similarity(item_a, item_b):
    # Custom similarity metric checking creator ID, audio ID, or embedding cosine distance
    if item_a["creator_id"] == item_b["creator_id"]:
        return 1.0  # Maximum penalty for same creator
    if item_a["audio_id"] == item_b["audio_id"]:
        return 0.8  # High penalty for same audio trend
    # Fallback to embedding cosine similarity
    return np.dot(item_a["embedding"], item_b["embedding"]) / (
        np.linalg.norm(item_a["embedding"]) * np.linalg.norm(item_b["embedding"])
    )


def maximal_marginal_relevance(
    candidates: list,
    top_n: int = 20,
    lambda_param: float = 0.7,
) -> list:
    """Selects top items balancing ML score and diversity using MMR."""
    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < top_n:
        best_score = -float("inf")
        best_item = None
        best_idx = -1

        for idx, item in enumerate(remaining):
            relevance = item["ml_score"]

            # Calculate max similarity to already selected items
            max_sim = 0.0
            if selected:
                max_sim = max(compute_similarity(item, s) for s in selected)

            # MMR formula: balance raw score against redundancy penalty
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * max_sim)

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = item
                best_idx = idx

        if best_item is not None:
            selected.append(best_item)
            remaining.pop(best_idx)
        else:
            break

    return selected
```

---

## 5. Production Gotchas & Implementation Notes

- **Out-of-Order Events**  
  Kafka partitions by `user_id`, guaranteeing chronological order per user, but cross-device actions can still arrive out of sequence. Use Flink’s `BoundedOutOfOrdernessTimestampExtractor` with a small watermark delay (e.g., 5 seconds) to handle network jitters.

- **State TTL in Flink**  
  Unbounded user states will balloon memory/RocksDB storage over time. Configure state TTL (e.g., clear user state if inactive for 7 days) to automatically prune dormant users.

- **Vector Dimension Drift**  
  Ensure that vector dimensions match strictly between content metadata ingestion and the user vector state. If offline models upgrade from 128 to 256 dimensions, use a version-prefixed Redis key structure (e.g., `user:v2:speed_vec:{user_id}`).

---

*This Phase-2 guide translates the conceptual framework into concrete, production-ready components for a short-form video recommendation pipeline.*
