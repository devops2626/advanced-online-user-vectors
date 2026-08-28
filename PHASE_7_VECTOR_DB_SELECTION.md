# Phase 7: Vector Database Selection & Integration for Recommendation Retrieval

Vector retrieval is a critical stage in modern recommendation systems. In the short-form video pipeline described in earlier phases, the vector database is responsible for the **Candidate Generation / Retrieval** step — taking the live user vector and returning a high-recall set of relevant item embeddings (typically top 500–1000 candidates) within a tight latency budget (~10 ms).

This document compares the major vector databases and shows how to integrate one into the existing architecture.

---

## 1. Role in the Recommendation Funnel

```
[ User Action Stream ]
         │
         ▼
[ Flink Speed Layer ] ──► Updates live user vector in Redis
         │
         ▼
[ Client Request ]
         │
         ▼
[ Vector DB Retrieval ]  ◄── Query with live user vector + metadata filters
         │                   (top ~500–1000 candidates)
         ▼
[ Feature Store Merge + Ranking Model ]
         │
         ▼
[ Greedy MMR Diversity ]
         │
         ▼
[ Final Feed ]
```

Key requirements for this stage:
- Low and predictable latency (p99 ideally under 15–20 ms)
- Strong metadata filtering (creator, category, language, freshness, safety flags)
- Support for hybrid (dense + sparse) search when useful
- Ability to scale to tens or hundreds of millions of item embeddings
- Easy upsert of new/updated item embeddings (including cold-start content)

---

## 2. Comparison of Major Vector Databases (2026)

| Database | Deployment | Best For | Hybrid Search | Metadata Filtering | Latency Profile | Ops Burden | Cost Profile |
|----------|------------|----------|---------------|---------------------|-----------------|------------|--------------|
| **Pinecone** | Fully managed (serverless) | Zero-ops production | Strong (dense + sparse) | Excellent | Low & predictable | Almost none | Higher at scale |
| **Qdrant** | Self-host or managed | High performance + filtering | Strong | Excellent (payload indexes) | Often fastest | Low–medium | Lower (self-host) |
| **Weaviate** | Self-host or managed | Hybrid + modular AI stack | Excellent (BM25 + vector) | Strong | Good | Low–medium | Medium |
| **Milvus / Zilliz** | Self-host or managed | Extreme scale (100M–billions) | Strong | Strong | Good at scale | Higher | Lowest at huge scale |
| **pgvector** | Postgres extension | Teams already on Postgres | Limited / manual | SQL WHERE | Higher | Low (if Postgres exists) | Lowest |
| **Chroma** | Embedded / self-host | Prototyping & local | Limited | Basic | Varies | Very low | Free (local) |

### Quick Decision Guide

- **Want zero operations and fastest path to production** → Pinecone
- **Need maximum performance + rich filtering and can run infrastructure** → Qdrant
- **Need best-in-class hybrid (keyword + semantic) search** → Weaviate
- **Expecting hundreds of millions to billions of vectors** → Milvus / Zilliz
- **Already heavily invested in Postgres and scale is moderate** → pgvector
- **Just prototyping** → Chroma

---

## 3. Practical Integration Pattern

### Architecture Placement

1. **Item Embedding Pipeline** (batch or streaming)  
   Generate or update video embeddings (content-based via CLIP, collaborative, or hybrid) and upsert them into the vector database with rich metadata.

2. **User Vector Source**  
   The live user vector comes from the Redis merge of batch + speed-layer deltas (Phases 1–2).

3. **Retrieval Call**  
   On each feed request, query the vector DB with the user vector + metadata filters (e.g., exclude already-seen items, apply language/region constraints, boost freshness).

4. **Handoff to Ranking**  
   Pass the returned candidate IDs + scores into the ranking model and subsequent MMR stage.

### Example Metadata Schema for Short-Form Video

```json
{
  "video_id": "vid_123456",
  "creator_id": "creator_789",
  "category": "gaming",
  "language": "en",
  "duration_sec": 28,
  "upload_ts": 1724800000,
  "is_new": true,
  "safety_score": 0.98,
  "embedding": [0.12, -0.05, ...]
}
```

---

## 4. Code Sketches

### A. Upserting Item Embeddings (Pinecone example)

```python
from pinecone import Pinecone
import numpy as np

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index("video-embeddings")

def upsert_video_embedding(
    video_id: str,
    embedding: list[float],
    creator_id: str,
    category: str,
    language: str,
    upload_ts: int,
    is_new: bool = False,
):
    index.upsert(
        vectors=[
            {
                "id": video_id,
                "values": embedding,
                "metadata": {
                    "creator_id": creator_id,
                    "category": category,
                    "language": language,
                    "upload_ts": upload_ts,
                    "is_new": is_new,
                },
            }
        ],
        namespace="production",
    )
```

### B. Querying with Live User Vector + Filters

```python
def retrieve_candidates(
    user_vector: list[float],
    top_k: int = 500,
    language: str = "en",
    exclude_creator_ids: list[str] | None = None,
    min_upload_ts: int | None = None,
) -> list[dict]:
    filter_dict = {"language": {"$eq": language}}

    if exclude_creator_ids:
        filter_dict["creator_id"] = {"$nin": exclude_creator_ids}

    if min_upload_ts:
        filter_dict["upload_ts"] = {"$gte": min_upload_ts}

    results = index.query(
        vector=user_vector,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
        namespace="production",
    )

    return [
        {
            "video_id": match["id"],
            "score": match["score"],
            "metadata": match.get("metadata", {}),
        }
        for match in results["matches"]
    ]
```

### C. Blending Cold-Start Exploration (ties to Phase 6)

```python
def retrieve_with_exploration(
    user_vector: list[float],
    top_k: int = 500,
    exploration_ratio: float = 0.08,
):
    # Main semantic retrieval
    main_candidates = retrieve_candidates(user_vector, top_k=int(top_k * (1 - exploration_ratio)))

    # Dedicated exploration slots for new items (from Thompson Sampling or simple filter)
    new_item_candidates = retrieve_candidates(
        user_vector,
        top_k=int(top_k * exploration_ratio),
        min_upload_ts=...  # or filter is_new=True
    )

    # Merge and return
    return main_candidates + new_item_candidates
```

---

## 5. Production Considerations

- **Latency Budget**  
  Keep retrieval well under 10–15 ms p99 so the downstream ranking + MMR stages still fit inside the overall 40–50 ms target.

- **Filtering Strategy**  
  Prefer pre-filtering (metadata constraints applied during ANN search) over post-filtering to avoid under-filled result sets.

- **Freshness & Updates**  
  New and updated videos should be upserted with low latency. Most modern vector DBs support near-real-time visibility of new vectors.

- **Multi-tenancy / Namespaces**  
  Use namespaces (Pinecone) or equivalent partitioning to isolate environments (prod / staging) or major content verticals if needed.

- **Cost Control**  
  Monitor read-unit consumption closely. Aggressive top_k values or very high QPS can dominate the bill on usage-based systems.

- **Fallback**  
  Always have a degraded path (e.g., popularity or trending pool) if the vector DB is unavailable or exceeds latency SLOs.

---

## 6. Recommendation for This Repository’s Architecture

For most teams building the system described in Phases 1–6:

- **Start with Pinecone** if you want to move fast with minimal operations.
- **Switch to Qdrant or Milvus** later if cost at very large scale or maximum control becomes the dominant concern.
- Keep the retrieval interface abstract (a simple `retrieve_candidates(user_vector, filters) → list`) so the underlying vector DB can be swapped without rewriting the ranking and diversity layers.

---

*This Phase-7 guide completes the retrieval layer of the recommendation stack and connects the real-time user vector pipeline to high-quality candidate generation.*
