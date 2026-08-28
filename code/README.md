# Minimal Working Example

This directory contains a self-contained toy example that demonstrates core ideas from the repository:

- User and item embeddings
- Simple vector retrieval (cosine similarity)
- Greedy Maximal Marginal Relevance (MMR) for diversity

No external services (Pinecone, Redis, Flink, etc.) are required. Everything runs with pure Python + NumPy.

## Quick Start

```bash
pip install numpy
python retrieve_and_mmr.py
```

## Files

| File | Description |
|------|-------------|
| `toy_data.py` | Generates a small synthetic dataset of user & item embeddings |
| `retrieve_and_mmr.py` | Runs retrieval + MMR and prints the diversified feed |
| `requirements.txt` | Minimal dependencies |

## What the demo shows

1. A user vector is compared against a catalog of video embeddings.
2. Top candidates are retrieved by cosine similarity.
3. MMR re-ranks the list to reduce redundancy (same creator / similar content).
4. The final diversified list is printed.

This mirrors the retrieval → ranking → diversity flow described in Phases 2 and 7.
