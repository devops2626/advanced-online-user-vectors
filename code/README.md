# Minimal Working Example

This directory contains a self-contained toy example that demonstrates core ideas from the repository:

- User and item embeddings
- Vector retrieval with optional metadata filters (category, language)
- Greedy Maximal Marginal Relevance (MMR) for diversity

No external services (Pinecone, Redis, Flink, etc.) are required. Everything runs with pure Python + NumPy.

## Quick Start

```bash
pip install -r requirements.txt
python retrieve_and_mmr.py
```

## Interactive Notebook

```bash
pip install jupyter
jupyter notebook demo_notebook.ipynb
```

## Run Tests

From the repository root:

```bash
python -m pytest tests/ -v
# or
python -m unittest tests.test_mmr -v
```

## Files

| File | Description |
|------|-------------|
| `toy_data.py` | Generates synthetic catalog with embeddings + metadata (category, language) |
| `retrieve_and_mmr.py` | Retrieval (with filters) + MMR demo script |
| `demo_notebook.ipynb` | Interactive Jupyter exploration |
| `requirements.txt` | Minimal dependencies |

## What the demo shows

1. A user vector is compared against a catalog of video embeddings.
2. Optional metadata filters (category / language) restrict the candidate pool.
3. Top candidates are retrieved by cosine similarity.
4. MMR re-ranks the list to reduce redundancy (same creator / similar content).
5. The final diversified list is printed.

This mirrors the retrieval → ranking → diversity flow described in Phases 2 and 7.
