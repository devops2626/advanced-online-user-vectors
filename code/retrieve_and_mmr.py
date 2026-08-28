"""
Minimal working example:
1. Retrieve top candidates by cosine similarity
2. Apply Greedy Maximal Marginal Relevance (MMR) for diversity
"""

import numpy as np
from toy_data import generate_toy_catalog, generate_user_vector, normalize


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve_candidates(user_vector: np.ndarray, catalog: list, top_k: int = 15) -> list:
    """Simple brute-force retrieval by cosine similarity."""
    scored = []
    for item in catalog:
        score = cosine_similarity(user_vector, item["embedding"])
        scored.append({**item, "ml_score": score})
    scored.sort(key=lambda x: x["ml_score"], reverse=True)
    return scored[:top_k]


def compute_similarity(item_a: dict, item_b: dict) -> float:
    """Similarity used by MMR – penalize same creator heavily."""
    if item_a["creator_id"] == item_b["creator_id"]:
        return 1.0
    return cosine_similarity(item_a["embedding"], item_b["embedding"])


def maximal_marginal_relevance(
    candidates: list,
    top_n: int = 8,
    lambda_param: float = 0.7,
) -> list:
    """
    Greedy MMR implementation.
    Balances relevance (ml_score) against diversity (penalty for similar items).
    """
    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < top_n:
        best_score = -float("inf")
        best_item = None
        best_idx = -1

        for idx, item in enumerate(remaining):
            relevance = item["ml_score"]
            max_sim = 0.0
            if selected:
                max_sim = max(compute_similarity(item, s) for s in selected)

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


def main():
    print("=== Minimal Recommendation Demo ===\n")

    # 1. Generate toy data
    catalog = generate_toy_catalog()
    user_vector = generate_user_vector()
    print(f"Catalog size: {len(catalog)} videos")
    print(f"User vector dimension: {len(user_vector)}\n")

    # 2. Retrieve top candidates
    candidates = retrieve_candidates(user_vector, catalog, top_k=15)
    print("Top 10 by pure relevance (cosine):")
    for i, item in enumerate(candidates[:10], 1):
        print(f"  {i:2d}. {item['video_id']} | creator={item['creator_id']} | "
              f"category={item['category']} | score={item['ml_score']:.3f}")

    # 3. Apply MMR for diversity
    diversified = maximal_marginal_relevance(candidates, top_n=8, lambda_param=0.7)
    print("\nFinal feed after MMR (λ=0.7):")
    for i, item in enumerate(diversified, 1):
        print(f"  {i:2d}. {item['video_id']} | creator={item['creator_id']} | "
              f"category={item['category']} | score={item['ml_score']:.3f}")

    # 4. Show diversity effect
    creators_before = [c["creator_id"] for c in candidates[:8]]
    creators_after = [c["creator_id"] for c in diversified]
    print("\nCreator distribution (top 8):")
    print(f"  Before MMR: {creators_before}")
    print(f"  After  MMR: {creators_after}")


if __name__ == "__main__":
    main()
