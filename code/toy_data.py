"""
Toy dataset generator for the minimal working example.
Creates synthetic user and item (video) embeddings with metadata.
"""

import numpy as np

np.random.seed(42)

# Configuration
EMBEDDING_DIM = 32
NUM_VIDEOS = 40
NUM_CREATORS = 8

CATEGORIES = ["gaming", "cooking", "tech", "comedy", "sports", "music", "travel", "edu"]
LANGUAGES = ["en", "es", "fr", "de"]


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def generate_toy_catalog():
    """Generate a small catalog of videos with embeddings and metadata."""
    catalog = []
    for i in range(NUM_VIDEOS):
        creator_id = f"creator_{i % NUM_CREATORS}"
        category = CATEGORIES[i % len(CATEGORIES)]
        language = LANGUAGES[i % len(LANGUAGES)]

        # Videos from the same creator share some embedding components
        base = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        creator_bias = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.4
        embedding = normalize(base + creator_bias * (i % NUM_CREATORS))

        catalog.append({
            "video_id": f"video_{i:03d}",
            "creator_id": creator_id,
            "category": category,
            "language": language,
            "embedding": embedding,
        })
    return catalog


def generate_user_vector(preferred_categories=None):
    """Create a synthetic user vector with mild preference toward certain categories."""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return normalize(vec)


def filter_catalog(catalog: list, category: str = None, language: str = None) -> list:
    """Filter catalog by optional metadata fields."""
    filtered = catalog
    if category is not None:
        filtered = [item for item in filtered if item["category"] == category]
    if language is not None:
        filtered = [item for item in filtered if item["language"] == language]
    return filtered


if __name__ == "__main__":
    catalog = generate_toy_catalog()
    user_vec = generate_user_vector()
    print(f"Generated {len(catalog)} videos and a user vector of dim {len(user_vec)}")
    print("Sample video:", {k: v for k, v in catalog[0].items() if k != "embedding"})

    # Demonstrate filtering
    gaming_en = filter_catalog(catalog, category="gaming", language="en")
    print(f"\nFiltered (category=gaming, language=en): {len(gaming_en)} videos")
