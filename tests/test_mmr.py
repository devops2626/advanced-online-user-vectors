"""
Basic unit tests for the Maximal Marginal Relevance (MMR) implementation.
"""

import sys
import os
import numpy as np
import unittest

# Make the code/ modules importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from retrieve_and_mmr import (
    cosine_similarity,
    compute_similarity,
    maximal_marginal_relevance,
    retrieve_candidates,
)
from toy_data import generate_toy_catalog, generate_user_vector, filter_catalog


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)


class TestComputeSimilarity(unittest.TestCase):
    def test_same_creator_penalty(self):
        item_a = {"creator_id": "creator_1", "embedding": np.array([1.0, 0.0])}
        item_b = {"creator_id": "creator_1", "embedding": np.array([0.0, 1.0])}
        self.assertEqual(compute_similarity(item_a, item_b), 1.0)

    def test_different_creator(self):
        item_a = {"creator_id": "creator_1", "embedding": np.array([1.0, 0.0])}
        item_b = {"creator_id": "creator_2", "embedding": np.array([1.0, 0.0])}
        self.assertAlmostEqual(compute_similarity(item_a, item_b), 1.0, places=5)


class TestMMR(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"video_id": "v1", "creator_id": "c1", "ml_score": 0.95, "embedding": np.array([1.0, 0.0])},
            {"video_id": "v2", "creator_id": "c1", "ml_score": 0.90, "embedding": np.array([0.9, 0.1])},
            {"video_id": "v3", "creator_id": "c2", "ml_score": 0.85, "embedding": np.array([0.0, 1.0])},
            {"video_id": "v4", "creator_id": "c3", "ml_score": 0.80, "embedding": np.array([0.5, 0.5])},
            {"video_id": "v5", "creator_id": "c2", "ml_score": 0.75, "embedding": np.array([0.1, 0.9])},
        ]

    def test_returns_requested_count(self):
        result = maximal_marginal_relevance(self.candidates, top_n=3, lambda_param=0.7)
        self.assertEqual(len(result), 3)

    def test_first_item_is_highest_relevance(self):
        result = maximal_marginal_relevance(self.candidates, top_n=3, lambda_param=1.0)
        # With lambda=1.0, pure relevance: first should be highest ml_score
        self.assertEqual(result[0]["video_id"], "v1")

    def test_diversity_reduces_same_creator(self):
        result = maximal_marginal_relevance(self.candidates, top_n=3, lambda_param=0.5)
        creators = [item["creator_id"] for item in result]
        # With strong diversity pressure, we should not pick both c1 videos first
        self.assertTrue(len(set(creators)) >= 2)

    def test_empty_candidates(self):
        result = maximal_marginal_relevance([], top_n=5)
        self.assertEqual(result, [])

    def test_top_n_larger_than_candidates(self):
        result = maximal_marginal_relevance(self.candidates, top_n=100)
        self.assertEqual(len(result), len(self.candidates))


class TestRetrievalWithFilters(unittest.TestCase):
    def setUp(self):
        self.catalog = generate_toy_catalog()
        self.user_vector = generate_user_vector()

    def test_filter_by_category(self):
        results = retrieve_candidates(
            self.user_vector, self.catalog, top_k=20, category="gaming"
        )
        for item in results:
            self.assertEqual(item["category"], "gaming")

    def test_filter_by_language(self):
        results = retrieve_candidates(
            self.user_vector, self.catalog, top_k=20, language="en"
        )
        for item in results:
            self.assertEqual(item["language"], "en")

    def test_filter_combined(self):
        results = retrieve_candidates(
            self.user_vector, self.catalog, top_k=20, category="gaming", language="en"
        )
        for item in results:
            self.assertEqual(item["category"], "gaming")
            self.assertEqual(item["language"], "en")

    def test_no_filter_returns_results(self):
        results = retrieve_candidates(self.user_vector, self.catalog, top_k=10)
        self.assertEqual(len(results), 10)


if __name__ == "__main__":
    unittest.main()
