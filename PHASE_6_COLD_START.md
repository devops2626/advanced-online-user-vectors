# Phase 6: Cold-Start Solutions for New Users and New Videos

The cold-start problem is the ultimate test of a feed algorithm: new users have zero history to build vectors from, and new videos have zero watch time or click data for ML scoring.

This document outlines strategies for bootstrapping recommendations when historical interaction data is completely absent.

---

## 1. New User Cold-Start (The Onboarding Problem)

When a brand-new user registers, `user:batch_vec` and `user:speed_vec` in Redis are empty. To prevent a blank or irrelevant feed, we use a tiered fallback approach:

- **Explicit Preference Harvesting (Onboarding)**  
  Prompt the user to select interest categories (e.g., Gaming, Cooking, Tech). Map these tags to pre-computed category centroid vectors stored in Redis, initializing their base profile vector instantly.

- **Contextual & Demographic Priors**  
  Seed the initial vector using coarse-grained signals such as country, device language, or referral source (e.g., users coming via a gaming campaign link get a slight weight tilt toward gaming clusters).

- **Global Popularity & Trending Pool**  
  If onboarding is skipped entirely, fall back to a globally curated "Trending / Top-N" pool while the Flink speed layer quietly logs their first clicks to build a real-time vector within minutes.

---

## 2. New Video Cold-Start (The Item Discovery Problem)

New videos lack interaction counts (likes, shares, watch duration), causing them to score poorly in collaborative filtering or historical embedding models. We solve this via content priors and forced exploration:

- **Content-Based Priors (Multimodal Embeddings)**  
  Generate initial item embeddings using raw metadata (e.g., text/image embeddings via CLIP from video frames, audio classification tags, and creator historical baseline performance).

- **Multi-Armed Bandits (Thompson Sampling)**  
  Allocate a dedicated percentage of feed slots (e.g., 5–10%) specifically for unproven new videos, treating them as "bandits" to rapidly measure their click-through and completion rates.

- **Promoted Seeding (Boosted Initial Impressions)**  
  Automatically inject new creator uploads into a low-risk, broad retrieval pool for a fixed number of initial impressions to bootstrap interaction data.

---

## 3. Code Sketch: Thompson Sampling Bandit for New Items (Python)

To give new videos a fair shot at exploration without tanking user engagement, we use a Beta-Binomial Thompson Sampling bandit wrapper that samples from a probability distribution based on historical successes (completing a video) and failures (skipping a video).

```python
import numpy as np


class ThompsonSamplingBandit:

    def __init__(self):
        # In production, these stats are pulled from Redis/Database per item
        # format: item_id -> {'successes': alpha, 'failures': beta}
        self.bandit_stats = {}

    def initialize_item(self, item_id: str):
        # Prior initialization (Beta(1, 1) represents uniform prior)
        if item_id not in self.bandit_stats:
            self.bandit_stats[item_id] = {"successes": 1.0, "failures": 1.0}

    def sample_score(self, item_id: str) -> float:
        if item_id not in self.bandit_stats:
            self.initialize_item(item_id)

        stats = self.bandit_stats[item_id]
        # Draw a random sample from the Beta distribution for this item
        return np.random.beta(stats["successes"], stats["failures"])

    def update_reward(self, item_id: str, watched_to_completion: bool):
        if item_id not in self.bandit_stats:
            self.initialize_item(item_id)

        if watched_to_completion:
            self.bandit_stats[item_id]["successes"] += 1.0
        else:
            self.bandit_stats[item_id]["failures"] += 1.0


# Example usage during retrieval/blending for a cold-start item:
bandit = ThompsonSamplingBandit()
new_video_id = "video_xyz_999"

bandit.initialize_item(new_video_id)
exploration_score = bandit.sample_score(new_video_id)
# This exploration_score can be blended into the final candidate ranking score
```

---

## 4. Production Gotchas & Implementation Notes

- **Exploration vs. Exploitation Trade-off**  
  Do not push unproven cold-start videos to high-intent users who are deep into a focused session. Target exploration slots toward users in early-session or broad browsing modes.

- **Cold-Start Expiration**  
  Define a strict threshold for when a video "graduates" out of the cold-start bandit pool (e.g., after receiving 500 impressions or 48 hours elapsed). Once graduated, it relies entirely on standard ML ranking scores.

- **Cold-User Poisoning**  
  Ensure that explicit onboarding tags do not trap a user in a permanent filter bubble. Fade out onboarding weights as the Flink real-time session vector accumulates actual interactive behavior.

---

*This Phase-6 guide completes the core architectural pillars by solving the cold-start problem for both new users and new content.*
