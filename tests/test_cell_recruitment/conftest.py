import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def noisy_seeds(stabilized_video):
    """Create seeds with both real and false positive detections."""
    video, truth, _ = stabilized_video

    # Get real cell positions
    real_seeds = truth[["height", "width"]].copy()

    # Generate false positive seeds
    num_false_positives = len(real_seeds) // 2  # Add 50% more seeds as false positives

    # Create random positions for false positives, avoiding real cell locations
    false_positives = []
    margin = 20  # Minimum distance from real cells

    while len(false_positives) < num_false_positives:
        h = np.random.randint(0, video.sizes["height"])
        w = np.random.randint(0, video.sizes["width"])

        # Check if this position is far enough from real cells
        distances = np.sqrt(
            (real_seeds["height"] - h) ** 2 + (real_seeds["width"] - w) ** 2
        )

        if np.all(distances > margin):
            false_positives.append({"height": h, "width": w})

    false_seeds = pd.DataFrame(false_positives)
    all_seeds = pd.concat([real_seeds, false_seeds], ignore_index=True)

    # Add ground truth labels for testing
    all_seeds["is_real"] = [True] * len(real_seeds) + [False] * len(false_seeds)

    return all_seeds
