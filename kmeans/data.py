"""Synthetic data generation for k-means experiments."""

import random

DEFAULT_CENTERS: list[tuple[float, float]] = [(0.0, 0.0), (10.0, 10.0), (-10.0, 10.0)]


def make_blobs(
    n_samples: int = 300,
    centers: list[tuple[float, float]] | None = None,
    spread: float = 1.0,
    seed: int | None = None,
) -> tuple[list[list[float]], list[int]]:
    """Generate 2D points around given centers.

    Returns points and the index of the true center for each point.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if centers is None:
        centers = DEFAULT_CENTERS
    rng = random.Random(seed)
    points: list[list[float]] = []
    true_labels: list[int] = []
    for i in range(n_samples):
        c = i % len(centers)
        cx, cy = centers[c]
        points.append([rng.gauss(cx, spread), rng.gauss(cy, spread)])
        true_labels.append(c)
    return points, true_labels
