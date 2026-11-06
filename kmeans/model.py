"""Pure-Python k-means clustering."""

import random
from collections.abc import Sequence

Point = Sequence[float]


def _sq_dist(a: Point, b: Point) -> float:
    """Return squared Euclidean distance between two points."""
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


class KMeans:
    """K-means clustering implemented with plain Python lists."""

    def __init__(
        self,
        k: int = 3,
        max_iter: int = 100,
        seed: int | None = None,
        tol: float = 1e-6,
    ) -> None:
        """Store hyperparameters and prepare empty model state."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.tol = tol
        self.centroids: list[list[float]] = []
        self.inertia_history: list[float] = []

    def _init_centroids(self, X: Sequence[Point]) -> list[list[float]]:
        """Pick k initial centroids with k-means++ seeding."""
        if self.k > len(X):
            raise ValueError(f"k={self.k} is larger than number of points {len(X)}")
        rng = random.Random(self.seed)
        centroids = [list(rng.choice(X))]
        while len(centroids) < self.k:
            # d2[i] = squared distance from point i to its nearest chosen centroid
            d2 = []
            for p in X:
                d2.append(min(_sq_dist(p, c) for c in centroids))
            if sum(d2) > 0:
                chosen = rng.choices(X, weights=d2)[0]
            else:
                chosen = rng.choice(X)
            centroids.append(list(chosen))
        return centroids

    def _assign(self, X: Sequence[Point]) -> list[int]:
        """Return index of the nearest centroid for every point."""
        labels = []
        for p in X:
            best = 0
            best_dist = _sq_dist(p, self.centroids[0])
            for j in range(1, self.k):
                dist = _sq_dist(p, self.centroids[j])
                if dist < best_dist:
                    best = j
                    best_dist = dist
            labels.append(best)
        return labels

    def _update(self, X: Sequence[Point], labels: list[int]) -> list[list[float]]:
        """Recompute each centroid as the mean of its assigned points."""
        dim = len(X[0])
        sums = [[0.0] * dim for _ in range(self.k)]
        counts = [0] * self.k
        for p, label in zip(X, labels, strict=True):
            counts[label] += 1
            for d in range(dim):
                sums[label][d] += p[d]
        new_centroids: list[list[float]] = []
        for j in range(self.k):
            if counts[j] == 0:
                new_centroids.append(list(self.centroids[j]))
            else:
                new_centroids.append([s / counts[j] for s in sums[j]])
        return new_centroids

    def inertia(self, X: Sequence[Point], labels: list[int]) -> float:
        """Return sum of squared distances from points to their centroids."""
        if not self.centroids:
            raise RuntimeError("Model is not fitted: call fit() first")
        return sum(
            _sq_dist(p, self.centroids[label])
            for p, label in zip(X, labels, strict=True)
        )

    def fit(self, X: Sequence[Point]) -> list[int]:
        """Run k-means on X and return cluster label for every point."""
        if not X:
            raise ValueError("X must contain at least one point")
        self.centroids = self._init_centroids(X)
        labels = self._assign(X)
        self.inertia_history = [self.inertia(X, labels)]
        for _ in range(self.max_iter):
            new_centroids = self._update(X, labels)
            # shift = how far the most-moved centroid travelled (squared)
            shift = 0.0
            for old, new in zip(self.centroids, new_centroids, strict=True):
                shift = max(shift, _sq_dist(old, new))
            self.centroids = new_centroids
            labels = self._assign(X)
            self.inertia_history.append(self.inertia(X, labels))
            if shift <= self.tol**2:
                break
        return labels

    def predict(self, X: Sequence[Point]) -> list[int]:
        """Return nearest-centroid labels for new points."""
        if not self.centroids:
            raise RuntimeError("Model is not fitted: call fit() first")
        return self._assign(X)


def elbow_curve(X: Sequence[Point], k_max: int, seed: int | None = None) -> list[float]:
    """Return final inertia for every k from 1 to k_max."""
    if k_max < 1:
        raise ValueError(f"k_max must be >= 1, got {k_max}")
    inertias: list[float] = []
    for k in range(1, min(k_max, len(X)) + 1):
        model = KMeans(k=k, seed=seed)
        model.fit(X)
        inertias.append(model.inertia_history[-1])
    return inertias
