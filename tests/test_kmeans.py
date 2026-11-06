"""Tests for the pure-Python KMeans implementation."""

from itertools import pairwise

import pytest

from kmeans.data import make_blobs
from kmeans.model import KMeans, _sq_dist


def test_sq_dist() -> None:
    """Squared distance of a 3-4-5 triangle is 25."""
    assert _sq_dist([0.0, 0.0], [3.0, 4.0]) == 25.0


def test_finds_true_clusters() -> None:
    """Each true blob maps to exactly one found cluster."""
    X, y = make_blobs(300, seed=0)
    labels = KMeans(k=3, seed=42).fit(X)
    for c in range(3):
        # labels the model gave to points of true blob c
        found = set()
        for lab, true in zip(labels, y, strict=True):
            if true == c:
                found.add(lab)
        assert len(found) == 1
    assert len(set(labels)) == 3


def test_inertia_non_increasing() -> None:
    """Inertia never grows between iterations."""
    X, _ = make_blobs(300, spread=3.0, seed=1)
    model = KMeans(k=3, seed=0)
    model.fit(X)
    h = model.inertia_history
    assert all(a >= b - 1e-9 for a, b in pairwise(h))


def test_same_seed_same_result() -> None:
    """Same seed gives identical labels."""
    X, _ = make_blobs(300, seed=0)
    assert KMeans(k=3, seed=7).fit(X) == KMeans(k=3, seed=7).fit(X)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_number_of_clusters(k: int) -> None:
    """Different k gives exactly k clusters."""
    X, _ = make_blobs(300, seed=0)
    assert len(set(KMeans(k=k, seed=7).fit(X))) == k


def test_invalid_k() -> None:
    """k < 1 is rejected."""
    with pytest.raises(ValueError):
        KMeans(k=0)


def test_predict_before_fit() -> None:
    """predict() without fit() raises a clear error."""
    with pytest.raises(RuntimeError):
        KMeans(k=2).predict([[0.0, 0.0]])
