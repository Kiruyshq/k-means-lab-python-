"""Matplotlib plots for k-means results, saved straight to PNG files."""

from collections.abc import Sequence

from matplotlib.figure import Figure

from kmeans.model import Point


def plot_clusters(
    X: Sequence[Point], labels: list[int], centroids: Sequence[Point], path: str
) -> None:
    """Save points coloured by cluster, with centroids as black crosses."""
    fig = Figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ax.scatter([p[0] for p in X], [p[1] for p in X], c=labels, cmap="tab10", s=15)
    ax.scatter(
        [c[0] for c in centroids],
        [c[1] for c in centroids],
        c="black",
        marker="X",
        s=200,
        label="centroids",
    )
    ax.set(title=f"K-means (k={len(centroids)})", xlabel="x", ylabel="y")
    ax.legend()
    fig.savefig(path, dpi=120)


def plot_elbow(inertias: list[float], path: str) -> None:
    """Save the elbow plot: inertia as a function of k."""
    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot()
    ks = list(range(1, len(inertias) + 1))
    ax.plot(ks, inertias, marker="o")
    ax.set(title="Elbow method", xlabel="k", ylabel="inertia", xticks=ks)
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120)
