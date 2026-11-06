"""Command-line interface for the k-means lab."""

import argparse
import sys
from collections.abc import Sequence

from kmeans.data import make_blobs
from kmeans.model import KMeans, Point, elbow_curve
from kmeans.plots import plot_clusters, plot_elbow


def run_clustering(
    X: Sequence[Point], k: int, max_iter: int, seed: int, plot: bool
) -> None:
    """Fit k-means, print cluster sizes and inertia, optionally save a plot."""
    model = KMeans(k=k, max_iter=max_iter, seed=seed)
    labels = model.fit(X)
    print(f"Iterations: {len(model.inertia_history) - 1}")
    for j in range(k):
        print(f"Cluster {j}: {labels.count(j)} points")
    print(f"Inertia: {model.inertia_history[-1]:.2f}")
    if plot:
        plot_clusters(X, labels, model.centroids, "clusters.png")
        print("Plot saved to clusters.png")


def run_elbow(X: Sequence[Point], k_max: int, seed: int) -> None:
    """Print inertia for k = 1..k_max and save the elbow plot."""
    inertias = elbow_curve(X, k_max, seed=seed)
    for k, value in enumerate(inertias, start=1):
        print(f"k={k:>2}: inertia={value:.2f}")
    plot_elbow(inertias, "elbow.png")
    print("Plot saved to elbow.png")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run clustering or the elbow method."""
    parser = argparse.ArgumentParser(
        prog="kmeans", description="Pure-Python k-means on generated 2D data."
    )
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--iter", type=int, default=100, dest="max_iter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true", help="save clusters.png")
    parser.add_argument("--elbow", action="store_true", help="save elbow.png")
    parser.add_argument("--k-max", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        X, _ = make_blobs(args.samples, spread=args.spread, seed=args.seed)
        if args.elbow:
            run_elbow(X, args.k_max, args.seed)
        else:
            run_clustering(X, args.k, args.max_iter, args.seed, args.plot)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
