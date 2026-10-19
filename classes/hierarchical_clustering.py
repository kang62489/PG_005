"""
Hierarchical clustering of flashes, cut at the best mean silhouette (SREAnalyzer.group_centroid / group_trace).

  Step 1. build_tree      : flash x flash distance -> average-linkage tree (n - 1 merges)
  Step 2. best_silhouette : cut the tree at every h of the sweep -> mean silhouette -> best h (first of the maximum)
  Step 3. clustering      : clusters of the best cut, renumbered 0..K-1 in first-flash order

Example:
    >>> hc = HierarchicalClustering(distance, CENTROID_CUT_PX)
    >>> hc.run()
    >>> hc.cluster_ids, hc.cut, hc.tree, hc.sweep
"""

## Modules
# Third-party imports
import numpy as np
import pandas as pd
from rich.console import Console
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

console = Console()


# ===========================================================================
#
#   CONFIG
#
# ===========================================================================

LINKAGE_METHOD = "average"    # cluster distance = mean of all cross pairs


class HierarchicalClustering:
    """Tree of a flash x flash distance, cut at its best mean silhouette h.

    Results:
        tree                              (build_tree; None if < 2 flashes)
        sweep, cut {h, n_clusters, silhouette}   (best_silhouette; empty sweep if < 3 flashes)
        cluster_ids (0..K-1 per flash)    (clustering)
    """

    def __init__(self, distance: np.ndarray, cut_heights: np.ndarray, linkage_method: str = LINKAGE_METHOD) -> None:
        self.distance = distance
        self.cut_heights = cut_heights
        self.linkage_method = linkage_method
        self.n = len(distance)
        self.tree: np.ndarray | None = None
        self.sweep = pd.DataFrame(columns=["cut_height", "n_clusters", "mean_silhouette"])
        self.cut = {"h": np.nan, "n_clusters": 0, "silhouette": np.nan}
        self.best_cut_ids: np.ndarray | None = None  # fcluster ids of the best cut, before renumbering

    def run(self) -> None:
        """build_tree -> best_silhouette -> clustering."""
        self.build_tree()
        self.best_silhouette()
        self.clustering()

    def build_tree(self) -> None:
        """Average-linkage tree of the distance (n - 1 merges)."""
        if self.n > 1:
            self.tree = linkage(squareform(self.distance, checks=False), method=self.linkage_method)

    def calculate_silhouette(self, cluster_ids: np.ndarray) -> float:
        """Mean silhouette over all flashes; a flash alone in its cluster scores 0 (= sklearn silhouette_score)."""
        clusters = np.unique(cluster_ids)
        one_hot = (cluster_ids[:, None] == clusters[None, :]).astype(float)  # flash x cluster
        sizes = one_hot.sum(axis=0)
        sum_to_cluster = self.distance @ one_hot  # flash x cluster: summed distance to each cluster
        own = np.searchsorted(clusters, cluster_ids)
        own_size = sizes[own]
        a = sum_to_cluster[np.arange(len(cluster_ids)), own] / np.maximum(own_size - 1, 1)
        mean_to_cluster = sum_to_cluster / sizes
        mean_to_cluster[np.arange(len(cluster_ids)), own] = np.inf
        b = mean_to_cluster.min(axis=1)
        scores = np.where(own_size > 1, (b - a) / np.maximum(a, b), 0.0)
        return float(scores.mean())

    def best_silhouette(self) -> None:
        """Cut at every h -> sweep (cut_height, n_clusters, mean_silhouette); best h = the first h of the maximum
        (start of a plateau). Only cuts with 1 < clusters < flashes have a silhouette; none -> h, silhouette NaN."""
        if self.n <= 2:
            return
        best_h, best_score = np.nan, -np.inf
        rows = []
        for cut_height in self.cut_heights:
            cluster_ids = fcluster(self.tree, t=cut_height, criterion="distance")
            score = (self.calculate_silhouette(cluster_ids) if 1 < cluster_ids.max() < len(cluster_ids)
                     else np.nan)
            rows.append({"cut_height": float(cut_height), "n_clusters": int(cluster_ids.max()),
                         "mean_silhouette": score})
            if score > best_score:
                best_h, best_score, self.best_cut_ids = float(cut_height), score, cluster_ids
        self.sweep = pd.DataFrame(rows)
        self.cut["h"] = best_h
        self.cut["silhouette"] = best_score if np.isfinite(best_score) else np.nan

    def clustering(self) -> None:
        """Clusters of the best cut, renumbered 0..K-1 in first-flash order. 0 flashes -> no clusters;
        no valid cut -> all flashes in one cluster."""
        if self.n == 0:
            raw_ids = np.array([], dtype=int)
            console.log("  [yellow]no flashes above threshold -- 0 recur_zones[/yellow]")
        elif self.best_cut_ids is None:
            raw_ids = np.ones(self.n, dtype=int)
            console.log(f"  [yellow]{self.n} flash(es): no cut with 1 < clusters < flashes -- all flashes in one "
                        f"cluster[/yellow]")
        else:
            raw_ids = self.best_cut_ids
        self.cut["n_clusters"] = int(raw_ids.max()) if self.n else 0
        new_id = {raw: new for new, raw in enumerate(pd.unique(raw_ids))}  # first-seen order
        self.cluster_ids = np.array([new_id[raw] for raw in raw_ids], dtype=int)
