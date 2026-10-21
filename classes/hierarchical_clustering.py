"""
Hierarchical clustering of flashes, cut at the best mean silhouette or at the acceleration elbow
(SREAnalyzer.group_z_peak / group_trace).

  Step 1. build_tree       : flash x flash distance -> average-linkage tree (n - 1 merges)
  Step 2. best_silhouette  : cut the tree at every h of the sweep -> mean silhouette -> best h (first of the maximum)
  Step 3. acceleration_cut : cut_rule "acceleration" only -- k of the largest M(k) - 2 M(k + 1) + M(k + 2)
                             (M(k) = merge height k -> k - 1), cut at h = M(k + 1)
  Step 4. clustering       : clusters of the cut used, renumbered 0..K-1 in first-flash order
                             (best-silhouette clusters kept too: silhouette_cut, silhouette_cluster_ids)

Example:
    >>> hc = HierarchicalClustering(distance, Z_PEAK_CUT_PX, cut_rule="acceleration")
    >>> hc.run()
    >>> hc.cluster_ids, hc.cut, hc.silhouette_cut, hc.tree, hc.sweep
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
ELBOW_K_MAX = 40              # acceleration rule looks at k = 2 .. ELBOW_K_MAX clusters


class HierarchicalClustering:
    """Tree of a flash x flash distance, cut at its best mean silhouette h (cut_rule "silhouette") or at the
    acceleration elbow (cut_rule "acceleration").

    Results:
        tree                              (build_tree; None if < 2 flashes)
        sweep, silhouette_cut {h, n_clusters, silhouette}   (best_silhouette; empty sweep if < 3 flashes)
        cut {h, n_clusters, silhouette}   (the cut used; = silhouette_cut for cut_rule "silhouette")
        cluster_ids, silhouette_cluster_ids (0..K-1 per flash)    (clustering)
    """

    def __init__(self, distance: np.ndarray, cut_heights: np.ndarray, linkage_method: str = LINKAGE_METHOD,
                 cut_rule: str = "silhouette") -> None:
        self.distance = distance
        self.cut_heights = cut_heights
        self.linkage_method = linkage_method
        self.cut_rule = cut_rule
        self.n = len(distance)
        self.tree: np.ndarray | None = None
        self.sweep = pd.DataFrame(columns=["cut_height", "n_clusters", "mean_silhouette"])
        self.cut = {"h": np.nan, "n_clusters": 0, "silhouette": np.nan}
        self.silhouette_cut = dict(self.cut)
        self.best_cut_ids: np.ndarray | None = None  # fcluster ids of the cut used, before renumbering
        self.silhouette_cut_ids: np.ndarray | None = None  # fcluster ids of the best-silhouette cut

    def run(self) -> None:
        """build_tree -> best_silhouette -> acceleration_cut (cut_rule "acceleration") -> clustering."""
        self.build_tree()
        self.best_silhouette()
        if self.cut_rule == "acceleration":
            self.acceleration_cut()
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
        self.silhouette_cut_ids = self.best_cut_ids
        self.silhouette_cut = dict(self.cut)

    def acceleration_cut(self) -> None:
        """Cut at the elbow: k of the largest accel(k) = M(k) - 2 M(k + 1) + M(k + 2), k = 2 .. ELBOW_K_MAX, at
        h = M(k + 1) (the highest merge kept inside clusters). < 4 flashes -> the best-silhouette cut stays."""
        if self.n < 4:
            return
        merge_h = self.merge_heights()
        ks = [k for k in range(2, ELBOW_K_MAX + 1) if k + 2 in merge_h]
        accel = [merge_h[k] - 2 * merge_h[k + 1] + merge_h[k + 2] for k in ks]
        k = ks[int(np.argmax(accel))]
        self.best_cut_ids = fcluster(self.tree, t=merge_h[k + 1], criterion="distance")
        self.cut["h"] = merge_h[k + 1]
        self.cut["silhouette"] = self.calculate_silhouette(self.best_cut_ids)

    def merge_heights(self) -> dict[int, float]:
        """{k: M(k)}, M(k) = h of the merge that takes k clusters -> k - 1, k = 2 .. n (average linkage: M never
        increases with k)."""
        return {k: float(self.tree[self.n - k, 2]) for k in range(2, self.n + 1)}

    def clustering(self) -> None:
        """Clusters of the cut used (and of the best-silhouette cut), renumbered 0..K-1 in first-flash order.
        0 flashes -> no clusters; no valid cut -> all flashes in one cluster."""
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
        self.cluster_ids = self.renumber(raw_ids)

        # best-silhouette clusters (= the clusters above for cut_rule "silhouette")
        silhouette_ids = raw_ids if self.silhouette_cut_ids is None else self.silhouette_cut_ids
        self.silhouette_cut["n_clusters"] = int(silhouette_ids.max()) if self.n else 0
        self.silhouette_cluster_ids = self.renumber(silhouette_ids)

    @staticmethod
    def renumber(raw_ids: np.ndarray) -> np.ndarray:
        """fcluster ids -> 0..K-1 in first-flash order."""
        new_id = {raw: new for new, raw in enumerate(pd.unique(raw_ids))}
        return np.array([new_id[raw] for raw in raw_ids], dtype=int)
