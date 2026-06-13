"""
Patch Similarity Graph Builder.

Constructs a patch similarity graph where:
  - Each patch is a node
  - Edges are weighted by similarity between patch embeddings

Supports two similarity metrics:
  1. Cosine Similarity:  A_ij = cos(φ_i, φ_j)
  2. Gaussian Kernel:    A_ij = exp(-||φ_i - φ_j||² / σ²)

The graph is built at population level by aggregating across many images,
then sparsified using k-NN to keep only the most meaningful connections.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class PatchGraphBuilder:
    """
    Build a population-level patch similarity graph.

    Pipeline:
        1. For each image, compute pairwise patch similarity: A_ij
        2. Average similarity across all images (population level)
        3. Sparsify using k-NN (keep top-k neighbors per patch)
        4. Symmetrize (undirected graph)

    Args:
        k: Number of nearest neighbors per patch for k-NN sparsification
        similarity_metric: 'cosine' or 'gaussian'
        sigma: Bandwidth parameter for Gaussian kernel (auto-estimated if None)
        threshold: Optional edge threshold (drop edges below this value)
    """

    def __init__(
        self,
        k: int = 8,
        similarity_metric: str = "cosine",
        sigma: float = None,
        threshold: float = None,
    ):
        self.k = k
        self.similarity_metric = similarity_metric
        self.sigma = sigma
        self.threshold = threshold

        # Store intermediate results for inspection
        self.raw_similarity_ = None
        self.adjacency_ = None

    # ── Similarity computation ──────────────────────────────

    def compute_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise similarity between all patches.

        Args:
            features: (B, N, D) or (N, D) patch embeddings

        Returns:
            similarity: (B, N, N) or (N, N) similarity matrix

        Where:
            Cosine:   A_ij = cos(φ_i, φ_j) = (φ_i · φ_j) / (||φ_i|| ||φ_j||)
            Gaussian: A_ij = exp(-||φ_i - φ_j||² / σ²)
        """
        single = features.dim() == 2
        if single:
            features = features.unsqueeze(0)

        if self.similarity_metric == "cosine":
            similarity = self._cosine_similarity(features)
        elif self.similarity_metric == "gaussian":
            similarity = self._gaussian_similarity(features)
        else:
            raise ValueError(
                f"Unknown metric: '{self.similarity_metric}'. "
                f"Choose 'cosine' or 'gaussian'."
            )

        if single:
            similarity = similarity.squeeze(0)

        return similarity

    def _cosine_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity: A_ij = cos(φ_i, φ_j)

        Values in [-1, 1], where 1 = identical direction.
        """
        # L2 normalize each patch embedding
        features_norm = F.normalize(features, p=2, dim=-1)
        # Pairwise cosine similarity via batch matrix multiply
        similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))
        return similarity

    def _gaussian_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Gaussian (RBF) kernel: A_ij = exp(-||φ_i - φ_j||² / σ²)

        Values in (0, 1], where 1 = identical embeddings.
        """
        B, N, D = features.shape

        # Compute pairwise squared Euclidean distances
        # ||φ_i - φ_j||² = ||φ_i||² + ||φ_j||² - 2·φ_i·φ_j
        sq_norms = (features ** 2).sum(dim=-1)  # (B, N)
        distances_sq = (
            sq_norms.unsqueeze(2) + sq_norms.unsqueeze(1)
            - 2.0 * torch.bmm(features, features.transpose(1, 2))
        )  # (B, N, N)

        # Clamp to avoid numerical issues
        distances_sq = distances_sq.clamp(min=0.0)

        # Estimate sigma if not provided
        sigma = self.sigma
        if sigma is None:
            # Use median heuristic: σ² = median of pairwise distances
            median_dist = distances_sq.median().item()
            sigma = max(np.sqrt(median_dist), 1e-6)
            self.sigma = sigma

        # Apply Gaussian kernel
        similarity = torch.exp(-distances_sq / (sigma ** 2))

        return similarity

    # ── Graph construction ──────────────────────────────────

    def build_knn_adjacency(self, similarity: torch.Tensor) -> torch.Tensor:
        """
        Sparsify similarity matrix using k-NN.

        For each patch, keep only the top-k most similar patches.
        Result is symmetrized (undirected graph).

        Args:
            similarity: (N, N) or (B, N, N) similarity matrix

        Returns:
            adjacency: same shape, binary k-NN adjacency
        """
        single = similarity.dim() == 2
        if single:
            similarity = similarity.unsqueeze(0)

        B, N, _ = similarity.shape

        # Zero out self-similarity for neighbor selection
        sim_copy = similarity.clone()
        mask = torch.eye(N, device=similarity.device).unsqueeze(0).bool()
        sim_copy = sim_copy.masked_fill(mask, float("-inf"))

        # Find top-k neighbors for each patch
        _, topk_indices = sim_copy.topk(self.k, dim=-1)  # (B, N, k)

        # Build binary adjacency
        adjacency = torch.zeros_like(similarity)
        adjacency.scatter_(2, topk_indices, 1.0)

        # Symmetrize: if i is neighbor of j OR j is neighbor of i → connected
        adjacency = torch.max(adjacency, adjacency.transpose(1, 2))

        if single:
            adjacency = adjacency.squeeze(0)

        return adjacency

    def build_population_graph(
        self,
        features: torch.Tensor,
        return_raw_similarity: bool = False,
    ) -> torch.Tensor:
        """
        Build a population-level patch graph from multiple images.

        Averages similarity across all images, then applies k-NN sparsification.

        Args:
            features: (B, N, D) patch features from B images
            return_raw_similarity: also return the raw average similarity

        Returns:
            adjacency: (N, N) weighted adjacency matrix (similarity × k-NN mask)
            raw_similarity: (N, N) if return_raw_similarity=True
        """
        B, N, D = features.shape
        print(f"  Building population graph from {B} images...")
        print(f"  Patches: {N}, Embedding dim: {D}")
        print(f"  Similarity: {self.similarity_metric}, k={self.k}")

        # Compute per-image similarities
        similarity = self.compute_similarity(features)  # (B, N, N)

        # Average across images → population-level similarity
        avg_similarity = similarity.mean(dim=0)  # (N, N)
        self.raw_similarity_ = avg_similarity.clone()

        if self.sigma is not None and self.similarity_metric == "gaussian":
            print(f"  Sigma (Gaussian bandwidth): {self.sigma:.4f}")

        # Apply optional threshold
        if self.threshold is not None:
            avg_similarity[avg_similarity < self.threshold] = 0.0
            print(f"  Threshold: {self.threshold} → {(avg_similarity > 0).sum().item()} edges")

        # k-NN sparsification
        knn_mask = self.build_knn_adjacency(avg_similarity)  # (N, N)

        # Weighted adjacency: similarity values at k-NN edges
        adjacency = knn_mask * avg_similarity
        self.adjacency_ = adjacency.clone()

        # Statistics
        n_edges = (adjacency > 0).sum().item()
        density = n_edges / (N * N)
        print(f"  Edges: {n_edges:,} ({density:.2%} density)")
        print(f"  Weight range: [{adjacency[adjacency > 0].min():.4f}, {adjacency.max():.4f}]")

        if return_raw_similarity:
            return adjacency, avg_similarity
        return adjacency

    @torch.no_grad()
    def build_from_dataloader(
        self,
        feature_extractor,
        dataloader,
        num_samples: int = 1000,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Build population graph from raw images using a feature extractor.

        Args:
            feature_extractor: model that outputs (B, N, D) patch features
            dataloader: yields (images, labels)
            num_samples: max images to process
            device: computation device

        Returns:
            adjacency: (N, N) population-level weighted adjacency
        """
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        all_similarities = []
        count = 0

        for images, _ in tqdm(dataloader, desc="Building patch graph"):
            if count >= num_samples:
                break

            images = images.to(device)
            patch_features = feature_extractor(images)  # (B, N, D)
            similarity = self.compute_similarity(patch_features)  # (B, N, N)

            all_similarities.append(similarity.cpu())
            count += images.shape[0]

        all_similarities = torch.cat(all_similarities, dim=0)
        avg_similarity = all_similarities.mean(dim=0)
        self.raw_similarity_ = avg_similarity.clone()

        # k-NN sparsification
        knn_mask = self.build_knn_adjacency(avg_similarity)
        adjacency = knn_mask * avg_similarity
        self.adjacency_ = adjacency.clone()

        return adjacency

    def build_from_saved_embeddings(self, embeddings_path: str) -> torch.Tensor:
        """
        Build graph from previously saved patch embeddings.

        Args:
            embeddings_path: path to .pt file from extract_features.py

        Returns:
            adjacency: (N, N) weighted adjacency matrix
        """
        from src.models.feature_extractor import load_embeddings
        data = load_embeddings(embeddings_path)
        features = data["embeddings"]  # (num_images, N, D)
        return self.build_population_graph(features)

    # ── Visualization ───────────────────────────────────────

    def visualize(
        self,
        adjacency: torch.Tensor = None,
        raw_similarity: torch.Tensor = None,
        save_dir: str = None,
        grid_size: int = 14,
    ):
        """
        Generate comprehensive visualizations of the patch similarity graph.

        Creates:
        1. Raw similarity heatmap (before k-NN)
        2. k-NN adjacency heatmap (after sparsification)
        3. Per-patch degree distribution
        4. Spatial connectivity pattern on 14×14 grid
        """
        if adjacency is None:
            adjacency = self.adjacency_
        if raw_similarity is None:
            raw_similarity = self.raw_similarity_
        if adjacency is None:
            raise ValueError("No adjacency matrix. Run build_population_graph() first.")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        A = adjacency.numpy() if isinstance(adjacency, torch.Tensor) else adjacency

        # ── Figure 1: Similarity & Adjacency Heatmaps ──
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Raw similarity
        if raw_similarity is not None:
            S = raw_similarity.numpy() if isinstance(raw_similarity, torch.Tensor) else raw_similarity
            im0 = axes[0].imshow(S, cmap="viridis", aspect="auto")
            axes[0].set_title(f"Raw {self.similarity_metric.title()} Similarity\n(population average)",
                            fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Patch j")
            axes[0].set_ylabel("Patch i")
            plt.colorbar(im0, ax=axes[0], shrink=0.8)
        else:
            axes[0].text(0.5, 0.5, "No raw similarity", ha="center", va="center")

        # k-NN adjacency
        im1 = axes[1].imshow(A, cmap="hot", aspect="auto")
        axes[1].set_title(f"k-NN Adjacency (k={self.k})\n{(A > 0).sum()} edges",
                        fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Patch j")
        axes[1].set_ylabel("Patch i")
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Binary structure
        binary = (A > 0).astype(float)
        im2 = axes[2].imshow(binary, cmap="Greys", aspect="auto")
        axes[2].set_title(f"Binary Connectivity\ndensity={binary.mean():.2%}",
                        fontsize=11, fontweight="bold")
        axes[2].set_xlabel("Patch j")
        axes[2].set_ylabel("Patch i")
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        plt.suptitle(f"Patch Similarity Graph — A_ij ({self.similarity_metric})",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, "similarity_heatmaps.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close(fig)

        # ── Figure 2: Degree Distribution & Spatial Pattern ──
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        # Degree histogram
        degrees = (A > 0).sum(axis=1)
        axes2[0].hist(degrees, bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
        axes2[0].set_xlabel("Node Degree (# neighbors)", fontsize=11)
        axes2[0].set_ylabel("Count", fontsize=11)
        axes2[0].set_title(f"Degree Distribution\nmean={degrees.mean():.1f}, max={degrees.max()}",
                          fontsize=11, fontweight="bold")
        axes2[0].axvline(degrees.mean(), color="red", linestyle="--", label=f"mean={degrees.mean():.1f}")
        axes2[0].legend()

        # Degree spatial map
        degree_grid = degrees.reshape(grid_size, grid_size)
        im3 = axes2[1].imshow(degree_grid, cmap="YlOrRd", interpolation="nearest")
        axes2[1].set_title("Degree per Patch Position", fontsize=11, fontweight="bold")
        plt.colorbar(im3, ax=axes2[1], shrink=0.8, label="Degree")

        # Mean edge weight spatial map
        weight_sums = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            neighbors = A[i] > 0
            if neighbors.any():
                weight_sums[i] = A[i, neighbors].mean()
        weight_grid = weight_sums.reshape(grid_size, grid_size)
        im4 = axes2[2].imshow(weight_grid, cmap="plasma", interpolation="bilinear")
        axes2[2].set_title("Mean Edge Weight per Patch", fontsize=11, fontweight="bold")
        plt.colorbar(im4, ax=axes2[2], shrink=0.8, label="Avg weight")

        plt.suptitle("Graph Topology Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, "graph_topology.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close(fig2)

        # ── Figure 3: Sample patch connectivity on grid ──
        fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
        axes3 = axes3.flatten()

        # Show connectivity for 8 sample patches spread across the grid
        sample_patches = [0, 27, 55, 97, 98, 140, 168, 195]
        for idx, patch_id in enumerate(sample_patches):
            if patch_id >= A.shape[0]:
                continue

            # Create a grid showing this patch's connections
            conn_map = np.zeros(A.shape[0])
            conn_map[patch_id] = 1.0  # mark the query patch
            conn_map[A[patch_id] > 0] = 0.5  # mark its neighbors
            conn_grid = conn_map.reshape(grid_size, grid_size)

            row, col = patch_id // grid_size, patch_id % grid_size
            axes3[idx].imshow(conn_grid, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
            axes3[idx].set_title(f"Patch ({row},{col}) [idx={patch_id}]\n{int((A[patch_id] > 0).sum())} neighbors",
                               fontsize=9, fontweight="bold")

            # Add grid
            axes3[idx].set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            axes3[idx].set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            axes3[idx].grid(which="minor", color="white", linewidth=0.3)
            axes3[idx].tick_params(which="minor", size=0)
            axes3[idx].set_xticks([])
            axes3[idx].set_yticks([])

        plt.suptitle("Sample Patch Connectivity (green=query, yellow=neighbors)",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, "patch_connectivity.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  ✓ Saved: {path}")
        plt.close(fig3)


def save_graph(adjacency: torch.Tensor, raw_similarity: torch.Tensor = None,
               metadata: dict = None, save_path: str = "outputs/patch_graph.pt"):
    """Save the adjacency matrix and metadata to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "adjacency": adjacency,
        "shape": list(adjacency.shape),
    }
    if raw_similarity is not None:
        data["raw_similarity"] = raw_similarity
    if metadata is not None:
        data["metadata"] = metadata
    torch.save(data, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✓ Saved graph: {save_path} ({size_mb:.1f} MB)")


def load_graph(load_path: str) -> dict:
    """Load a saved adjacency matrix."""
    data = torch.load(load_path, map_location="cpu")
    print(f"  ✓ Loaded graph: {load_path}, shape={data['shape']}")
    return data
