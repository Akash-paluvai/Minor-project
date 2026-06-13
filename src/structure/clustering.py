"""
Unsupervised Clustering for Patch Structure Discovery.

Implements 4 clustering algorithms to identify structural groups of patches:
  1. K-Means Clustering — on raw patch embeddings
  2. Spectral Clustering — on adjacency/similarity matrix
  3. Diffusion Maps + K-Means — nonlinear dimensionality reduction on graph
  4. Graph Neural Network Clustering — learnable GNN-based grouping

Each algorithm:
  - Accepts patch embeddings φ ∈ R^(N×D) OR adjacency matrix A ∈ R^(N×N)
  - Outputs cluster assignments: cluster_i ∈ {1, 2, ..., k}
  - Returns labels array of size N

Usage:
    labels = kmeans_clustering(embeddings, k=16)
    labels = spectral_clustering(adj_matrix, k=16)
    labels = diffusion_map_clustering(adj_matrix, k=16)
    labels = gnn_clustering(adj_matrix, embeddings, k=16)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering as SklearnSpectral
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ════════════════════════════════════════════════════════════
#  1. K-Means Clustering
# ════════════════════════════════════════════════════════════

def kmeans_clustering(
    embeddings: np.ndarray,
    k: int = 16,
    random_state: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> np.ndarray:
    """
    K-Means clustering on raw patch embeddings.

    Groups patches based on feature similarity in embedding space.
    Best for well-separated, spherical clusters.

    Args:
        embeddings: (N, D) patch embeddings φ_i ∈ R^D
        k: Number of clusters
        random_state: Random seed
        n_init: Number of K-Means initializations
        max_iter: Max iterations per run

    Returns:
        labels: (N,) cluster assignments, each in {0, 1, ..., k-1}
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # L2 normalize embeddings for better clustering
    embeddings_norm = normalize(embeddings, norm="l2", axis=1)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    labels = kmeans.fit_predict(embeddings_norm)

    print(f"  K-Means: {k} clusters, inertia={kmeans.inertia_:.2f}")
    print(f"  Cluster sizes: {np.bincount(labels)}")

    return labels


# ════════════════════════════════════════════════════════════
#  2. Spectral Clustering
# ════════════════════════════════════════════════════════════

def spectral_clustering(
    adj_matrix: np.ndarray,
    k: int = 16,
    random_state: int = 42,
    use_sklearn: bool = True,
) -> np.ndarray:
    """
    Spectral Clustering on the patch similarity graph.

    Uses the graph Laplacian eigendecomposition to embed patches
    into a spectral space, then clusters with K-Means.

    Algorithm:
        1. Compute normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
        2. Find k smallest eigenvectors of L_sym
        3. Row-normalize eigenvector matrix
        4. Apply K-Means on spectral embedding

    Args:
        adj_matrix: (N, N) adjacency/similarity matrix (symmetric, non-negative)
        k: Number of clusters
        random_state: Random seed
        use_sklearn: Use sklearn implementation (True) or custom (False)

    Returns:
        labels: (N,) cluster assignments
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    # Ensure symmetric and non-negative
    adj = np.maximum(adj_matrix, 0)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)

    if use_sklearn:
        labels = _spectral_sklearn(adj, k, random_state)
    else:
        labels = _spectral_custom(adj, k, random_state)

    print(f"  Spectral: {k} clusters")
    print(f"  Cluster sizes: {np.bincount(labels)}")

    return labels


def _spectral_sklearn(adj: np.ndarray, k: int, random_state: int) -> np.ndarray:
    """Spectral clustering via sklearn."""
    sc = SklearnSpectral(
        n_clusters=k,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
        n_init=10,
    )
    return sc.fit_predict(adj)


def _spectral_custom(adj: np.ndarray, k: int, random_state: int) -> np.ndarray:
    """Custom spectral clustering via scipy eigendecomposition."""
    N = adj.shape[0]

    # Degree matrix
    degrees = adj.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)

    # Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_sym = np.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) / 2  # numerical symmetry

    # Smallest k+1 eigenvectors
    n_components = min(k + 1, N - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_sym, k=n_components, which="SM", maxiter=1000)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        eigenvectors = eigenvectors[:, :n_components]

    # Skip trivial eigenvector, take k
    embedding = eigenvectors[:, 1:k + 1]

    # Row-normalize
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.maximum(norms, 1e-10)

    # K-Means on spectral embedding
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return kmeans.fit_predict(embedding)


# ════════════════════════════════════════════════════════════
#  3. Diffusion Maps + K-Means
# ════════════════════════════════════════════════════════════

def diffusion_map_clustering(
    adj_matrix: np.ndarray,
    k: int = 16,
    n_diffusion_components: int = 20,
    diffusion_time: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Diffusion Maps followed by K-Means clustering.

    Diffusion maps capture multi-scale geometry of the data manifold
    by analyzing random walks on the similarity graph.

    Algorithm:
        1. Build Markov transition matrix: P = D^{-1} A
        2. Compute P^t (diffusion at time t)
        3. Eigendecompose P → diffusion coordinates
        4. Apply K-Means on diffusion embedding

    Args:
        adj_matrix: (N, N) adjacency/similarity matrix
        k: Number of clusters
        n_diffusion_components: Dimensions of diffusion embedding
        diffusion_time: Number of diffusion steps (t)
        random_state: Random seed

    Returns:
        labels: (N,) cluster assignments
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    adj = np.maximum(adj_matrix, 0)
    adj = (adj + adj.T) / 2
    np.fill_diagonal(adj, 0)
    N = adj.shape[0]

    # Row-normalize → Markov transition matrix P
    row_sums = adj.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-10)
    P = adj / row_sums[:, np.newaxis]

    # Eigendecompose P
    n_comp = min(n_diffusion_components + 1, N - 1)
    try:
        eigenvalues, eigenvectors = eigsh(P, k=n_comp, which="LM", maxiter=1000)
    except Exception:
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        idx = np.argsort(-eigenvalues)[:n_comp]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Sort by eigenvalue magnitude (descending)
    sort_idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Diffusion coordinates: ψ_j(x) = λ_j^t · φ_j(x)
    # Skip first eigenvector (trivial constant)
    n_keep = min(n_diffusion_components, eigenvectors.shape[1] - 1)
    diffusion_coords = eigenvectors[:, 1:n_keep + 1]
    diffusion_eigenvalues = eigenvalues[1:n_keep + 1]

    # Scale by eigenvalue^t (diffusion time)
    for j in range(diffusion_coords.shape[1]):
        diffusion_coords[:, j] *= np.abs(diffusion_eigenvalues[j]) ** diffusion_time

    # K-Means on diffusion embedding
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(diffusion_coords)

    print(f"  Diffusion Maps: {k} clusters, t={diffusion_time}, "
          f"components={diffusion_coords.shape[1]}")
    print(f"  Top eigenvalues: {eigenvalues[:5].round(4)}")
    print(f"  Cluster sizes: {np.bincount(labels)}")

    return labels


# ════════════════════════════════════════════════════════════
#  4. GNN Clustering (Graph Neural Network)
# ════════════════════════════════════════════════════════════

class GNNClusterNet(nn.Module):
    """
    Simple GNN for unsupervised clustering via graph convolution.

    Architecture:
        GCN Layer 1 → ReLU → GCN Layer 2 → Softmax → Cluster probabilities

    The GCN layers perform message passing:
        H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})

    where Ã = A + I (self-loop), D̃ = degree matrix of Ã.

    Trained with a modularity-inspired loss that encourages dense
    intra-cluster and sparse inter-cluster connectivity.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, k: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, k)
        self.k = k

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (N, D) node features
            A_norm: (N, N) normalized adjacency with self-loops

        Returns:
            cluster_probs: (N, k) soft cluster assignments
        """
        # GCN layer 1: message passing + nonlinearity
        H = A_norm @ X
        H = self.fc1(H)
        H = F.relu(H)

        # GCN layer 2: produce cluster logits
        H = A_norm @ H
        H = self.fc2(H)

        # Soft cluster assignments
        cluster_probs = F.softmax(H, dim=-1)
        return cluster_probs


def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute D̃^{-1/2} Ã D̃^{-1/2} where Ã = A + I.
    """
    N = adj.shape[0]
    A_tilde = adj + torch.eye(N, device=adj.device)
    degrees = A_tilde.sum(dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees.clamp(min=1e-8)))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


def _modularity_loss(cluster_probs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Modularity-based loss for unsupervised GNN clustering.

    Encourages hard assignments and high modularity.

    L = -Tr(S^T B S) / (2m) + entropy_reg

    where S = cluster assignment matrix, B = modularity matrix.
    """
    N = adj.shape[0]
    S = cluster_probs  # (N, k)

    # Degree
    d = adj.sum(dim=1)  # (N,)
    m = adj.sum() / 2

    # Modularity matrix: B_ij = A_ij - d_i * d_j / (2m)
    B = adj - torch.outer(d, d) / (2 * m + 1e-8)

    # Modularity: Tr(S^T B S)
    mod = torch.trace(S.T @ B @ S) / (2 * m + 1e-8)

    # Entropy regularization: encourage hard assignments
    entropy = -(S * torch.log(S + 1e-8)).sum() / N

    # Collapse regularization: prevent all nodes in one cluster
    cluster_sizes = S.sum(dim=0)  # (k,)
    collapse_reg = -torch.log(cluster_sizes + 1e-8).sum() / S.shape[1]

    loss = -mod + 0.5 * entropy + 0.1 * collapse_reg
    return loss


def gnn_clustering(
    adj_matrix: np.ndarray,
    embeddings: np.ndarray,
    k: int = 16,
    hidden_dim: int = 128,
    epochs: int = 200,
    lr: float = 0.01,
    random_state: int = 42,
) -> np.ndarray:
    """
    GNN-based unsupervised clustering.

    Trains a 2-layer Graph Convolutional Network with a modularity loss
    to discover clusters that respect the graph structure.

    Args:
        adj_matrix: (N, N) adjacency/similarity matrix
        embeddings: (N, D) patch embeddings as node features
        k: Number of clusters
        hidden_dim: GCN hidden dimension
        epochs: Training epochs
        lr: Learning rate
        random_state: Random seed

    Returns:
        labels: (N,) cluster assignments (hard, from argmax)
    """
    torch.manual_seed(random_state)

    if isinstance(adj_matrix, torch.Tensor):
        adj = adj_matrix.float()
    else:
        adj = torch.tensor(adj_matrix, dtype=torch.float32)

    if isinstance(embeddings, torch.Tensor):
        X = embeddings.float()
    else:
        X = torch.tensor(embeddings, dtype=torch.float32)

    N, D = X.shape

    # Normalize adjacency
    A_norm = _normalize_adjacency(adj)

    # Build model
    model = GNNClusterNet(input_dim=D, hidden_dim=hidden_dim, k=k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    best_loss = float("inf")
    best_labels = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        cluster_probs = model(X, A_norm)
        loss = _modularity_loss(cluster_probs, adj)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_labels = cluster_probs.argmax(dim=-1).detach().cpu().numpy()

        if (epoch + 1) % 50 == 0:
            print(f"    GNN epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    labels = best_labels if best_labels is not None else cluster_probs.argmax(dim=-1).detach().cpu().numpy()

    print(f"  GNN Clustering: {k} clusters, best_loss={best_loss:.4f}")
    print(f"  Cluster sizes: {np.bincount(labels, minlength=k)}")

    return labels


# ════════════════════════════════════════════════════════════
#  Multi-Algorithm Runner
# ════════════════════════════════════════════════════════════

def run_all_clustering(
    embeddings: np.ndarray,
    adj_matrix: np.ndarray,
    k: int = 16,
    random_state: int = 42,
) -> dict:
    """
    Run all 4 clustering algorithms and return results.

    Args:
        embeddings: (N, D) patch embeddings
        adj_matrix: (N, N) adjacency matrix
        k: Number of clusters

    Returns:
        dict mapping algorithm name → cluster labels (N,)
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    results = {}

    print("\n[1/4] K-Means Clustering")
    results["kmeans"] = kmeans_clustering(embeddings, k=k, random_state=random_state)

    print("\n[2/4] Spectral Clustering")
    results["spectral"] = spectral_clustering(adj_matrix, k=k, random_state=random_state)

    print("\n[3/4] Diffusion Maps + K-Means")
    results["diffusion"] = diffusion_map_clustering(adj_matrix, k=k, random_state=random_state)

    print("\n[4/4] GNN Clustering")
    results["gnn"] = gnn_clustering(adj_matrix, embeddings, k=k, random_state=random_state)

    return results


# ════════════════════════════════════════════════════════════
#  Visualization
# ════════════════════════════════════════════════════════════

def visualize_clusters(
    clusters: np.ndarray,
    grid_size: int = 14,
    title: str = "Cluster Assignments",
    save_path: str = None,
    original_image: np.ndarray = None,
    patch_size: int = 16,
):
    """
    Visualize cluster assignments on the patch grid and optionally
    overlay on the original image.

    Args:
        clusters: (N,) cluster labels
        grid_size: Patch grid size (14 for 224/16)
        title: Plot title
        save_path: Path to save figure
        original_image: (H, W, 3) original image for overlay
        patch_size: Patch size in pixels
    """
    k = len(np.unique(clusters))
    cluster_grid = clusters.reshape(grid_size, grid_size)

    # Generate colormap
    cmap = plt.cm.get_cmap("tab20", k)

    if original_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Cluster grid
    im0 = axes[0].imshow(cluster_grid, cmap=cmap, interpolation="nearest")
    axes[0].set_title(f"{title}\n({k} clusters)", fontsize=12, fontweight="bold")
    for i in range(grid_size):
        for j in range(grid_size):
            axes[0].text(j, i, str(cluster_grid[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cluster_grid[i, j] % 2 == 0 else "black")
    axes[0].set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    axes[0].set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    axes[0].grid(which="minor", color="white", linewidth=0.5)
    axes[0].tick_params(which="minor", size=0)
    plt.colorbar(im0, ax=axes[0], shrink=0.8, label="Cluster ID")

    # Plot 2: Cluster heatmap (smoothed)
    from scipy.ndimage import zoom
    cluster_upscaled = zoom(cluster_grid.astype(float), patch_size, order=0)
    axes[1].imshow(cluster_upscaled, cmap=cmap, interpolation="nearest")
    axes[1].set_title("Cluster Map (upscaled to image size)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Plot 3: Overlay on original image
    if original_image is not None:
        axes[2].imshow(original_image)
        # Create RGBA overlay
        overlay = np.zeros((*cluster_upscaled.shape, 4))
        for c in range(k):
            mask = cluster_upscaled == c
            color = cmap(c)
            overlay[mask] = (*color[:3], 0.35)

        overlay_resized = overlay[:original_image.shape[0], :original_image.shape[1]]
        axes[2].imshow(overlay_resized)
        axes[2].set_title("Cluster Overlay on X-ray", fontsize=12, fontweight="bold")
        axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close(fig)


def compare_clusterings(
    results: dict,
    grid_size: int = 14,
    save_path: str = None,
):
    """
    Compare all clustering algorithms side by side.

    Args:
        results: dict mapping algorithm_name → labels (N,)
        grid_size: Patch grid side length
        save_path: Save path for comparison figure
    """
    n_methods = len(results)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 10))

    for idx, (name, labels) in enumerate(results.items()):
        k = len(np.unique(labels))
        cmap = plt.cm.get_cmap("tab20", k)
        cluster_grid = labels.reshape(grid_size, grid_size)

        # Row 1: Cluster grid
        im = axes[0, idx].imshow(cluster_grid, cmap=cmap, interpolation="nearest")
        axes[0, idx].set_title(
            f"{name.upper()}\n{k} clusters",
            fontsize=12, fontweight="bold"
        )
        for i in range(grid_size):
            for j in range(grid_size):
                axes[0, idx].text(j, i, str(cluster_grid[i, j]),
                                 ha="center", va="center", fontsize=5,
                                 color="white")
        axes[0, idx].set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        axes[0, idx].set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        axes[0, idx].grid(which="minor", color="white", linewidth=0.3)
        axes[0, idx].tick_params(which="minor", size=0)

        # Row 2: Cluster size distribution
        sizes = np.bincount(labels, minlength=k)
        bars = axes[1, idx].bar(range(k), sizes, color=[cmap(i) for i in range(k)])
        axes[1, idx].set_xlabel("Cluster ID", fontsize=10)
        axes[1, idx].set_ylabel("# Patches", fontsize=10)
        axes[1, idx].set_title(f"Cluster Sizes", fontsize=11, fontweight="bold")
        axes[1, idx].set_xticks(range(0, k, max(1, k // 8)))

    plt.suptitle(
        "Clustering Algorithm Comparison",
        fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved comparison: {save_path}")
    plt.close(fig)


def visualize_graph_clusters(
    adj_matrix: np.ndarray,
    labels: np.ndarray,
    grid_size: int = 14,
    title: str = "Graph with Cluster Colors",
    save_path: str = None,
):
    """
    Visualize the patch graph with cluster-colored nodes.

    Draws patches as nodes in a grid layout, colored by cluster,
    with edges from the adjacency matrix.
    """
    import networkx as nx

    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.cpu().numpy()

    N = adj_matrix.shape[0]
    k = len(np.unique(labels))
    cmap = plt.cm.get_cmap("tab20", k)

    # Create graph
    G = nx.Graph()
    for i in range(N):
        row, col = i // grid_size, i % grid_size
        G.add_node(i, pos=(col, -row))  # grid layout

    # Add edges (subsample for clarity)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj_matrix[i, j] > 0:
                edges.append((i, j, adj_matrix[i, j]))

    # Limit edges for readability
    if len(edges) > 500:
        edges = sorted(edges, key=lambda x: -x[2])[:500]

    for i, j, w in edges:
        G.add_edge(i, j, weight=w)

    # Draw
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pos = nx.get_node_attributes(G, "pos")
    node_colors = [cmap(labels[i]) for i in range(N)]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=80, node_color=node_colors,
        edgecolors="white", linewidths=0.5
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[0.5 + 2 * w / max_w for w in edge_weights],
        alpha=0.2, edge_color="gray"
    )

    ax.set_title(f"{title} ({k} clusters, {len(edges)} edges)",
                fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {save_path}")
    plt.close(fig)
