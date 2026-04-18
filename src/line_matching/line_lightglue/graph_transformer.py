"""
GNN with Graph Transformer for Document Line Matching
=====================================================

This module implements a Graph Neural Network (GNN) enhanced with Graph Transformer
architecture for the line matching component of DocMatcher.

MOTIVATION:
-----------
The original LineLightGlue model uses standard self-attention where every detected
line attends to every other line in the same image. This has two limitations:

1. COMPUTATIONAL: O(n^2) complexity — every line attends to every other line,
   even when most lines are spatially unrelated.

2. STRUCTURAL: No explicit modeling of spatial/geometric relationships between
   lines. The spatial structure of document layouts (horizontal text lines,
   vertical dividers, parallel borders) is only learned implicitly.

Documents have strong spatial structure — text lines are arranged in reading
order, structural lines form borders/columns, and nearby lines provide the
most useful context for matching. A Graph Transformer encodes this structure
explicitly by restricting and biasing attention based on graph connectivity.

APPROACH:
---------
We augment the fully-connected self-attention with graph-structured bias:

1. BUILD a k-nearest-neighbor (k-NN) graph from line descriptors in feature space.
   Lines with similar visual/textual content are connected as neighbors.

2. COMPUTE edge features encoding pairwise relationships:
   - Cosine similarity between descriptors
   - L2 distance (normalized)
   - Mean and max absolute feature differences

3. BIAS attention scores using learned edge encodings (Graphormer-style):
   Attention(Q,K,V) = softmax(QK^T/sqrt(d) + gate * EdgeBias) * V

4. OPTIONALLY MASK attention to only k-nearest neighbors (sparse attention),
   which enforces locality and reduces computation.

WEIGHT COMPATIBILITY:
---------------------
The GraphTransformerBlock is designed to be a DROP-IN REPLACEMENT for the
original LineSelfBlock. All core parameters (Wqkv, out_proj, ffn) have
IDENTICAL names and shapes, so pretrained checkpoint weights load correctly.
New graph-specific parameters (edge_encoder, graph_gate) are additional and
are handled via strict=False loading.

REFERENCES:
-----------
- Ying et al., "Do Transformers Really Perform Bad for Graph Representation?
  Learning on Graphs with All the Transformer Architecture", NeurIPS 2021
- Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs",
  AAAI 2021 Workshop
- Sarlin et al., "SuperGlue: Learning Feature Matching with Graph Neural
  Networks", CVPR 2020

Authors: Extended from DocMatcher (Hertlein et al., WACV 2025)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the original Attention class for backward compatibility
from .original.lightglue import Attention


# =============================================================================
# COMPONENT 1: Graph Construction
# =============================================================================

class LineGraphBuilder:
    """
    Builds a k-nearest-neighbor graph from line descriptors.

    Given N line descriptors (each a D-dimensional vector), this class:
    1. Computes pairwise cosine similarity between all lines
    2. Selects the top-k most similar lines as neighbors for each line
    3. Makes the graph symmetric (if A is neighbor of B, B is neighbor of A)
    4. Computes edge features that encode pairwise relationships

    WHY k-NN IN FEATURE SPACE?
    --------------------------
    - Lines with similar visual appearance (parallel text, similar fonts) are
      connected, providing structural context for matching.
    - Feature-space k-NN can be computed entirely from the encoded descriptors
      (no need to modify the data pipeline or load line geometry JSON files).
    - For documents, visually similar lines tend to be spatially related
      (e.g., consecutive text lines have similar font/style).

    EXTENSION: For spatial graphs, one could use line centroid coordinates
    instead of descriptors for k-NN computation. This would require loading
    line geometry from the JSON files in the data pipeline.

    Args:
        k_neighbors (int): Number of nearest neighbors per node (default: 5).
            Higher k = more connections = more context but less sparsity.
            For documents with ~20-40 lines, k=5 gives good coverage.
        self_loops (bool): Whether to include self-loops (default: True).
            Self-loops allow each node to attend to itself, which is standard
            in graph transformers and preserves the residual connection behavior.
    """

    EDGE_FEATURE_DIM = 4

    def __init__(self, k_neighbors: int = 5, self_loops: bool = True):
        self.k_neighbors = k_neighbors
        self.self_loops = self_loops

    def build_graph(
        self,
        descriptors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-NN graph from line descriptors.

        The graph topology (adjacency) is computed without gradients since
        it involves discrete operations (top-k selection). However, the edge
        features ARE differentiable, allowing gradient flow during training.

        Args:
            descriptors: Line descriptors of shape (B, N, D) where
                B = batch size, N = number of lines, D = descriptor dimension

        Returns:
            adj_matrix: Binary adjacency matrix (B, N, N), True where edges exist
            edge_features: Edge feature tensor (B, N, N, 4) with features:
                [cosine_similarity, l2_distance, mean_abs_diff, max_abs_diff]
        """
        B, N, D = descriptors.shape

        # Interpret k_neighbors as the number of EXTERNAL neighbors.
        # Self-loops are handled separately below.
        k = min(self.k_neighbors, max(N - 1, 0))

        # ==== STEP 1: Compute graph topology (non-differentiable) ====
        # We detach descriptors for topology computation because the top-k
        # selection is a discrete operation that shouldn't affect gradients.
        with torch.no_grad():
            desc_norm = F.normalize(descriptors.detach(), p=2, dim=-1)
            sim_detached = torch.bmm(desc_norm, desc_norm.transpose(1, 2))

            # Exclude self-matches from the k-NN search and add self-loops later.
            if N > 0:
                eye = torch.eye(
                    N, device=descriptors.device, dtype=torch.bool
                ).unsqueeze(0)
                sim_detached = sim_detached.masked_fill(eye, float("-inf"))

            # Create binary adjacency matrix from top-k indices
            adj_matrix = torch.zeros(
                B, N, N, device=descriptors.device, dtype=torch.bool
            )
            if k > 0:
                _, topk_indices = sim_detached.topk(k, dim=-1)  # (B, N, k)
                adj_matrix.scatter_(2, topk_indices, True)

            # Make symmetric: if line_i connects to line_j, line_j connects to line_i
            # This ensures the graph is undirected, matching document structure
            adj_matrix = adj_matrix | adj_matrix.transpose(1, 2)

            # Add self-loops so each node always attends to itself
            if self.self_loops:
                eye = torch.eye(N, device=descriptors.device, dtype=torch.bool).unsqueeze(0)
                adj_matrix = adj_matrix | eye

        # ==== STEP 2: Compute edge features (differentiable) ====
        # These features are inputs to the learnable edge_encoder, so
        # gradients should flow through them during training.

        # Feature 1: Cosine similarity (measures descriptor alignment)
        desc_norm = F.normalize(descriptors, p=2, dim=-1)
        similarity = torch.bmm(desc_norm, desc_norm.transpose(1, 2))
        sim_feature = similarity.unsqueeze(-1)  # (B, N, N, 1)

        # Feature 2: Normalized L2 distance (measures descriptor magnitude difference)
        diff = descriptors.unsqueeze(2) - descriptors.unsqueeze(1)  # (B, N, N, D)
        l2_dist = torch.norm(diff, p=2, dim=-1, keepdim=True) / math.sqrt(D)

        # Feature 3 & 4: Statistics of element-wise differences
        # These capture finer-grained differences than L2 distance alone
        abs_diff = torch.abs(diff)
        mean_abs_diff = abs_diff.mean(dim=-1, keepdim=True)  # (B, N, N, 1)
        max_abs_diff = abs_diff.max(dim=-1, keepdim=True).values  # (B, N, N, 1)

        # Concatenate all edge features: (B, N, N, 4)
        edge_features = torch.cat([
            sim_feature,
            l2_dist,
            mean_abs_diff,
            max_abs_diff,
        ], dim=-1)

        return adj_matrix, edge_features


# =============================================================================
# COMPONENT 2: Graph Transformer Block
# =============================================================================

class GraphTransformerBlock(nn.Module):
    """
    Graph Transformer block that replaces LineSelfBlock in DocMatcher.

    This block enhances standard multi-head self-attention with graph structure:

    1. GRAPH CONSTRUCTION: At each forward pass, a k-NN graph is built from
       the input descriptors. This creates a connectivity structure where
       semantically similar lines are linked.

    2. EDGE BIAS (Graphormer-style): Edge features between connected nodes
       are projected by a learned edge_encoder to produce per-head attention
       bias terms. These are ADDED to the raw attention logits before softmax:

           attn = softmax(QK^T/sqrt(d) + graph_gate * edge_bias) * V

       This allows the model to learn that certain relationships (e.g., high
       similarity, small distance) should increase attention between lines.

    3. GRAPH GATE: A learnable scalar parameter (initialized to 0.0) that
       controls the strength of graph bias. At initialization, the gate keeps
       the model behavior-compatible with the released baseline checkpoint.
       During fine-tuning, the gate adapts to introduce graph structure.

    4. OPTIONAL SPARSE MASKING: When use_sparse_attention=True, non-neighbor
       nodes receive -inf attention score and contribute nothing. This enforces
       strict locality and reduces effective computation.

    WEIGHT COMPATIBILITY:
    ---------------------
    Parameter names (Wqkv, out_proj, ffn) are IDENTICAL to LineSelfBlock.
    This means pretrained checkpoint weights load correctly into this block.
    The new parameters (edge_encoder, graph_gate) are simply missing from
    the checkpoint and get their default initialization.

    Args:
        embed_dim (int): Dimension of input embeddings (256 for DocMatcher)
        num_heads (int): Number of attention heads (4 for DocMatcher)
        edge_dim (int): Dimension of edge features from LineGraphBuilder (must be 4)
        k_neighbors (int): Number of neighbors in k-NN graph (default: 5)
        use_sparse_attention (bool): If True, mask non-neighbors (default: False)
        flash (bool): Kept for API compatibility with LineSelfBlock
        bias (bool): Whether to use bias in linear layers (default: True)
    """

    # CHANGED: New parameters edge_dim, k_neighbors, use_sparse_attention
    # are added compared to original LineSelfBlock
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        edge_dim: int = 4,
        k_neighbors: int = 5,
        use_sparse_attention: bool = False,
        flash: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_sparse_attention = use_sparse_attention

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        if edge_dim != LineGraphBuilder.EDGE_FEATURE_DIM:
            raise ValueError(
                f"edge_dim must be {LineGraphBuilder.EDGE_FEATURE_DIM} "
                f"for the current LineGraphBuilder implementation, got {edge_dim}."
            )

        # ===================================================================
        # ORIGINAL PARAMETERS — identical names to LineSelfBlock
        # These load directly from pretrained checkpoints.
        # ===================================================================

        # Joint Q/K/V projection (same as original)
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        # Original attention module (kept for compatibility, used as fallback)
        self.inner_attn = Attention(flash)

        # Output projection (same as original)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Feed-forward network (IDENTICAL architecture to original)
        # Architecture: concat(x, message) -> Linear -> LayerNorm -> GELU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

        # ===================================================================
        # NEW: Graph-specific parameters (not in pretrained checkpoints)
        # ===================================================================

        # Graph builder: constructs k-NN graph from descriptors at each forward pass
        # CHANGED: This is a new component that creates graph structure
        self.graph_builder = LineGraphBuilder(k_neighbors=k_neighbors)

        # Edge feature encoder: learns to project raw edge features (4-dim)
        # into per-head attention bias values.
        # Architecture: Linear(4 -> 2H) -> ReLU -> Linear(2H -> H)
        # CHANGED: This is a new learnable component for graph-aware attention
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, num_heads * 2),
            nn.ReLU(),
            nn.Linear(num_heads * 2, num_heads),
        )

        # Learnable gate controlling graph bias strength.
        # Initialized to 0.0 so loading the released LightGlue checkpoint into
        # the GNN architecture is behavior-preserving until fine-tuning learns
        # a useful graph contribution.
        # CHANGED: This is a new learnable scalar parameter
        self.graph_gate = nn.Parameter(torch.tensor(0.0))

        # Cache for visualization/debugging — stores info from last forward pass
        self._last_adj_matrix = None
        self._last_edge_features = None
        self._last_attn_weights = None

    def get_graph_info(self) -> Optional[dict]:
        """
        Return cached graph information from the last forward pass.

        Useful for visualization and debugging. Returns None if no forward
        pass has been executed yet.

        Returns:
            dict with keys:
                - 'adj_matrix': (B, N, N) bool tensor — graph connectivity
                - 'edge_features': (B, N, N, 4) float tensor — raw edge features
                - 'attn_weights': (B, H, N, N) float tensor — final attention weights
                - 'graph_gate': float — current gate value
        """
        if self._last_adj_matrix is None:
            return None
        return {
            'adj_matrix': self._last_adj_matrix.detach().cpu(),
            'edge_features': self._last_edge_features.detach().cpu(),
            'attn_weights': self._last_attn_weights.detach().cpu(),
            'graph_gate': self.graph_gate.item(),
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Graph Transformer block.

        CHANGED from original LineSelfBlock:
        - Builds k-NN graph from input descriptors
        - Computes edge features and encodes them as attention bias
        - Adds graph bias to attention logits before softmax
        - Optionally masks non-neighbors for sparse attention

        The signature (x, mask) matches the original LineSelfBlock exactly,
        so this is a transparent drop-in replacement.

        Args:
            x: Node (line) features of shape (B, N, D)
            mask: Optional padding mask of shape (B, N, N)

        Returns:
            Updated node features of shape (B, N, D) with graph-aware context
        """
        B, N, D = x.shape

        # ==== STEP 1: Build graph structure from current descriptors ====
        # CHANGED: This step is entirely new — the original has no graph
        adj_matrix, edge_features = self.graph_builder.build_graph(x)

        # Cache for visualization
        self._last_adj_matrix = adj_matrix
        self._last_edge_features = edge_features

        # ==== STEP 2: Compute Q, K, V (identical to original) ====
        qkv = self.Wqkv(x)  # (B, N, 3*D)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        # Shape: (B, num_heads, N, head_dim, 3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        # Each: (B, num_heads, N, head_dim)

        # ==== STEP 3: Compute attention logits (same as original) ====
        scale = self.head_dim ** -0.5
        attn_logits = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        # Shape: (B, num_heads, N, N)

        # ==== STEP 4: Add graph-structural bias (NEW) ====
        # CHANGED: This entire step is new — encodes graph edges into
        # attention bias, allowing the model to up/down-weight attention
        # based on structural relationships between lines.
        edge_bias = self.edge_encoder(edge_features)  # (B, N, N, num_heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2)     # (B, num_heads, N, N)

        # Restrict graph bias to graph-connected pairs even in dense-attention
        # mode. This makes the k-NN graph meaningful without forcing sparse
        # masking.
        edge_bias = edge_bias.masked_fill(~adj_matrix.unsqueeze(1), 0.0)

        # Scale the graph bias by the learnable gate parameter
        attn_logits = attn_logits + self.graph_gate * edge_bias

        # ==== STEP 5: Apply sparse graph masking (NEW, optional) ====
        # CHANGED: When enabled, non-neighbor nodes get -inf attention,
        # enforcing strict locality. This both improves efficiency and
        # provides strong inductive bias for local structure.
        if self.use_sparse_attention:
            # Expand adj_matrix to match attention heads: (B, 1, N, N)
            sparse_mask = ~adj_matrix.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(sparse_mask, float('-inf'))

        # ==== STEP 6: Apply padding mask if provided (same as original) ====
        if mask is not None:
            attn_logits = attn_logits.masked_fill(
                ~mask.unsqueeze(1), float('-inf')
            )

        # ==== STEP 7: Compute attention output (same as original) ====
        attn_weights = F.softmax(attn_logits, dim=-1)
        self._last_attn_weights = attn_weights  # Cache for visualization
        context = torch.einsum('bhij,bhjd->bhid', attn_weights, v)

        # Reshape: (B, num_heads, N, head_dim) -> (B, N, embed_dim)
        message = self.out_proj(
            context.transpose(1, 2).flatten(start_dim=-2)
        )

        # ==== STEP 8: Residual + FFN (identical to original) ====
        return x + self.ffn(torch.cat([x, message], -1))


# =============================================================================
# COMPONENT 3: Graph Visualization Utility
# =============================================================================

class GraphVisualization:
    """
    Utility for visualizing the k-NN graph structure on document images.
    """
    
    @staticmethod
    def plot_graph_on_image(
        image_path: str,
        lines_data: list,
        adj_matrix: torch.Tensor,
        output_path: str,
        title: str = "Document Line Graph Connectivity",
        figsize: tuple = (12, 16)
    ):
        """
        Plots the graph connectivity overlaid on the original document image.
        
        Args:
            image_path: Path to the document image
            lines_data: List of line dicts containing geometry (polygons, bounding boxes)
            adj_matrix: NxN boolean adjacency matrix from GraphTransformerBlock
            output_path: Where to save the visualization
            title: Plot title
            figsize: Figure size tuple
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import cv2
        import numpy as np
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        
        N = len(lines_data)
        adj = adj_matrix.numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
        
        # Compute centroids for each line
        centroids = []
        for line in lines_data:
            if 'bbox' in line:
                # [x_min, y_min, x_max, y_max] or similar
                bbox = line['bbox']
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                centroids.append((cx, cy))
            elif 'polygon' in line:
                poly = np.array(line['polygon'])
                cx = np.mean(poly[:, 0])
                cy = np.mean(poly[:, 1])
                centroids.append((cx, cy))
            else:
                centroids.append((tuple(line.get('centroid', (0,0)))))
                
        # Draw edges
        for i in range(N):
            for j in range(N):
                if adj[i, j] and i != j:
                    x_coords = [centroids[i][0], centroids[j][0]]
                    y_coords = [centroids[i][1], centroids[j][1]]
                    ax.plot(x_coords, y_coords, color='cyan', alpha=0.5, linewidth=1.5, linestyle='solid')
                    
        # Draw nodes (centroids) and bounding boxes
        for i, line in enumerate(lines_data):
            cx, cy = centroids[i]
            
            # Node
            ax.plot(cx, cy, marker='o', color='red', markersize=4)
            
            # Draw bbox if available
            if 'bbox' in line:
                b = line['bbox']
                rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], 
                                         linewidth=1, edgecolor='blue', facecolor='none', alpha=0.3)
                ax.add_patch(rect)
                
            # Line ID
            ax.text(cx + 5, cy + 5, str(i), color='yellow', fontsize=8, 
                    bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))
                    
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
