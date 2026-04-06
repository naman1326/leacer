"""
GAT Encoder — Layer 2, LEACER Framework

Encodes the road network graph G=(V,E) into spatial node embeddings h_i
using Graph Attention Networks. Each node = intersection, each edge = road.

Node features x_i = [S_i, D_i, Q_i, L_i, degree_i, centrality_i]
Edge features     = [length, speed_limit, lanes]

Output: h_i in R^d_emb per node  →  fed into PPO Policy Agent
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Road Graph Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoadGraph:
    """
    Lightweight adjacency representation of a road network.

    node_features: (N, F_n)  — per-intersection features
    edge_index:    (2, E)    — [source_nodes; target_nodes]
    edge_features: (E, F_e)  — per-road-segment features
    node_ids:      list of N intersection IDs
    """
    node_features: np.ndarray      # (N, F_n)
    edge_index:    np.ndarray      # (2, E)  int
    edge_features: np.ndarray      # (E, F_e)
    node_ids:      List[str]

    @property
    def num_nodes(self): return self.node_features.shape[0]
    @property
    def num_edges(self): return self.edge_index.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# GAT Layer (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class GATLayer(nn.Module):
        """
        Single GAT layer with multi-head attention.

        Attention coefficient:
          e_ij = LeakyReLU( a^T [W h_i || W h_j] )
          alpha_ij = softmax_j( e_ij )
          h_i' = ELU( sum_j alpha_ij W h_j )

        Concatenates K heads, then projects to out_dim.
        """

        def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                     dropout: float = 0.1, negative_slope: float = 0.2):
            super().__init__()
            self.K = num_heads
            self.d = out_dim
            self.dropout = dropout

            # Shared linear per head
            self.W   = nn.Linear(in_dim, num_heads * out_dim, bias=False)
            # Attention vector per head: a in R^{2*out_dim}
            self.att = nn.Parameter(torch.Tensor(1, num_heads, 2 * out_dim))
            nn.init.xavier_uniform_(self.att)
            self.leaky = nn.LeakyReLU(negative_slope)
            self.proj  = nn.Linear(num_heads * out_dim, out_dim)

        def forward(self, h: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
            """
            h:          (N, in_dim)
            edge_index: (2, E)
            returns:    (N, out_dim)
            """
            N = h.size(0)
            # Transform: (N, K, d)
            Wh = self.W(h).view(N, self.K, self.d)

            src, dst = edge_index[0], edge_index[1]
            Wh_src = Wh[src]   # (E, K, d)
            Wh_dst = Wh[dst]   # (E, K, d)

            # Attention logits (E, K)
            cat  = torch.cat([Wh_src, Wh_dst], dim=-1)   # (E, K, 2d)
            e    = (cat * self.att).sum(-1)               # (E, K)
            e    = self.leaky(e)

            # Softmax per target node (sparse scatter)
            # Simple dense version for readability:
            alpha = torch.zeros(N, N, self.K, device=h.device)
            alpha[dst, src] = e
            alpha = F.softmax(alpha + (~(alpha != 0)).float() * -1e9, dim=1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            # Aggregate: (N, K, d)
            agg = torch.einsum('ijk,jkd->ikd', alpha, Wh)  # (N, K, d)
            agg = agg.view(N, self.K * self.d)
            return F.elu(self.proj(agg))


    class GATEncoder(nn.Module):
        """
        2-layer GAT producing node embeddings h_i in R^{emb_dim}.

        Architecture:
          Input (N, F_n) → GATLayer → GATLayer → LayerNorm → h (N, emb_dim)
        """

        def __init__(self, node_feat_dim=6, hidden_dim=32,
                     emb_dim=64, num_heads=4, dropout=0.1):
            super().__init__()
            self.gat1 = GATLayer(node_feat_dim, hidden_dim, num_heads, dropout)
            self.gat2 = GATLayer(hidden_dim,    emb_dim,    num_heads, dropout)
            self.norm = nn.LayerNorm(emb_dim)

        def forward(self, graph: "RoadGraphTensor") -> torch.Tensor:
            h = self.gat1(graph.x, graph.edge_index)
            h = self.gat2(h,       graph.edge_index)
            return self.norm(h)

        def encode(self, road_graph: RoadGraph) -> np.ndarray:
            """Convenience: RoadGraph numpy → embeddings numpy (N, emb_dim)."""
            self.eval()
            with torch.no_grad():
                x  = torch.tensor(road_graph.node_features, dtype=torch.float32)
                ei = torch.tensor(road_graph.edge_index,    dtype=torch.long)

                class G: pass
                g = G(); g.x = x; g.edge_index = ei
                return self.forward(g).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Numpy fallback: degree-weighted mean aggregation
# ─────────────────────────────────────────────────────────────────────────────

class GATEncoderNumpy:
    """
    Fallback GNN: 2-step mean aggregation of neighbour features.
    No learnable parameters. Useful for quick inference tests.
    """
    def __init__(self, emb_dim=64):
        np.random.seed(42)
        self.W = np.random.randn(6, emb_dim) * 0.1

    def encode(self, road_graph: RoadGraph) -> np.ndarray:
        """Returns (N, emb_dim) embeddings."""
        N = road_graph.num_nodes
        src, dst = road_graph.edge_index
        # Mean-aggregate neighbours
        agg = road_graph.node_features.copy()
        for _ in range(2):
            new_agg = np.zeros_like(agg)
            counts  = np.zeros(N)
            for s, d in zip(src, dst):
                new_agg[d] += agg[s]
                counts[d]  += 1
            new_agg /= np.maximum(counts[:,None], 1)
            agg = (agg + new_agg) / 2.0
        return np.tanh(agg @ self.W)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder utility
# ─────────────────────────────────────────────────────────────────────────────

def build_road_graph(tsv_dict: Dict[str, "TrafficStateVector"],
                     adjacency: List[Tuple[str,str]],
                     edge_attrs: Dict[Tuple[str,str], List[float]] = None) -> RoadGraph:
    """
    Builds RoadGraph from DTSA output.

    tsv_dict:   {edge_id: TrafficStateVector}
    adjacency:  list of (src_id, dst_id) tuples
    edge_attrs: optional {(src,dst): [length_m, speed_limit, lanes]}
    """
    node_ids = list(tsv_dict.keys())
    idx = {n: i for i, n in enumerate(node_ids)}

    # Node features: [S, D, Q, L, degree, 0-pad]
    degrees = {n: 0 for n in node_ids}
    for s, d in adjacency:
        degrees[s] = degrees.get(s, 0) + 1
        degrees[d] = degrees.get(d, 0) + 1

    feats = []
    for nid in node_ids:
        tsv = tsv_dict[nid]
        feats.append([tsv.S, tsv.D, tsv.Q, tsv.L, degrees[nid], tsv.confidence])
    node_features = np.array(feats, dtype=np.float32)

    # Edge index
    srcs = [idx[s] for s, d in adjacency if s in idx and d in idx]
    dsts = [idx[d] for s, d in adjacency if s in idx and d in idx]
    edge_index = np.array([srcs, dsts], dtype=np.int64)

    # Edge features
    if edge_attrs:
        ef = [edge_attrs.get((s,d), [500.0, 60.0, 2.0])
              for s,d in adjacency if s in idx and d in idx]
    else:
        ef = [[500.0, 60.0, 2.0]] * len(srcs)
    edge_features = np.array(ef, dtype=np.float32)

    return RoadGraph(node_features, edge_index, edge_features, node_ids)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dtsa import TrafficStateVector
    import time

    # Synthetic 4-node graph: N0-N1-N2-N3 in a loop
    nodes = ["N0","N1","N2","N3"]
    tsvs = {n: TrafficStateVector(n, time.time(),
                45.0-i*3, 20+i*2, i, 40+i*5, 0.8, 0b0011)
            for i,n in enumerate(nodes)}
    adj = [("N0","N1"),("N1","N2"),("N2","N3"),("N3","N0")]
    graph = build_road_graph(tsvs, adj)

    encoder = GATEncoderNumpy(emb_dim=16)
    embs = encoder.encode(graph)
    print(f"Node embeddings shape: {embs.shape}")   # (4, 16)
    print(f"N0 embedding (first 4 dims): {embs[0,:4].round(4)}")
