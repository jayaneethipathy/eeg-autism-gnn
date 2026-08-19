"""LSTM baseline and Graph Attention Network (GAT) for EEG window classification."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


class LSTMBaseline(nn.Module):
    """Each channel is a time series; LSTM over time, mean-pool over channels."""

    def __init__(self, n_channels: int, hidden: int = 32, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (batch, n_channels, n_samples)
        b, c, t = x.shape
        x = x.reshape(b * c, t, 1)
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1].reshape(b, c, -1).mean(dim=1)  # mean-pool over channels
        return self.classifier(h_n)


class EEGGAT(nn.Module):
    """
    Graph Attention Network over the PLV connectivity graph.
    Node features = per-channel band power (5 bands). Edges = thresholded PLV.
    """

    def __init__(self, n_features: int = 5, hidden: int = 16, n_classes: int = 2, heads: int = 4):
        super().__init__()
        self.gat1 = GATConv(n_features, hidden, heads=heads)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1)
        self.classifier = nn.Linear(hidden, n_classes)
        self.attn_weights = None  # populated on forward for interpretability

    def forward(self, batch: Batch):
        x, edge_index = batch.x, batch.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x, (edge_idx_out, alpha) = self.gat2(x, edge_index, return_attention_weights=True)
        self.attn_weights = (edge_idx_out, alpha.detach())
        x = torch.relu(x)
        # global mean pool per graph in the batch
        pooled = torch.zeros(batch.num_graphs, x.size(-1), device=x.device)
        pooled = pooled.index_add(0, batch.batch, x) / torch.bincount(batch.batch).unsqueeze(1)
        return self.classifier(pooled)


def build_graph_data(node_features: torch.Tensor, adjacency: torch.Tensor, label: int) -> Data:
    """node_features: (n_channels, n_bands); adjacency: (n_channels, n_channels) PLV matrix."""
    edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
    return Data(x=node_features, edge_index=edge_index, y=torch.tensor([label]))
