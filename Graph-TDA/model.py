import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.nn import global_mean_pool


class GraphSequenceModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()
        self.recurrent = GConvLSTM(in_channels, hidden_dim, K=2)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, dataset_batch):
        device = next(self.parameters()).device
        h_state = None
        c_state = None

        for snapshot in dataset_batch:
            x = snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_attr = (
                snapshot.edge_attr.to(device)
                if snapshot.edge_attr is not None
                else None
            )
            batch = snapshot.batch.to(device)
            h_state, c_state = self.recurrent(
                x, edge_index, edge_attr, h_state, c_state
            )

        pooled = global_mean_pool(h_state, batch)

        return self.classifier(pooled)
