import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
import torch_geometric
from torch.nn import Linear


class LatticeGNN(torch.nn.Module):
    def __init__(self, gnn_model, input_channels, hidden_channels, num_layers, p_dropout):
        super(LatticeGNN, self).__init__()
        self.model = self._set_model(gnn_model)
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self._set_layers(input_channels, hidden_channels)
        self.out = Linear(hidden_channels, 1)

    @staticmethod
    def _set_model(gnn_model):
        if gnn_model == 'SAGE':
            return SAGEConv
        elif gnn_model == 'GAT':
            # Can't use GAT for heterogeneous graphs
            return GATConv
        else:
            raise ValueError(f"Invalid GNN model: {gnn_model}")

    def _set_layers(self, input_channels, hidden_channels):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'conv{layer_idx}', self.model(input_channels, hidden_channels))
            else:
                setattr(self, f'conv{layer_idx}', self.model(hidden_channels, hidden_channels))
        return

    def forward(self, x, edge_index):
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f'conv{layer_idx}')
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        output = self.out(x).squeeze()
        return output


