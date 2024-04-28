import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch.nn import Linear


class LatticeGNN(torch.nn.Module):
    def __init__(self, gnn_model, input_channels, hidden_channels, num_layers, p_dropout):
        super(LatticeGNN, self).__init__()
        self.model = self._set_model(gnn_model)
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self._set_layers(input_channels, hidden_channels)
        self.out = self._set_prediction_head(gnn_model, hidden_channels)

    @staticmethod
    def _set_model(gnn_model):
        if gnn_model == 'SAGE':
            return SAGEConv
        elif gnn_model == 'SAGE_HEAD':
            return SAGEConv
        else:
            raise ValueError(f"Invalid GNN model: {gnn_model}")

    def _set_layers(self, input_channels, hidden_channels):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'conv{layer_idx}', self.model(input_channels, hidden_channels))
            else:
                setattr(self, f'conv{layer_idx}', self.model(hidden_channels, hidden_channels))
        return

    def _set_prediction_head(self, gnn_model, hidden_channels):
        if gnn_model == 'SAGE_HEAD':
            return self.model(hidden_channels, 1)
        else:
            return Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f'conv{layer_idx}')
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        output = self.out(x).squeeze()
        # output = self.out(x, edge_index).squeeze()
        return output


