import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric import seed_everything
import torch_geometric
from torch.nn import Linear



class LatticeGraphSAGE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=3, p_dropout=0.2, seed=1):
        super(LatticeGraphSAGE, self).__init__()
        seed_everything(seed)
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self._set_layers(input_channels, hidden_channels)
        self.out = Linear(hidden_channels, 1)

    def _set_layers(self, input_channels, hidden_channels):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'conv{layer_idx}', SAGEConv(input_channels, hidden_channels))
            else:
                setattr(self, f'conv{layer_idx}', SAGEConv(hidden_channels, hidden_channels))
        return

    def forward(self, x, edge_index):
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f'conv{layer_idx}')
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        return self.out(x)


class LatticeGAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=3, p_dropout=0.2, seed=2):
        super(LatticeGAT, self).__init__()
        seed_everything(seed)
        self.num_layers = num_layers
        self.p_dropout = p_dropout
        self._set_layers(input_channels, hidden_channels)
        self.out = Linear(hidden_channels, 1)

    def _set_layers(self, input_channels, hidden_channels):
        for layer_idx in range(1, self.num_layers + 1):
            if layer_idx == 1:
                setattr(self, f'conv{layer_idx}', GATConv(input_channels, hidden_channels))
            else:
                setattr(self, f'conv{layer_idx}', GATConv(hidden_channels, hidden_channels))
        return

    def forward(self, x, edge_index):
        for layer_idx in range(1, self.num_layers + 1):
            conv = getattr(self, f'conv{layer_idx}')
            x = F.leaky_relu(conv(x, edge_index))
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        output = self.out(x).squeeze()
        return output


