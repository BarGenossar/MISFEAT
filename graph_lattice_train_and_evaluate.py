import torch
from GNN_models import LatticeGraphSAGE, LatticeGAT
from config import LatticeGeneration, GNN
import tqdm
from utils import *
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(lattice_graph, model, optimizer, criterion):
    lattice_graph.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(lattice_graph.x, lattice_graph.edge_index)
    loss = criterion(out, lattice_graph.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(lattice_graph, model):
    lattice_graph.to(device)
    model.eval()
    with torch.no_grad():
        out = model(lattice_graph.x, lattice_graph.edge_index)
    return out



if __name__ == "__main__":
    formula_idx = LatticeGeneration.formula_idx
    hyperparams_idx = LatticeGeneration.hyperparams_idx
    min_k = LatticeGeneration.min_k
    within_level_edges = LatticeGeneration.within_level_edges
    is_hetero = LatticeGeneration.is_hetero
    with_edge_attrs = LatticeGeneration.with_edge_attrs

    hidden_channels = GNN.hidden_channels
    num_layers = GNN.num_layers
    p_dropout = GNN.p_dropout
    epochs = GNN.epochs

    dataset_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset.pkl"
    lattice_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset_lattice.pt"
    feature_num = read_feature_num_from_txt(dataset_path)

    model = LatticeGAT(feature_num, hidden_channels, num_layers, p_dropout)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()


    lattice_graph = torch.load(lattice_path)

    for epoch in range(1, epochs + 1):
        loss_val = train(lattice_graph, model, optimizer, criterion)
        print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')

    output = test(lattice_graph, model)
    # save the output
    output_path = lattice_path.replace('lattice', 'output')






