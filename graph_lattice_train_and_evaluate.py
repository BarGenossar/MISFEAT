import torch
from GNN_models import LatticeGNN, LatticeGraphSAGE, LatticeGAT
from config import LatticeGeneration, GNN, Evaluation
import tqdm
from utils import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(lattice_graph, model, optimizer, criterion):
    lattice_graph.to(device)
    model.train()
    optimizer.zero_grad()
    out = model(lattice_graph.x, lattice_graph.edge_index)
    loss = criterion(out, lattice_graph.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(lattice_graph, model, at_k=20):
    lattice_graph.to(device)
    model.eval()
    with torch.no_grad():
        out = model(lattice_graph.x, lattice_graph.edge_index)
    ndcg = compute_ndcg(lattice_graph.y, out, at_k)
    print(5*'======')
    print(f'NDCG: {ndcg}')
    return


def compute_dcg(ground_truth, sorted_indices):
    DCG = 0
    for i in range(1, len(sorted_indices) + 1):
        DCG += (ground_truth[sorted_indices[i-1]].item() / math.log2(i+1))
    return DCG


def compute_ndcg(ground_truth, predictions, at_k):
    sorted_gt_indices = torch.argsort(ground_truth, descending=True)[:at_k]
    sorted_pred_indices = torch.argsort(predictions, descending=True)[:at_k]
    IDCG = compute_dcg(ground_truth, sorted_gt_indices)
    DCG = compute_dcg(ground_truth, sorted_pred_indices)
    return round(DCG / IDCG, 4)


if __name__ == "__main__":
    formula_idx = LatticeGeneration.formula_idx
    hyperparams_idx = LatticeGeneration.hyperparams_idx
    min_k = LatticeGeneration.min_k
    within_level_edges = LatticeGeneration.within_level_edges
    is_hetero = LatticeGeneration.is_hetero
    with_edge_attrs = LatticeGeneration.with_edge_attrs

    gnn_model = GNN.gnn_model
    hidden_channels = GNN.hidden_channels
    num_layers = GNN.num_layers
    p_dropout = GNN.p_dropout
    epochs = GNN.epochs

    at_k = Evaluation.at_k

    dataset_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset.pkl"
    lattice_path = f"GeneratedData/Formula{formula_idx}/Config{hyperparams_idx}/dataset_lattice.pt"
    feature_num = read_feature_num_from_txt(dataset_path)

    lattice_graph = torch.load(lattice_path)
    model = LatticeGNN(gnn_model, feature_num, hidden_channels, num_layers, p_dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        loss_val = train(lattice_graph, model, optimizer, criterion)
        print(f'Epoch: {epoch}, Loss: {round(loss_val, 4)}')

    test(lattice_graph, model, at_k)







