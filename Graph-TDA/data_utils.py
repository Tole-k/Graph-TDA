from pyballmapper import BallMapper
import numpy as np
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
import torch
from torch.utils.data import Dataset


def dataset_to_signal(dataset, epsilons):
    snapshots = []
    points = list(range(len(dataset)))
    for eps in epsilons:
        bm = BallMapper(X=dataset, eps=eps, order=points)
        graph = bm.Graph
        new_edges = [
            (graph.nodes[edge[0]]["landmark"], graph.nodes[edge[1]]["landmark"])
            for edge in graph.edges
        ]
        nodes_dict = {point: -1 for point in points}
        for node_id in range(len(graph.nodes)):
            node = graph.nodes[node_id]
            nodes_dict[node["landmark"]] = node["size"]
        new_graph = nx.Graph()
        for node_id, size in nodes_dict.items():
            new_graph.add_node(node_id, x=size)
        new_graph.add_edges_from(new_edges)
        tg_graph = from_networkx(new_graph)
        snapshots.append(tg_graph)
    return snapshots


class TemporalSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def temporal_collate_fn(batch):
    sequences, labels = zip(*batch)
    T = len(sequences[0])
    edge_indices = []
    edge_weights = []
    features = []
    batches = []
    targets = []

    for t in range(T):
        edge_indices_t = []
        edge_weights_t = []
        features_t = []
        batch_t = []
        targets_t = []

        node_offset = 0
        for b, seq in enumerate(sequences):
            snapshot = seq[t]
            edge_indices_t.append(snapshot.edge_index + node_offset)
            edge_weights_t.append(torch.ones(snapshot.edge_index.size(1)))
            features_t.append(snapshot.x.float())
            targets_t.append(torch.zeros(snapshot.x.size(0)))
            batch_t.extend([b] * snapshot.x.size(0))
            node_offset += snapshot.x.size(0)

        edge_indices.append(torch.cat(edge_indices_t, dim=1))
        edge_weights.append(torch.cat(edge_weights_t, dim=0))
        features.append(torch.cat(features_t, dim=0).unsqueeze(-1))
        batches.append(torch.tensor(batch_t))
        targets.append(torch.cat(targets_t, dim=0).numpy())

    return DynamicGraphTemporalSignalBatch(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets,
        batches=batches,
    ), torch.tensor(labels)


def get_dataset(
    datasets: list[np.ndarray],
    names: list[str],
    labels: list[int],
    epsilons: list[float],
):
    sequences = []
    for dataset, name in zip(datasets, names):
        try:
            sequence = dataset_to_signal(dataset=dataset, epsilons=epsilons)
        except Exception as e:
            print(f"Error processing dataset {name}: {e}")
            continue
        sequences.append(sequence)
    temporal_dataset = TemporalSequenceDataset(sequences, labels)
    return temporal_dataset


def get_dataloader(
    datasets: list[np.ndarray],
    names: list[str],
    labels: list[int],
    epsilons: list[float],
    batch_size=16,
):
    temporal_dataset = get_dataset(
        datasets, names=names, labels=labels, epsilons=epsilons
    )
    dataloader = torch.utils.data.DataLoader(
        temporal_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=temporal_collate_fn,
    )
    return dataloader
