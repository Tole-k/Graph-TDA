import base64
import hashlib
import os
from pyballmapper import BallMapper
import numpy as np
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def cached_dataset_to_signal(dataset, name, epsilons, cache_dir:str="cache"):
    snapshots = []
    points = list(range(len(dataset)))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for eps in epsilons:
        hash_name = short_hash(f"{name}_{eps}")
        cache_path = os.path.join(cache_dir, f"{hash_name}.pt")
        if os.path.exists(cache_path):
            snapshots.append(torch.load(cache_path,weights_only=False))
        else:
            bm = BallMapper(X=dataset, eps=eps, order=points)
            graph = bm.Graph
            new_edges = [(graph.nodes[edge[0]]["landmark"], graph.nodes[edge[1]]["landmark"]) for edge in graph.edges]
            nodes_dict = {point: 0 for point in points}
            for node_id in range(len(graph.nodes)):
                node = graph.nodes[node_id]
                nodes_dict[node["landmark"]] = node["size"]
            new_graph = nx.Graph()
            for node_id, size in nodes_dict.items():
                new_graph.add_node(node_id, x=size)
            new_graph.add_edges_from(new_edges)
            tg_graph = from_networkx(new_graph)
            torch.save(tg_graph, cache_path)
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

def short_hash(s: str, length=10) -> str:
    h = hashlib.blake2b(s.encode(), digest_size=8).digest()
    return base64.urlsafe_b64encode(h).decode("ascii")[:length]


def get_dataset(
    datasets: list[np.ndarray],
    names: list[str],
    scores: list[int],
    epsilons: list[float],
):
    sequences = []
    for dataset,name in tqdm(zip(datasets,names), total=len(datasets)):
        try:
            sequence = cached_dataset_to_signal(dataset=dataset,name=name, epsilons=epsilons)
        except Exception as e:
            print(f"Error processing dataset: {e}")
            continue
        sequences.append(sequence)
    temporal_dataset = TemporalSequenceDataset(sequences, scores)
    return temporal_dataset
