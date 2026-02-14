import numpy as np
from torch import nn
from data_utils import get_dataset, temporal_collate_fn
from model import GraphSequenceModel
from autoirad_datasets import generate_training_parameters
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    epsilons = [0.25, 1, 5, 25]

    device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

    datasets,names, scores = generate_training_parameters()
    train_datasets, val_datasets, train_names, val_names, train_scores, val_scores = train_test_split(
    datasets,names, scores, test_size=0.2, random_state=42)

    train_dataset = get_dataset(train_datasets,train_names, train_scores, epsilons=epsilons)
    test_dataset = get_dataset(val_datasets,val_names, val_scores, epsilons=epsilons)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=temporal_collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=temporal_collate_fn,
    )

    model = GraphSequenceModel(in_channels=1, hidden_dim=256, out_dim=17).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()

    epochs = 100
    best_loss = float("inf")
    early_stopping_patience = 10
    pbar = tqdm(range(epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        model.train()

        losses = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            X, scores = batch
            outputs = model(X)
            loss = criterion(outputs, scores.to(device))
            loss.backward()
            optimizer.step()

        for batch in test_dataloader:
            model.eval()
            with torch.inference_mode():
                X, scores = batch
                outputs = model(X)
                loss = criterion(outputs, scores.to(device))
                losses.append(loss.item())

        mean_loss = np.mean(losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            early_stopping_patience = 10
        else:
            early_stopping_patience -= 1
            if early_stopping_patience == 0:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        pbar.set_postfix({"Loss": f"{np.mean(losses):.4f}"})

if __name__ == "__main__":
    main()