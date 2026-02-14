import numpy as np
from torch import nn
from data_utils import get_dataloader
from model import GraphSequenceModel
from autoirad_datasets import generate_training_parameters
import torch
from sklearn.model_selection import train_test_split

epsilons = [0.25, 1, 5, 25]

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

datasets, scores = generate_training_parameters()
train_datasets, val_datasets, train_scores, val_scores = train_test_split(
    datasets, scores, test_size=0.2, random_state=42
)
train_dataloader = get_dataloader(train_datasets, train_scores, epsilons=epsilons)

test_loader = get_dataloader(val_datasets, val_scores, epsilons=epsilons, shuffle=False)

model = GraphSequenceModel(in_channels=1, hidden_dim=256, out_dim=17).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.MSELoss()

epochs = 10000
best_loss = float("inf")
early_stopping_patience = 10

for epoch in range(epochs):
    model.train()

    losses = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        X, scores = batch
        outputs = model(X)
        loss = criterion(outputs, scores.to(device))
        loss.backward()
        optimizer.step()

    for batch in test_loader:
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

    print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")
