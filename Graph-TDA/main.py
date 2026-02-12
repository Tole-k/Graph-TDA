from palmerpenguins import load_penguins
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    load_diabetes,
    load_linnerud,
)
import numpy as np
from torch import nn
from data_utils import get_dataloader
from model import SequenceClassifier
import torch

penguins, _ = load_penguins(return_X_y=True, drop_na=True)
penguins_Y = 0

iris, _ = load_iris(return_X_y=True)
iris_Y = 1

wine, _ = load_wine(return_X_y=True)
wine_Y = 0

breast_cancer, _ = load_breast_cancer(return_X_y=True)
breast_cancer_Y = 1

digits, _ = load_digits(return_X_y=True)
digits_Y = 0

diabetes, _ = load_diabetes(return_X_y=True)
diabetes_Y = 1

linnerud, _ = load_linnerud(return_X_y=True)
linnerud_Y = 0

datasets = [penguins, iris, wine, breast_cancer, digits, diabetes, linnerud]
names = ["penguins", "iris", "wine", "breast_cancer", "digits", "diabetes", "linnerud"]
labels = [penguins_Y, iris_Y, wine_Y, breast_cancer_Y, digits_Y, diabetes_Y, linnerud_Y]

epsilons = list(np.arange(0, 5, 1))

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

dataloader = get_dataloader(
    datasets=datasets,
    names=names,
    labels=labels,
    epsilons=epsilons,
    device=device,
    batch_size=4,
)
model = SequenceClassifier(in_channels=1, hidden_dim=64, num_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.CrossEntropyLoss()

epochs = 100

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    losses = []
    for batch in dataloader:
        X, labels = batch
        outputs = model(X)
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")
