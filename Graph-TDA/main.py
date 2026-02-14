# from palmerpenguins import load_penguins
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    # load_digits,
    # load_diabetes,
    # load_linnerud,
)
import numpy as np
from torch import nn
from data_utils import get_dataloader
from model import GraphSequenceModel
import torch

# penguins, _ = load_penguins(return_X_y=True, drop_na=True)
# penguins_Y = 0

iris, _ = load_iris(return_X_y=True)
iris_Y = [
    0.9939577039274925,
    0.9808660624370595,
    0.973816717019134,
    0.9939577039274925,
    1.0,
    0.9939577039274925,
    0.9798590130916415,
    0.9939577039274925,
    0.972809667673716,
    1.0,
    0.972809667673716,
    0.9808660624370595,
    0.9808660624370595,
    0.973816717019134,
    0.973816717019134,
    0.32225579053373615,
    0.986908358509567,
]

wine, _ = load_wine(return_X_y=True)
wine_Y = [
    0.9940000000000001,
    0.988,
    0.9890000000000001,
    0.977,
    1.0,
    0.9940000000000001,
    0.983,
    0.9940000000000001,
    0.972,
    1.0,
    0.977,
    0.983,
    0.9890000000000001,
    0.9890000000000001,
    0.966,
    0.39799999999999996,
    0.9890000000000001,
]

breast_cancer, _ = load_breast_cancer(return_X_y=True)
breast_cancer_Y = [
    0.9816272965879265,
    0.9724409448818897,
    1.0,
    0.9724409448818897,
    0.9566929133858268,
    0.9776902887139107,
    0.9671916010498688,
    1.0,
    0.9816272965879265,
    0.9790026246719159,
    0.9540682414698163,
    0.9842519685039369,
    0.9448818897637795,
    0.9908136482939632,
    1.0,
    0.9212598425196851,
    0.9842519685039369,
]

# digits, _ = load_digits(return_X_y=True)
# digits_Y = 0

# diabetes, _ = load_diabetes(return_X_y=True)
# diabetes_Y = 1

# linnerud, _ = load_linnerud(return_X_y=True)
# linnerud_Y = 0

# datasets = [penguins, iris, wine, breast_cancer, digits, diabetes, linnerud]
# names = ["penguins", "iris", "wine", "breast_cancer", "digits", "diabetes", "linnerud"]
# labels = [penguins_Y, iris_Y, wine_Y, breast_cancer_Y, digits_Y, diabetes_Y, linnerud_Y]

datasets = [iris, wine, breast_cancer]
names = ["iris", "wine", "breast_cancer"]
labels = [iris_Y, wine_Y, breast_cancer_Y]

epsilons = [0.25, 1, 5, 25]

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
    batch_size=4,
)
model = GraphSequenceModel(in_channels=1, hidden_dim=256, out_dim=17).to(device)
optimizer = torch.optim.AdamW(model.parameters())
criterion = nn.MSELoss()

epochs = 10000
best_loss = float("inf")
early_stopping_patience = 10

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
