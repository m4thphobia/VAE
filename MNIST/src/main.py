import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import math
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple
from MnistVAE import *
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

rng = np.random.RandomState(1234)
random_state = 42

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ]
)

batch_size = 128
n_epochs = 15

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

valid_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False
)

model_0 = VAE(2).to(device)
optimizer = optim.Adam(model_0.parameters(), lr=0.001)

epochs = 100

for epoch in range(epochs):
    losses = []
    KL_losses = []
    reconstruction_losses = []

    model_0.train()

    for x_train, _ in train_loader:
        x_train = x_train.to(device)

        model_0.zero_grad()

        KL_loss, reconstruction_loss = model_0.loss(x_train)
        # エビデンス下界の最大化のためマイナス付きの各項の値を最小化するようにパラメータを更新
        loss = KL_loss + reconstruction_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().numpy())
        KL_losses.append(KL_loss.cpu().detach().numpy())
        reconstruction_losses.append(reconstruction_loss.cpu().detach().numpy())

    losses_val = []
    model_0.eval()
    for x_valid, t in valid_loader:

        x_valid = x_valid.to(device)

        KL_loss, reconstruction_loss = model_0.loss(x_valid)
        loss = - KL_loss - reconstruction_loss

        losses_val.append(loss.cpu().detach().numpy())

    print(f'EPOCH:{epoch+1} Train Lower Bound:{np.average(losses)} (KL_loss: {np.average(KL_losses)} reconstruction_loss: {np.average(reconstruction_losses)})    Valid Lower Bound: {np.average(losses_val)}')



DIR_PATH = "../out"
if not os.path.exists(DIR_PATH):
    os.makedirs(DIR_PATH)
MODEL = "/model_0"
PATH = DIR_PATH + MODEL + ".pth"
torch.save(model_0.state_dict(), PATH)
