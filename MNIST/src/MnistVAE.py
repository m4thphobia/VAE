import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from typing import Tuple
device = "cuda:1" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.eps = np.spacing(1)
        self.in_features = 28*28

        self.enc_fc1 = nn.Linear(self.in_features, 400)
        self.enc_fc2 = nn.Linear(400, 200)

        self.enc_mean = nn.Linear(200, z_dim)
        self.enc_var = nn.Linear(200, z_dim)

        self.dec_fc1 = nn.Linear(z_dim, 200)
        self.dec_fc2 = nn.Linear(200, 400)
        self.dropout = nn.Dropout(0.2)
        self.dec_fc3 = nn.Linear(400, self.in_features)

    def _encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))

        mean = F.relu(self.enc_mean(h))
        std = F.softplus(self.enc_var(h))

        return mean, std

    def _sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.training:
            epsilon = torch.randn(mean.shape).to(device)
            return mean + epsilon*std
        else:
            return mean

    def _decoder(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        h = self.dropout(h)
        y = torch.sigmoid(self.dec_fc3(h))

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self._encoder(x)
        z = self._sample_z(mean, std)
        y = self._decoder(z)

        return y, z

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        mean, std = self._encoder(x)
        KL = 0.5*torch.mean(torch.sum(1+torch.log(std**2+self.eps)-mean**2-std**2, dim=1))

        z = self._sample_z(mean, std)
        y = self._decoder(z)
        reconstruction = torch.mean(torch.sum(x * torch.log(y + self.eps) + (1 - x) * torch.log(1 - y + self.eps), dim=1))

        return -KL, -reconstruction
