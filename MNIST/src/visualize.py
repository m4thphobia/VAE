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


valid_dataset = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transform)

## plot true image
fig = plt.figure(figsize=(10, 4))
for i in range(40):
    x, t = valid_dataset[i]

    im = x.view(-1, 28, 28).permute(1, 2, 0).squeeze().numpy()

    ax = fig.add_subplot(4, 10, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')
plt.savefig("../out/true.png")

## load model
DIR_PATH = "../out"
MODEL = "/model_0"
PATH = DIR_PATH + MODEL + ".pth"

model_1 = VAE(2).to(device)
model_1.load_state_dict(torch.load(PATH))

## reconstruction
fig = plt.figure(figsize=(10, 4))
model_1.eval()
for i in range(40):
    x, t = valid_dataset[i]

    x = x.unsqueeze(0).to(device)

    y, z = model_1(x)

    im = y.view(-1, 28, 28).permute(1, 2, 0).detach().cpu().squeeze().numpy()

    ax = fig.add_subplot(4, 10, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')
plt.savefig("../out/reconstruction.png")

## random sampling from latent space
z_dim=2
z = torch.randn([40, z_dim]).to(device)

fig = plt.figure(figsize=(10, 4))
model_1.eval()
for i in range(40):
    y = model_1._decoder(z[i])

    im = y.view(-1, 28, 28).permute(1, 2, 0).detach().cpu().squeeze().numpy()

    ax = fig.add_subplot(4, 10, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')
plt.savefig("../out/latent_sample.png")

## visualize latent space

z_list = []
t_list = []
for x, t in valid_dataset:
    t_list.append(t)
    x = x.to(device).unsqueeze(0)
    _, z = model_1(x)
    z_list.append(z.cpu().detach().numpy()[0])

z_val = np.stack(z_list)

REDUC = 'TSNE' # 'TSNE' または 'PCA'
if z_dim > 2:
    if REDUC == "TSNE":
        z_reduc = TSNE(n_components=2).fit_transform(z_val).T
    elif REDUC == "PCA":
        z_reduc = PCA(n_components=2).fit_transform(z_val).T
    else:
        raise ValueError("Please choose dimensionality reduction method from TSNE or PCA.")
elif z_dim == 2:
    z_reduc = z_val.T
else:
    raise ValueError("z dimensionality must be larger or equal to 2.")

colors = ['khaki', 'lightgreen', 'cornflowerblue', 'violet', 'sienna',
            'darkturquoise', 'slateblue', 'orange', 'darkcyan', 'tomato']

plt.figure(figsize=(5,5))
plt.scatter(*z_reduc, s=0.7, c=[colors[t] for t in t_list])
# 凡例を追加
for i in range(10):
    plt.scatter([],[], c=colors[i], label=i)
plt.legend()
plt.savefig("../out/latent_space.png")


## walk through the latent space
START_SAMPLE_IDX = 0
END_SAMPLE_IDX = 3
TITLES = ["start", "end"]

z_dict = {}
fig = plt.figure(figsize=(5, 5))
for i, sample_idx in enumerate([START_SAMPLE_IDX, END_SAMPLE_IDX]):
    x, t = valid_dataset[sample_idx]
    print(f"{TITLES[i]} sample label: {t}")

    x = x.unsqueeze(0).to(device)
    y, z = model_1(x)

    image = y.view(-1, 28, 28).permute(1, 2, 0).detach().cpu().squeeze().numpy()
    z_dict.update({TITLES[i]: z})

    ax = fig.add_subplot(1, 2, i+1, xticks=[], yticks=[])
    ax.set_title(f"{TITLES[i]} image")
    ax.imshow(image, 'gray')


# start sampleのzとend sampleのzの間を線形に補間
z_interp = torch.cat(
    [torch.lerp(z_dict["start"], z_dict["end"], weight=w) for w in np.arange(0, 1.1, 0.1)]
)

images = model_1._decoder(z_interp).view(-1, 28, 28).detach().cpu().numpy()

fig = plt.figure(figsize=(15, 15))
for i, im in enumerate(images):
    ax = fig.add_subplot(1, 11, i+1, xticks=[], yticks=[])
    ax.imshow(im, 'gray')

z_interp = np.concatenate([z_val, z_interp.detach().cpu().numpy()], axis=0)

REDUC = "TSNE"
if z_dim > 2:
    if REDUC == "TSNE":
        z_reduc = TSNE(n_components=2).fit_transform(z_interp).T
    elif REDUC == "PCA":
        z_reduc = PCA(n_components=2).fit_transform(z_interp).T
    else:
        raise ValueError("Please choose dimensionality reduction method from TSNE or PCA.")
elif z_dim == 2:
    z_reduc = z_interp.T
else:
    raise ValueError("z dimensionality must be larger or equal to 2.")

t_list_interp = t_list + [10] * 11
colors_interp = colors + ["black"]
plt.figure(figsize=(5,5))
plt.scatter(
    *z_reduc,
    s=[0.7 if t < 10 else 20.0 for t in t_list_interp],
    c=[colors_interp[t] for t in t_list_interp]
)
# 凡例を追加
for i in range(10):
    plt.scatter([],[], c=colors[i], label=i)
plt.legend()
plt.savefig("../out/latent_walkthrough.png")
