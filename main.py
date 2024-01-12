import sys

import torch
import transformers
import itertools
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import torch
from vit_pytorch import ViT, MAE

from dataset import FSDKaggle2019
root_path = sys.argv[1]

width, height = 256, 256
truncate_len = 128000
sample_rate = 32000
batch_size = 512
epochs = 500
max_subset_samples = 5000
lr = 2e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = dict(paused=False)
train_set = FSDKaggle2019(root_path + '/FSDKaggle2019.meta/train_curated_post_competition.csv',
                          root_path + '/FSDKaggle2019.audio_train_curated',
                          sample_rate, truncate_len, 1, width, height)
train_set.preload(state)
print("Finished preloading")
train_dl = DataLoader(train_set, batch_size=batch_size)

n_classes = len(train_set.label_map)
print(device)
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = n_classes,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    channels=1
).to(device)

mae = MAE(
    encoder = v,
    masking_ratio = 0.75,   # the paper recommended 75% masked patches
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6       # anywhere from 1 to 8
).to(device)

n_pre_epochs = 10
optimizer = optim.Adam(itertools.chain(mae.parameters(), v.parameters()), lr=lr)

best_loss = 1000

for epoch in range(n_pre_epochs):
    step = 0
    for batch in train_dl:
        optimizer.zero_grad()
        inputs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        loss = mae(inputs)
        loss.backward()
        optimizer.step()
        best_loss = min(best_loss, loss.item())
        step += 1
        print(f"e={epoch} s={step} l={loss.item():.4f}")

torch.save(v.state_dict(), f'./trained-vit-{best_loss:0.4f}.pt')

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(v.parameters(), lr=lr)

for epoch in range(epochs):
    step = 0
    for batch in train_dl:
        optimizer.zero_grad()
        inputs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)

        # predict our outputs with the model
        outputs = v(inputs)

        # calculate the loss (error) - how well our model did in matching the labels
        loss = loss_fn(outputs, labels)

        # calculate the loss surface -- if you could tweak every parameter in the model slightly, for each one,
        # which way makes the loss go down.
        loss.backward()

        # take a small step in that direction that makes the loss go down
        optimizer.step()
        step += 1
        print(f"e={epoch} s={step} l={loss.item():.4f}")
