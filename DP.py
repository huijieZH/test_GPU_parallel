from DiT import DiT_S_8, DiT_L_8
from dataset import SampleDataset
import torch
from torch import optim
import tqdm
import os
from time import time
from torch import nn
import sys

assert len(sys.argv) == 4
num_GPU = int(sys.argv[1])
batch_size = int(sys.argv[2])
num_workers = int(sys.argv[3])
print(num_GPU, batch_size, num_workers)

assert num_GPU in [1, 2, 4]
if num_GPU == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif num_GPU == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
elif num_GPU == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

assert batch_size in [32, 64, 128, 256, 512, 1024, 2048]
model = DiT_L_8().to("cuda")
model = nn.DataParallel(model)
train_dataset = SampleDataset(data_size = 32)
assert num_workers in [2, 4, 8, 16]
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,num_workers=num_workers)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

time_lst = []
for data in tqdm.tqdm(train_dataloader):
    tt = time()
    optimizer.zero_grad()
    rgb = data['rgb'].to("cuda")
    t = data['time_step'].squeeze(dim=1).to("cuda")
    pred_data = model(rgb, t)
    (pred_data[:, :4] - rgb).mean().backward()
    optimizer.step()
    time_lst.append(time() - tt)
print(time_lst)