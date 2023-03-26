from DiT import DiT_XL_8
from dataset import SampleDataset
import torch
from torch import optim
import tqdm
import os
from time import time
from torch import nn
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--num_GPU", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--type_GPU", type=str)
args = parser.parse_args()

assert args.num_GPU in [1, 2, 4]
if args.num_GPU == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device_ids = [0]
elif args.num_GPU == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    device_ids = [0, 1]
elif args.num_GPU == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    device_ids = [0, 1, 2, 3]

assert args.batch_size in [32, 64, 128, 256, 512, 1024, 2048]
model = DiT_XL_8().to("cuda")
model = nn.DataParallel(model, device_ids = device_ids)
train_dataset = SampleDataset(data_size = 32)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(f"Starting exp :  {args.type_GPU}:numGPU{args.num_GPU}_bs{args.batch_size}_DP")
time_lst = []
for data in train_dataloader:
    tt = time()
    optimizer.zero_grad()
    rgb = data['rgb'].to("cuda")
    t = data['time_step'].squeeze(dim=1).to("cuda")
    pred_data = model(rgb, t)
    (pred_data[:, :4] - rgb).mean().backward()
    optimizer.step()
    time_lst.append(time() - tt)



if not os.path.exists("exp"):
    os.mkdir("exp")
np.save(f"exp/{args.type_GPU}:numGPU{args.num_GPU}_bs{args.batch_size}_DP", np.array(time_lst))