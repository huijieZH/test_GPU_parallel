from DiT import DiT_XL_8
from dataset import SampleDataset
import torch
from torch import optim
import tqdm
import os
from time import time
from torch import nn
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num_GPU", type=int)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

assert args.num_GPU in [1, 2, 3, 4]
if args.num_GPU == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif args.num_GPU == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
elif args.num_GPU == 3:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
elif args.num_GPU == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

assert args.batch_size in [32, 64, 128, 256, 512, 1024, 2048]
model = DiT_XL_8().to("cuda")
model = nn.DataParallel(model)
train_dataset = SampleDataset(data_size = 32)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8)
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