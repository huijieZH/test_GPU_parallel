from DiT import DiT_L_8
from dataset import SampleDataset
import torch
from torch import optim
import tqdm
import os
from time import time
import numpy as np
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()

parser.add_argument("--num_GPU", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--type_GPU", type=str)
args = parser.parse_args()

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')

assert args.batch_size in [32, 64, 128, 256, 512, 1024, 2048]
model = DiT_L_8().to("cuda")
model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids = [local_rank])

train_dataset = SampleDataset(data_size = 32)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last = False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, pin_memory = True, num_workers=1)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

print(f"Starting exp {local_rank}:  {args.type_GPU}:numGPU{args.num_GPU}_bs{args.batch_size}_DDP")

time_lst = []
for epoch in range(10):
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

torch.save(model.module.state_dict(), f"t{local_rank}.pt")

# if not os.path.exists("exp"):
#     os.mkdir("exp")
# np.save(f"exp/{args.type_GPU}:numGPU{args.num_GPU}_bs{args.batch_size}_DDP", np.array(time_lst))