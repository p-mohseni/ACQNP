import os
import sys
import torch
import argparse
import torchvision
from utils import *


#######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

#Fix seed (optional)
torch.manual_seed(args.seed)


#######################################################################


transforms = torchvision.transforms.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# Train dataset
dataset = torchvision.datasets.SVHN('./data',
                                    split='test',
                                    transform=transforms,
                                    download=True)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False)


#######################################################################


batches = []
for _, (img, label) in enumerate(dataloader):
    batches.append(img_to_task(img))

path = os.path.join(os.getcwd(), 'eval-data')
if not os.path.isdir(path):
    os.makedirs(path)

filename = 'test-data.tar'
torch.save(batches, os.path.join(path, filename))








