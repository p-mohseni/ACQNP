import os
import sys
import math
import torch
import argparse
from utils import *


#######################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--min_num_context', type=int, default=500)
parser.add_argument('--num_iterations', type=int, default=int(1e5))

args = parser.parse_args()

torch.manual_seed(args.seed)


#######################################################################


train_data, test_data = load_data()
dataset = SpeedFlow(train_data, test_data)


#######################################################################

train_batches = []
for iteration in range(args.num_iterations):
    train_batches.append(dataset.sample(
        min_num_context=args.min_num_context)
    )

test_batches = []
test_batches.append(dataset.sample(testing=True))

root = os.path.join(os.getcwd(), 'data')
train_file_name, test_file_name = 'train-data', 'test-data'
torch.save(train_batches, os.path.join(root, f'{train_file_name}.tar'))
torch.save(test_batches, os.path.join(root, f'{test_file_name}.tar'))








