import os
import sys
import torch
import argparse
from utils import *


#######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_points', type=int, default=500)
parser.add_argument('--max_num_context', type=int, default=100)
parser.add_argument('--num_iterations', type=int, default=int(5e3))
parser.add_argument('--kernel', type=str, default='doule-sine',
                    choices=['doule-sine', 'circle', 'lissajous', 'sawtooth', 'rbf', 'matern52'])
args = parser.parse_args()

#Fix seed (optional)
torch.manual_seed(args.seed)


#######################################################################

if args.kernel == 'doule-sine':
    sampler = CurveSampler(DoubleSineKernel())
elif args.kernel == 'circle':
    sampler = CurveSampler(CircularKernel())
elif args.kernel == 'lissajous':
    sampler = CurveSampler(LissajousKernel())
elif args.kernel == 'sawtooth':
    sampler = CurveSampler(SawtoothKernel())
elif args.kernel == 'rbf':
    sampler = CurveSampler(RBFKernel())
elif args.kernel == 'matern52':
    sampler = CurveSampler(Matern52Kernel())
else:
    raise Exception("Kernel not found")


#######################################################################

batches = []
for iteration in range(args.num_iterations):
    batches.append(sampler.sample(
        batch_size=args.batch_size,
        num_points=args.num_points,
        max_num_context=args.max_num_context)
    )

path = os.path.join(os.getcwd(), 'eval-data')
if not os.path.isdir(path):
    os.makedirs(path)

filename = 'test-data.tar'
torch.save(batches, os.path.join(path, filename))








