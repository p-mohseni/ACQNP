import os
import sys
import time
import math
import argparse
import platform
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.getcwd(), '..'))
from utils import *
sys.path.insert(0, os.path.join(os.getcwd(), '../../models'))
from BNP.bnp import BNP


#######################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--train_seed', type=int, default=0)

parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--max_num_points', type=int, default=100)
parser.add_argument('--kernel', type=str, default='doule-sine',
                    choices=['doule-sine', 'circle', 'lissajous', 'sawtooth', 'rbf', 'matern52'])

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l2_penalty', type=float, default=0)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--num_iterations', type=int, default=int(1e5))
parser.add_argument('--save_freq', type=int, default=int(5e3))

parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--output_dim', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--rep_dim', type=int, default=128)
parser.add_argument('--encoder_pre_depth', type=int, default=5)
parser.add_argument('--encoder_post_depth', type=int, default=3)
parser.add_argument('--decoder_depth', type=int, default=4)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Fix seed (optional)
torch.manual_seed(args.train_seed)
if device == torch.device('cuda'):
    torch.cuda.manual_seed(args.train_seed)


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


model_type = 'BNP'
model = BNP(dim_x=args.input_dim,
            dim_y=args.output_dim,
            dim_hid=args.hidden_dim,
            enc_pre_depth=args.encoder_pre_depth,
            enc_post_depth=args.encoder_post_depth,
            dec_depth=args.decoder_depth).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.l2_penalty)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_iterations)


#######################################################################


#training
for iteration in range(args.num_iterations):
    t_start = time.time()
    model.train()
    optimizer.zero_grad()
    batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device=device)
    outs = model(batch, num_samples=args.num_samples)
    outs['loss'].backward()
    optimizer.step()
    scheduler.step()
    t_end = time.time()

    if iteration % args.save_freq == 0 or iteration == args.num_iterations-1:
        model_path = os.path.join('train-results', 'seed-{}'.format(args.train_seed))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(model, os.path.join(model_path, model_type+'.pt'))
        with open(os.path.join(model_path, 'train-loss.txt'), 'a') as f:
            f.write('iteration: {}, time: {:.2f}, train log-likelihood: {:.4f}\n'.format(
                iteration, t_end - t_start, -outs['loss'].item()))
            # f.write('-'*50+'\n')


with open(os.path.join(model_path, 'train-loss.txt'), 'a') as f:
    f.write('\nSeed: {}\n'.format(args.train_seed))
    f.write('CPU: {}\n'.format(platform.processor()))
    if device == torch.device('cuda'):
        device_idx = torch.cuda.current_device()
        f.write('GPU: {}\n'.format(torch.cuda.get_device_properties(device_idx)))








