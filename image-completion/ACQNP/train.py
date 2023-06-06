import os
import sys
import time
import math
import argparse
import platform
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.getcwd(), '..'))
from utils import *
sys.path.insert(0, os.path.join(os.getcwd(), '../../models'))
from ACQNP.acqnp import ACQNP


#######################################################################


parser = argparse.ArgumentParser()

parser.add_argument('--train_seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--l2_penalty', type=float, default=1e-5)
parser.add_argument('--num_tau_samples', type=int, default=25)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=int(5e3))

parser.add_argument('--input_dim', type=int, default=2)
parser.add_argument('--output_dim', type=int, default=3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--rep_dim', type=int, default=128)
parser.add_argument('--encoder_depth', type=int, default=3)
parser.add_argument('--decoder_depth', type=int, default=3)
parser.add_argument('--adaptor_depth', type=int, default=3)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Fix seed (optional)
torch.manual_seed(args.train_seed)
if device == torch.device('cuda'):
    torch.cuda.manual_seed(args.train_seed)


#######################################################################


transforms = torchvision.transforms.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

dataset = torchvision.datasets.SVHN('./../data',
                                    split='train',
                                    transform=transforms,
                                    download=True)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True)


#######################################################################

model_type = 'ACQNP'

encoder_sizes = [args.input_dim + args.output_dim] +\
                [args.hidden_dim]*args.encoder_depth +\
                [args.rep_dim]
decoder_sizes = [args.rep_dim + args.input_dim + 1] +\
                [args.hidden_dim]*args.decoder_depth +\
                [3*args.output_dim]
adaptor_sizes = [args.rep_dim + args.input_dim + 1] +\
                [args.hidden_dim]*args.adaptor_depth + [1]


model = ACQNP(encoder_sizes, decoder_sizes, adaptor_sizes).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.l2_penalty)


#######################################################################


#training
model.train()
with torch.enable_grad():
    log_p_list = []

    for epoch in range(args.num_epochs):
        batch_log_p = []
        t_start = time.time()
        for _, (img, label) in enumerate(dataloader):
            batch = img_to_task(img, device=device)
            optimizer.zero_grad()
            outs = model(batch, num_tau=args.num_tau_samples)
            outs['loss'].backward()
            optimizer.step()
            batch_log_p.append(-outs['loss'].item())
            break

        t_end = time.time()
        log_p_list.append(sum(batch_log_p) / len(batch_log_p))

        if epoch % args.save_freq == 0 or epoch == args.num_epochs-1:
            model_path = os.path.join('train-results', 'seed-{}'.format(args.train_seed))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model, os.path.join(model_path, model_type+'.pt'))
            with open(os.path.join(model_path, 'train-loss.txt'), 'a') as f:
                f.write('epoch: {}, time: {:.2f}, train log-likelihood: {:.4f}\n'.format(epoch, t_end - t_start, log_p_list[-1]))
                # f.write('-'*50+'\n')


with open(os.path.join(model_path, 'train-loss.txt'), 'a') as f:
    f.write('\nSeed: {}\n'.format(args.train_seed))
    f.write('CPU: {}\n'.format(platform.processor()))
    if device == torch.device('cuda'):
        device_idx = torch.cuda.current_device()
        f.write('GPU: {}\n'.format(torch.cuda.get_device_properties(device_idx)))











