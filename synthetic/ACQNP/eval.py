import os
import sys
import time
import torch
import argparse
import platform

sys.path.insert(0, os.path.join(os.getcwd(), '..'))
from utils import *
sys.path.insert(0, os.path.join(os.getcwd(), '../../models'))
from ACQNP.acqnp import ACQNP


#######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--eval_seed', type=int, default=43)
parser.add_argument('--model_seed', type=int, default=0)
parser.add_argument('--num_tau_samples', type=int, default=100)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Fix seed (optional)
torch.manual_seed(args.eval_seed)
if device == torch.device('cuda'):
    torch.cuda.manual_seed(args.eval_seed)


#######################################################################

root = os.getcwd()
test_data = torch.load(os.path.join(root, '..', 'eval-data',  'test-data.tar'))

model_type = 'ACQNP'
model_path = os.path.join(root, 'train-results', 'seed-{}'.format(
    args.model_seed), f'{model_type}.pt'
)
model = torch.load(model_path)


#######################################################################

#testing
model.eval()
with torch.no_grad():
    target_log_p = []
    context_log_p = []
    t_start = time.time()
    for batch in test_data:
        if device == torch.device('cuda'):
            batch = to_cuda(batch)
        outs = model(batch, num_tau=args.num_tau_samples)
        target_log_p.append(outs['tar_ll'].item())
        context_log_p.append(outs['ctx_ll'].item())

    t_end = time.time()
    
    eval_target_log_p = sum(target_log_p) / len(target_log_p)
    eval_context_log_p = sum(context_log_p) / len(context_log_p)


with open('eval-results.txt', 'a') as f:
    f.write('context ll: {}, target ll: {}\n\n'.format(eval_context_log_p, eval_target_log_p))
    f.write('Model seed: {}, Eval seed: {}\n'.format(args.model_seed, args.eval_seed))
    f.write('CPU: {}\n'.format(platform.processor()))
    if device == torch.device('cuda'):
        device_idx = torch.cuda.current_device()
        f.write('GPU: {}\n'.format(torch.cuda.get_device_properties(device_idx)))










