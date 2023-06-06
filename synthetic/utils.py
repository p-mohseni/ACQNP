import os
import math
import collections

import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributions import MultivariateNormal


Batch_Description = collections.namedtuple(
    "Batch_Description",
    ("y_values", "x_values",
     "target_y", "target_x",
     "context_y", "context_x")
    )


def to_cuda(batch):
    return Batch_Description(
            x_values=batch.x_values.to('cuda'),
            y_values=batch.y_values.to('cuda'),
            target_x=batch.target_x.to('cuda'),
            target_y=batch.target_y.to('cuda'),
            context_x=batch.context_x.to('cuda'),
            context_y=batch.context_y.to('cuda'))


class CurveSampler(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def sample(self,
            batch_size=16,
            num_context=None,
            max_num_context=None,
            num_points=None,
            max_num_points=50,
            s_range=(-2, 2),
            device='cpu'):

        # if max_num_context is not None:
        #     assert max_num_points >= max_num_context, \
        #        f"number of context cannnot be greater than maximum number of data points"

        max_num_context = max_num_context or max_num_points
        num_points = num_points or torch.randint(low=6, high=max_num_points+1, size=[1]).item()
        num_context = num_context or torch.randint(low=3, high=min(num_points, max_num_context)-2, size=[1]).item()

        s = s_range[0] + (s_range[1] - s_range[0]) \
            * torch.rand([batch_size, num_points, 1], device=device)

        # batch_size * num_points * num_points
        x_values, y_values = self._kernel(s)

        context_x = x_values[:,:num_context]
        target_x = x_values[:,num_context:]
        context_y = y_values[:,:num_context]
        target_y = y_values[:,num_context:]

        return Batch_Description(
            x_values=x_values, y_values=y_values,
            target_x=target_x, target_y=target_y,
            context_x=context_x, context_y=context_y)


class DoubleSineKernel(object):
    def __init__(self, amp_range=(0.5, 1.5), freq_range=(1, 3)):
        self._amp_range = amp_range
        self._freq_range = freq_range

    # s: batch_size * num_points * dim
    def __call__(self, s):
        amp = self._amp_range[0] + (self._amp_range[1] - self._amp_range[0]) \
              * torch.rand([2, s.shape[0], 1, 1], device=s.device)
        freq = self._freq_range[0] + (self._freq_range[1] - self._freq_range[0]) \
               * torch.rand([2, s.shape[0], 1, 1], device=s.device)
        
        amp = torch.tile(amp, dims=[1, 1, s.shape[1], 1])
        freq = torch.tile(freq, dims=[1, 1, s.shape[1], 1])

        y_1 = amp[0] * torch.sin(freq[0] * s)
        y_2 = amp[1] * torch.cos(freq[1] * s)
        p_assignment = torch.rand(size=s.shape, device=s.device)

        x_values = s
        y_values = torch.where(p_assignment < 0.5, y_1, y_2)

        return x_values, y_values


class CircularKernel(object):
    def __init__(self, radius_range=(0.5, 1.5), shift_range=(-0.5, 0.5)):
        self._radius_range = radius_range
        self._shift_range = shift_range

    # s: batch_size * num_points * dim
    def __call__(self, s):
        radius = self._radius_range[0] + (self._radius_range[1] - self._radius_range[0]) \
                 * torch.rand([s.shape[0], 1, 1], device=s.device)
        shift = self._shift_range[0] + (self._shift_range[1] - self._shift_range[0]) \
                 * torch.rand([s.shape[0], 1, 1], device=s.device)
        
        radius = torch.tile(radius, dims=[1, s.shape[1], 1])
        shift = torch.tile(shift, dims=[1, s.shape[1], 1])

        x_values = radius * torch.cos(s) + shift
        y_values = radius * torch.sin(s) + shift

        return x_values, y_values


class LissajousKernel(object):
    def __init__(self, x_amp_range=(1, 2), y_amp_range=(1, 2),
                 freq_range=(0.5, 2), shift_range=(0, 2)):
        self._x_amp_range = x_amp_range
        self._y_amp_range = y_amp_range
        self._freq_range = freq_range
        self._shift_range = shift_range

    # s: batch_size * num_points * dim
    def __call__(self, s):
        x_amp = self._x_amp_range[0] + (self._x_amp_range[1] - self._x_amp_range[0]) \
                * torch.rand([s.shape[0], 1, 1], device=s.device)
        y_amp = self._y_amp_range[0] + (self._y_amp_range[1] - self._y_amp_range[0]) \
                * torch.rand([s.shape[0], 1, 1], device=s.device)
        freq = self._freq_range[0] + (self._freq_range[1] - self._freq_range[0]) \
               * torch.rand([s.shape[0], 1, 1], device=s.device)
        shift = self._shift_range[0] + (self._shift_range[1] - self._shift_range[0]) \
                * torch.rand([s.shape[0], 1, 1], device=s.device)
        
        x_amp = torch.tile(x_amp, dims=[1, s.shape[1], 1])
        y_amp = torch.tile(y_amp, dims=[1, s.shape[1], 1])
        freq = torch.tile(freq, dims=[1, s.shape[1], 1])
        shift = torch.tile(shift, dims=[1, s.shape[1], 1])

        x_values = x_amp * torch.sin(freq * s + shift)
        y_values = y_amp * torch.sin(s)

        return x_values, y_values


class SawtoothKernel(object):
    def __init__(self, amp_range=(1, 2), freq_range=(1, 3),
                 shift_range=(-2, 2), trunc_range=(10, 20)):
        self._amp_range = amp_range
        self._freq_range = freq_range
        self._shift_range = shift_range
        self._trunc_range = trunc_range

    # s: batch_size * num_points * dim
    def __call__(self, s):
        amp = self._amp_range[0] + (self._amp_range[1] - self._amp_range[0]) \
              * torch.rand([s.shape[0]], device=s.device)
        freq = self._freq_range[0] + (self._freq_range[1] - self._freq_range[0]) \
               * torch.rand([s.shape[0]], device=s.device)
        shift = self._shift_range[0] + (self._shift_range[1] - self._shift_range[0]) \
                * torch.rand([s.shape[0]], device=s.device)
        trunc = torch.randint(low=self._trunc_range[0], high=self._trunc_range[1]+1, size=[s.shape[0]],device=s.device)

        y_values = []
        for i in range(s.shape[0]):
            x = torch.tile(s[i, :, :, None], dims=[1, 1, trunc[i]]) + shift[i]
            k = torch.tile(torch.reshape(torch.arange(start=1, end=trunc[i] + 1, device=s.device), shape=(1, 1, -1)),
                           dims=[s.shape[1], 1, 1])
            y = amp[i]/2 - amp[i] / math.pi * torch.sum((-1)**k * torch.sin(2 * math.pi * k * freq[i] * x) / k, dim=-1)
            y -= 0.5 * amp[i]
            y_values.append(y)
        
        x_values = s
        y_values = torch.stack(y_values)

        return x_values, y_values


class RBFKernel(object):
    def __init__(self, sigma_eps=2e-2, length=0.25, scale=0.75):
        self._sigma_eps = sigma_eps
        self._length = length
        self._scale = scale

    # s: batch_size * num_points * dim
    def __call__(self, s):
        # batch_size * num_points * num_points * dim
        dist = (s.unsqueeze(-2) - s.unsqueeze(-3))/self._length

        # batch_size * num_points * num_points
        cov = (self._scale**2) * torch.exp(-0.5 * dist.pow(2).sum(-1)) \
              + self._sigma_eps**2 * torch.eye(s.shape[-2]).to(s.device)
        mean = torch.zeros(s.shape[0], s.shape[1], device=s.device)

        x_values = s
        y_values = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        return x_values, y_values


class Matern52Kernel(object):
    def __init__(self, sigma_eps=2e-2, length=0.25, scale=0.75):
        self._sigma_eps = sigma_eps
        self._length = length
        self._scale = scale

    # s: batch_size * num_points * dim
    def __call__(self, s):
        # batch_size * num_points * num_points
        dist = torch.norm((s.unsqueeze(-2) - s.unsqueeze(-3))/self._length, dim=-1)

        cov = (self._scale**2) * (1 + math.sqrt(5.0)*dist + 5.0*dist.pow(2)/3.0) \
              * torch.exp(-math.sqrt(5.0) * dist) \
              + self._sigma_eps**2 * torch.eye(s.shape[-2]).to(s.device)
        mean = torch.zeros(s.shape[0], s.shape[1], device=s.device)

        x_values = s
        y_values = MultivariateNormal(mean, cov).rsample().unsqueeze(-1)

        return x_values, y_values


