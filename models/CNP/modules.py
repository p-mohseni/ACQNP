import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 dimensions):
        super(MLP, self).__init__()
        self._dimensions = dimensions
        modules = []
        for i in range(len(dimensions) - 2):
            modules.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self._net = nn.Sequential(*modules)

    def forward(self, input):
        batch_size, set_size, filter_size = input.shape
        input = input.reshape((batch_size * set_size, -1))
        output = self._net(input)
        output = output.view(batch_size, set_size, -1)
        return output


class Encoder(nn.Module):
    def __init__(self,
                 dimensions):
        super(Encoder, self).__init__()
        self._dimensions = dimensions

        self._mlp = MLP(dimensions)

    def forward(self, context_x, context_y):
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        hidden = self._mlp(encoder_input)
        representation = torch.mean(hidden, dim=1, keepdim=True)
        return representation


class Decoder(nn.Module):
    def __init__(self,
                 dimensions):
        super(Decoder, self).__init__()
        self._dimensions = dimensions
        self._mlp = MLP(dimensions)

        self._softplus = nn.Softplus()

    def forward(self, representation, target_x):
        hidden = torch.cat((representation, target_x), dim=-1)
        hidden = self._mlp(hidden)
        loc, log_scale = torch.chunk(hidden, chunks=2, dim=-1)
        scale = 0.1 + 0.9 * self._softplus(log_scale)
        return loc, scale



