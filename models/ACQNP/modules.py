import torch
import torch.nn as nn


class AsymmetricLaplacian(object):
    def __init__(self, loc, scale, tau):
        self._loc = loc
        self._scale = scale
        self._tau = tau

    @property
    def mean(self):
        return self._loc + (1 - 2 * self._tau) \
               / (self._tau * (1 - self._tau)) * self._scale

    @property
    def variance(self):
      return (1 - 2 * self._tau + 2 * torch.square(self._tau)) \
             / (self._tau * (1 - self._tau)) * torch.square(self._scale)

    @property
    def stddev(self):
        return torch.sqrt(self.variance)

    def log_prob(self, y):
        log_p = torch.log(self._tau) + torch.log(1 - self._tau) \
                - torch.log(self._scale) \
                - (y - self._loc) * (self._tau - (y < self._loc).float()) / self._scale
        return log_p


class UMAL(object):
    def __init__(self, loc, scale, tau,
                 mixture_weights=None):
        if mixture_weights is None:
            mixture_weights = torch.ones_like(loc)
        self._mixture_weights = nn.functional.softmax(mixture_weights, dim=1)
        self._ALD = AsymmetricLaplacian(loc, scale, tau)
        
    @property
    def component_mean(self):
        return self._ALD.mean

    @property
    def component_stddev(self):
        return self._ALD.stddev

    @property
    def component_weight(self):
        return self._mixture_weights

    @property
    def mean(self):
        return torch.sum(torch.mul(self._ALD.mean,
                                   self._mixture_weights),
                         dim=1)

    def log_prob(self, y):
        num_tau = self._ALD._tau.shape[1]
        y = torch.tile(torch.unsqueeze(y, dim=1),
                       dims=[1, num_tau, 1, 1])
        log_p = torch.logsumexp(torch.log(self._mixture_weights+1e-10) + 
                                self._ALD.log_prob(y), dim=1)
        return log_p


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
        representation = torch.mean(hidden, dim=1)
        return representation


class Decoder(nn.Module):
    def __init__(self,
                 dimensions):
        super(Decoder, self).__init__()
        self._dimensions = dimensions
        self._mlp = MLP(dimensions)

        self._softplus = nn.Softplus()

    def forward(self, representation, target_x, tau):
        hidden = torch.cat((representation, target_x, tau), dim=-1)
        hidden = self._mlp(hidden)
        loc, log_scale, mixture_weights = torch.chunk(hidden, chunks=3, dim=-1)
        scale = 0.01 + 0.99 * self._softplus(log_scale)
        return loc, scale, mixture_weights

