import torch
import torch.nn as nn
from CQNP.modules import *


TAU_MAX, TAU_MIN = 0.999, 0.001 #avoiding numerical issues

class CQNP(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super(CQNP, self).__init__()
        self._encoder = Encoder(encoder_dim)
        self._decoder = Decoder(decoder_dim)

    def forward(self, batch, tau=None, num_tau=50):
        hidden = self._encoder(batch.context_x, batch.context_y)

        batch_size, set_size, _ = batch.x_values.shape
        _, num_context, output_dim = batch.context_y.shape
        
        if tau is None:
            tau = TAU_MIN + (TAU_MAX - TAU_MIN) \
                * torch.rand((batch_size, num_tau, set_size, 1), 
                    device=batch.x_values.device)
        else:
            _, num_tau, _, _ = tau.shape

        representation = torch.tile(hidden[:, None, None, :],
                                    dims=[1, num_tau, set_size, 1])
        x_values = torch.tile(batch.x_values[:, None, :, :],
                              dims=[1, num_tau, 1, 1])
        
        shape = [batch_size*num_tau, set_size, -1]
        representation = representation.view(shape)
        x_values = x_values.view(shape)
        tau = tau.view(shape)

        loc, scale, mixture_weights = self._decoder(representation, x_values, tau)

        shape = [batch_size, num_tau, set_size, output_dim]
        loc = loc.view(shape)
        scale = scale.view(shape)
        mixture_weights = mixture_weights.view(shape)
        tau = torch.tile(tau.view(shape[:-1]+[1]),
            dims=[1, 1, 1, shape[-1]])

        dist = UMAL(loc, scale, tau, mixture_weights)
        log_likelihood = dist.log_prob(batch.y_values).sum(-1)

        output = {'dist': dist}
        if self.training:
            output['loss'] = - log_likelihood.mean(-1).mean()
        else:
            output['ctx_ll'] = log_likelihood[...,:num_context].mean(-1).mean()
            output['tar_ll'] = log_likelihood[...,num_context:].mean(-1).mean()

        return output









