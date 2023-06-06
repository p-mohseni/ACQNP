import torch
import torch.nn as nn
from ACQNP.modules import *


class ACQNP(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, adaptor_dim):
        super(ACQNP, self).__init__()
        self._encoder = Encoder(encoder_dim)
        self._decoder = Decoder(decoder_dim)
        self._adaptor = MLP(adaptor_dim)

    def forward(self, batch, tau=None, num_tau=50):
        hidden = self._encoder(batch.context_x, batch.context_y)

        batch_size, set_size, _ = batch.x_values.shape
        _, num_context, output_dim = batch.context_y.shape
        
        if tau is None:
            tau = torch.rand((batch_size, num_tau, set_size, 1),
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

        tau = torch.sigmoid(
            self._adaptor(torch.cat((representation, x_values, tau), dim=-1))
            )

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









