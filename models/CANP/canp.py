import torch
import torch.nn as nn
from CANP.modules import *


class CANP(nn.Module):
    def __init__(self, encoder_dims, decoder_dims, attention):
        super(CANP, self).__init__()
        self._encoder = Encoder(encoder_dims, attention)
        self._decoder = Decoder(decoder_dims)

    def forward(self, batch):
        representation = self._encoder(batch.context_x, batch.context_y, batch.x_values)
        loc, scale = self._decoder(representation, batch.x_values)

        dist = torch.distributions.normal.Normal(loc=loc, scale=scale)

        log_likelihood = dist.log_prob(batch.y_values).sum(-1)

        output = {'dist': dist}
        if self.training:
            output['loss'] = -log_likelihood.mean(-1).mean()
        else:
            _, num_context, _ = batch.context_x.shape
            output['ctx_ll'] = log_likelihood[...,:num_context].mean(-1).mean()
            output['tar_ll'] = log_likelihood[...,num_context:].mean(-1).mean()
            
        return output