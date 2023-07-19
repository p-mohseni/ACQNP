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
                 dimensions,
                 attention):
        super(Encoder, self).__init__()
        self._dimensions = dimensions

        self._mlp = MLP(dimensions)
        self._attention = attention

    def forward(self, context_x, context_y, target_x):
        encoder_input = torch.cat((context_x, context_y), dim=-1)
        hidden = self._mlp(encoder_input)
        representation = self._attention(context_x, target_x, hidden)
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


class UniformAttention():
    def __call__(self, q, v):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)  # [B,1,d_v]
        rep = torch.tile(rep, dims=[1, total_points, 1])
        return rep


class LaplaceAttention():
    def __init__(self,
                 scale,
                 normalize):
        self._scale = scale
        self._normalize = normalize

    def __call__(self, q, k, v):
        k = torch.unsqueeze(k, dim=1)  # [B,1,n,d_k]
        q = torch.unsqueeze(q, dim=2)  # [B,m,1,d_k]
        unnorm_weights = - torch.abs((k - q) / self._scale)  # [B,m,n,d_k]
        unnorm_weights = torch.sum(unnorm_weights, dim=-1)  # [B,m,n]
        if self._normalise:
            weight_fn = nn.Softmax(dim=-1)
        else:
            weight_fn = lambda x: 1 + torch.tanh(x)
        weights = weight_fn(unnorm_weights)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
        return rep


class DotProductAttention():
    def __init__(self,
                 normalize):
        self._normalize = normalize

    def __call__(self, q, k, v):
        d_k = q.shape[-1]
        scale = math.sqrt(float(d_k))
        unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale # [B,m,n]
        if self._normalize:
            weight_fn = nn.Softmax(dim=-1)
        else:
            weight_fn = nn.Sigmoid()
        weights = weight_fn(unnorm_weights)  # [B,m,n]
        rep = torch.einsum('bik,bkj->bij', weights, v)  # [B,m,d_v]
        return rep


class MultiheadAttention(nn.Module):
    def __init__(self,
                 query_feat_size,
                 key_feat_size,
                 value_feat_size,
                 ouput_size,
                 num_heads):
        super(MultiheadAttention, self).__init__()
        self._query_feat_size = query_feat_size
        self._key_feat_size = key_feat_size
        self._value_feat_size = value_feat_size
        self._ouput_size = ouput_size
        self._num_heads = num_heads

        head_size = value_feat_size // num_heads

        query_layers = []
        key_layers = []
        value_layers = []
        output_layers = []

        for i in range(self._num_heads):
            query_layers.append(
                nn.Conv1d(in_channels=query_feat_size,
                          out_channels=head_size,
                          kernel_size=1,
                          bias=False)
                )
            key_layers.append(
                nn.Conv1d(in_channels=key_feat_size,
                          out_channels=head_size,
                          kernel_size=1,
                          bias=False)
                )
            value_layers.append(
                nn.Conv1d(in_channels=value_feat_size,
                          out_channels=head_size,
                          kernel_size=1,
                          bias=False)
                )

            output_layers.append(
                nn.Conv1d(in_channels=head_size,
                          out_channels=ouput_size,
                          kernel_size=1,
                          bias=False)
                )
        self._query_layers = nn.ModuleList(query_layers)
        self._key_layers = nn.ModuleList(key_layers)
        self._value_layers = nn.ModuleList(value_layers)
        self._output_layers = nn.ModuleList(output_layers)

        self._dot_product_attention = DotProductAttention(normalize=True)

    def forward(self, q, k, v):
        rep = 0.0
        for h in range(self._num_heads):
            query_h = self._query_layers[h](q.permute([0, 2, 1]))
            key_h = self._key_layers[h](k.permute([0, 2, 1]))
            value_h = self._value_layers[h](v.permute([0, 2, 1]))
            out = self._dot_product_attention(query_h.permute([0, 2, 1]),
                                              key_h.permute([0, 2, 1]),
                                              value_h.permute([0, 2, 1]))
            rep += self._output_layers[h](out.permute([0, 2, 1])).permute([0, 2, 1])
        return rep


class Attention(nn.Module):
    def __init__(self,
                 rep='identity',
                 q_sizes=None,
                 k_sizes=None,
                 v_size=None,
                 out_size=None,
                 attention_type='uniform',
                 scale=1.,
                 normalise=True,
                 num_heads=8
                 ):
        super(Attention, self).__init__()
        self._rep = rep
        self._q_sizes = q_sizes
        self._k_sizes = k_sizes
        self._v_size = v_size
        self._out_size = out_size
        self._attention_type = attention_type
        
        if attention_type == 'multihead':
            self._num_heads = num_heads

        if rep == 'identity':
            pass
        elif rep == 'mlp':
            self._query_mlp = MLP(q_sizes)
            self._key_mlp = MLP(k_sizes)
        else:
            raise NameError("'rep' not among ['identity','mlp']")

        if attention_type == 'uniform':
            self._attention = UniformAttention()
        elif attention_type == 'laplace':
            self._attention = LaplaceAttention(scale,
                                               normalise)
        elif attention_type == 'dot_product':
            self._attention = DotProductAttention(normalise)
        elif attention_type == 'multihead':
            self._attention = MultiheadAttention(q_sizes[-1],
                                                 k_sizes[-1],
                                                 v_size,
                                                 out_size,
                                                 num_heads)
        else:
            raise NameError(("'attention_type' not among ['uniform','laplace','dot_product'"
                           ",'multihead']"))

    def forward(self, x1, x2, v):
        if self._rep == 'identity':
            k, q = (x1, x2)
        elif self._rep == 'mlp':
            # Pass through MLP
            k = self._key_mlp(x1)
            q = self._query_mlp(x2)

        if self._attention_type == 'uniform':
            rep = self._attention(q, v)
        elif self._attention_type == 'laplace':
            rep = self._attention(q, k, v)
        elif self._attention_type == 'dot_product':
            rep = self._attention(q, k, v)
        elif self._attention_type == 'multihead':
            rep = self._attention(q, k, v)
        else:
            raise NameError(("'attention_type' not among ['uniform','laplace','dot_product'"
                           ",'multihead']"))

        return rep



