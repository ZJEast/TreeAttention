# -*- coding: utf-8 -*-

import torch
from torch import nn, Tensor


class Coder(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.n_bit = 8
    
    def build(self):
        mul = torch.arange(self.n_bit)
        mul = self.n_bit - 1 - mul
        mul = (2 ** mul).long()
        self.mul = mul.view(1, -1)
    
    def bool2long(self, x: Tensor):
        shape = x.shape
        assert x.dtype == torch.bool
        x = x.view(-1, self.n_bit)
        x = (x * self.mul).sum(dim=-1)
        x = x.view(shape[:-1])
        return x

    def long2bool(self, x: Tensor):
        shape = x.shape
        assert x.dtype == torch.long
        x = x.view(-1, 1)
        x = (x // self.mul) % 2
        x = x.view(shape + (self.n_bit, ))
        return x.bool()
    

class TreeLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.q_dim = 8
        self.n_head = 4
        self.depth = 8
        self.v_dim = 8
        self.k_bias = 3.0
    
    def build(self):
        nv = int(2 ** self.depth)
        self.key = nn.Parameter(torch.randn((self.n_head, nv - 1, 2*self.q_dim)) + self.k_bias, requires_grad=True)
        self.value = nn.Parameter(torch.randn((self.n_head, nv, self.v_dim)) + self.k_bias, requires_grad=True)
    
    def forward(self, q: Tensor, qs: Tensor):
        assert q.dtype == torch.bool
        B, Q = q.shape
        H = self.n_head
        nv = int(2 ** self.depth)
        nk = nv - 1

        q = q.view(B, 1, Q).expand(B, H, Q)
        q = q.reshape(q.shape).view(-1, Q)
        qs = (2 * q.float() - 1) * qs
        ix_h = torch.arange(B * H, device=q.device) % H
        ix = torch.zeros(B * H, device=q.device, dtype=torch.long)
        support = None

        for depth in range(self.depth):
            ix_k = ix_h * nk + int(2 ** depth) - 1 + ix
            key_w = self.key.view(-1, 2 * Q).index_select(0, ix_k)
            key = torch.bernoulli(key_w.sigmoid()).bool().view(-1, Q, 2)
            key1 = key[:, :, 0]
            key2 = key[:, :, 1]
            key1 = -torch.logaddexp(-key1, qs)
            key2 = -torch.logaddexp(-key2, -qs)
            lor_s = torch.logaddexp(key1, key2)
            lor_s = -(-lor_s).logsumexp(dim=-1)
            lor = torch.bernoulli(lor_s.sigmoid())
            lor_s = (2 * lor - 1) * lor_s
            ix = 2 * ix + lor.long()

            if support is None:
                support = lor_s
            else:
                support = -torch.logaddexp(-support, -lor_s)
        
        value_w = self.value.view(-1, self.v_dim).index_select(0, ix_h * nv + ix)
        value_w = value_w.view(B, H, -1)
        value_b = torch.bernoulli(value_w.sigmoid())
        value_s = (2 * value_b - 1) * value_w
        value_s = -torch.logaddexp(-value_s, -support.view(B, H, 1))

        return value_b.bool(), value_s


class SSMLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.n_head = 4
        self.n_bit = 8
        self.in_dim = 8
        self.ffn_v_dim = 4
        self.ffn_depth = 8
        self.ffn_k_bias = 3.0
    
    def build(self):
        ffn = TreeLayer()
        ffn.q_dim = self.in_dim
        ffn.n_head = self.n_head
        ffn.depth = self.ffn_depth
        ffn.v_dim = self.ffn_v_dim
        ffn.k_bias = self.ffn_k_bias
        ffn.build()
        self.ffn = ffn

        self.table_len = int(2 ** self.n_bit)
        table = nn.Parameter(
            torch.randn((1, int(2 ** self.ffn_v_dim) * self.n_head, self.table_len, self.n_bit)), 
            requires_grad=True)
        self.table = table

        self.state0 = nn.Parameter(
            torch.randn((1, self.n_head, self.n_bit)), 
            requires_grad=True)

        coder = Coder()
        coder.n_bit = self.n_bit
        coder.build()
        self.ssm_coder = coder

        coder = Coder()
        coder.n_bit = self.ffn_v_dim
        coder.build()
        self.ffn_coder = coder
    
    def mul(self, x: Tensor, xs: Tensor, y: Tensor, ys: Tensor):
        shape = x.shape
        T = self.table_len
        x = x.view(-1, T)
        B, T = x.shape
        assert x.dtype == torch.long
        assert (x.shape == xs.shape) and (x.shape == y.shape) and (x.shape == ys.shape)

        z = y.gather(1, x)
        zs = - torch.logaddexp(-xs, -ys.gather(1, x))

        return z.view(shape), zs.view(shape)

    def forward(self, x: Tensor, xs: Tensor):
        B, L, D = x.shape
        assert D == self.in_dim
        assert x.shape == xs.shape
        H = self.n_head
        V = self.ffn_v_dim
        T = self.table_len

        x, xs = self.ffn.forward(x.view(-1, D), xs.view(-1, D))
        x = x.view(B, L, H, V)
        xs = xs.view(x.shape)
        x = self.ffn_coder.bool2long(x)
        xs = -(-xs).logsumexp(dim=-1)

        table = torch.bernoulli(self.table.expand(B, -1, -1, -1).sigmoid())
        table_s = self.table * (2 * table - 1)
        table = self.ssm_coder.bool2long(table.bool())
        table_s = -(-table_s).logsumexp(dim=-1)

        _x = (torch.arange(L * H, device=x.device).view(1, -1, 1) % H) * int(2 ** V)
        _x = x.view(B, L * H, 1).expand(-1, -1, T) + _x
        y = table.gather(1, _x)
        ys = table_s.gather(1, _x)
        y = y.view(B, L, H, T)
        ys = ys.view(B, L, H, T)
        ys = -torch.logaddexp(-ys, -xs.view(B, L, H, 1))

        i = 1
        while i < L:
            y1 = y[:, 0:L-i, :, :]
            y1s = ys[:, 0:L-i, :, :]
            y2 = y[:, i:L, :, :]
            y2s = ys[:, i:L, :, :]
            y3, y3s = self.mul(y1, y1s, y2, y2s)
            y = torch.cat([y[:, :i, :, :], y3], dim=1)
            ys = torch.cat([ys[:, :i, :, :], y3s], dim=1)
            i = 2 * i

        s0 = torch.bernoulli(self.state0.expand(B, -1, -1).sigmoid())
        s0s = self.state0 * (2 * s0 - 1)
        s0 = self.ssm_coder.bool2long(s0.bool())
        s0s = -(-s0s).logsumexp(dim=-1)

        _s0 = s0.view(B, 1, H).expand(B, L, H).reshape(-1, 1)
        z = y.view(-1, T).gather(1, _s0).view(B, L, H)
        zs = ys.view(-1, T).gather(1, _s0).view(B, L, H)
        zs = -torch.logaddexp(-s0s.view(B, 1, H), -zs)

        z = self.ssm_coder.long2bool(z)
        zs = zs.view(B, L, H, 1).expand(z.shape)
        z = z.view(B, L, -1)
        zs = zs.reshape(z.shape)

        return z, zs


class SSM(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.n_head = 4
        self.n_bit = 8
        self.in_dim = 8
        self.ffn_v_dim = 4
        self.ffn_depth = 8
        self.ffn_k_bias = 3.0
        self.out_dim = 8
        self.n_layers = 8
        self.out_ffn_depth = 1
    
    def build(self):
        layers = []
        dim = self.in_dim
        for i in range(self.n_layers):
            layer = SSMLayer()
            layer.n_head = self.n_head
            layer.n_bit = self.n_bit
            layer.in_dim = dim
            layer.ffn_v_dim = self.ffn_v_dim
            layer.ffn_depth = self.ffn_depth
            layer.ffn_k_bias = self.ffn_k_bias
            layer.build()
            layers.append(layer)
            dim = layer.n_head * layer.n_bit
        self.layers = nn.ModuleList(layers)

        ffn = TreeLayer()
        ffn.q_dim = self.n_layers * self.n_head * self.n_bit
        ffn.n_head = self.out_dim
        ffn.depth = self.out_ffn_depth
        ffn.v_dim = 1
        ffn.k_bias = self.ffn_k_bias
        ffn.build()
        self.out_ffn = ffn
    
    def forward(self, x: Tensor, xs: Tensor = None):
        if xs is None:
            xs = torch.full(x.shape, torch.inf, device=x.device)
        B, L, D = x.shape
        assert D == self.in_dim
        assert x.shape == xs.shape

        y = []
        ys = []
        for i in range(self.n_layers):
            layer: SSMLayer = self.layers[i]
            x, xs = layer.forward(x, xs)
            y.append(x)
            ys.append(xs)
        y = torch.cat(y, dim=-1)
        ys = torch.cat(ys, dim=-1)

        y, ys = self.out_ffn.forward(y.view(B * L, -1), ys.view(B * L, -1))

        return y.view(B, L, -1), ys.view(B, L, -1)