# -*- coding: utf-8 -*-

import torch
from torch import nn

Tensor = torch.Tensor


def and_(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    return -torch.logaddexp(-x, -y)


def or_(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    return torch.logaddexp(x, y)
    

def match(query: Tensor, key: Tensor):
    B, q_dim = query.shape
    assert key.shape == (B, 2, q_dim)

    k1 = key[:, 0, :].view(query.shape)
    k2 = key[:, 1, :].view(query.shape)
    _support = or_(and_(k1, query), and_(k2, -query))
    _support = -(-_support).logsumexp(dim=-1)

    return _support.view(B)


def equal(x: Tensor, y: Tensor):
    B, D = x.shape
    assert x.shape == y.shape
    support = - (- (2 * y.float() - 1) * x).logsumexp(dim=-1)
    return support.view(B)


def target_loss(support: Tensor, value: Tensor, target: Tensor):
    B, = support.shape
    _, D = value.shape
    assert value.shape == (B, D)
    assert target.shape == (B, D)

    _support = equal(value, target)
    _support = and_(_support, support)
    _support = or_(_support, - support)

    return - nn.functional.logsigmoid(_support).mean()


def compare(x: Tensor, y: Tensor, sign=-1):
    B, D = x.shape
    assert x.shape == y.shape

    eq = - (- (2 * y.float() - 1) * x).logcumsumexp(dim=-1)
    support1 = and_(-eq[:, 0], sign * x[:, 0])
    if D <= 1:
        return support1
    
    eq1 = eq[:, 0:D-1]
    eq2 = eq[:, 1:D]
    support2 = and_(eq1, -eq2)
    support2 = and_(support2, sign * x[:, 1:D])
    support2 = torch.logsumexp(support2, dim=-1)

    return or_(support1, support2)


def less_than(x: Tensor, y: Tensor):
    return compare(x, y, -1)


def greater_than(x: Tensor, y: Tensor):
    return compare(x, y, 1)


def tree_attention(
        query: Tensor, 
        depth: int, 
        key: Tensor, 
        value: Tensor, 
        k_id: Tensor = None, 
        v_id: Tensor = None,
        random: bool = True
    ):

    B, q_dim = query.shape
    v_dim = value.shape[-1]
    k_num = int(2**depth) - 1
    v_num = int(2**depth)

    if k_id is None:
        k_id = torch.zeros((B,), device=query.device, dtype=torch.long)
    if v_id is None:
        v_id = torch.zeros((B,), device=query.device, dtype=torch.long)


    assert depth > 0
    assert k_id.shape == (B,) and k_id.dtype == torch.long
    assert v_id.shape == (B,) and v_id.dtype == torch.long

    key = key.view(-1, 2*q_dim)
    value = value.view(-1, v_dim)

    device = query.device
    ix = torch.zeros((B,), device=device, dtype=torch.long)
    support: Tensor = None
    for d in range(depth):
        _ix: Tensor = int(2**d) - 1 + ix
        _key = key.index_select(0, k_id * k_num + _ix)
        _key = _key.view(B, 2, q_dim)
        _support = match(query, _key)
        if random:
            _ix = torch.bernoulli(_support.sigmoid().detach()).view(B)
        else:
            _ix = (_support.detach() >= 0).float().view(B)
        ix = 2 * ix + _ix.long()
        _support = (2 * _ix - 1) * _support
        if support is None:
            support = _support
        else:
            support = and_(support, _support)

    _ix = ix
    value = value.index_select(0, v_id * v_num + _ix).view(B, v_dim)

    return support, value


class Coder(nn.Module):

    def __init__(self, lower_bound=0.1):
        nn.Module.__init__(self)
        self.lower_bound = lower_bound
        self.abs = nn.Parameter(torch.tensor(lower_bound), requires_grad=True)
    
    def bool2support(self, x: Tensor):
        return (2 * x.float() - 1) * self.abs
    
    def support2bool(self, x: Tensor, random=True):
        B, D = x.shape
        if random:
            b = torch.bernoulli(x.sigmoid())
        else:
            b = (x >= 0).float()
        support = (2 * b - 1) * x
        support = - (- support).logsumexp(dim=-1)
        return support, b.bool()
    
    def check(self):
        data = self.abs.data
        self.abs.data = torch.maximum(data, torch.full_like(data, self.lower_bound))
    
    def long2bool(self, x: Tensor, n_bits=8):
        D = n_bits

        div = D - 1 - torch.arange(D, device=x.device)
        div = torch.pow(torch.full_like(div, 2), div).long()

        x = x.flatten()
        B, = x.shape
        x = x.view(B, 1).expand(B, D)
        div = div.view(1, D).expand(B, D)
        bits = (x // div) % 2

        return bits.bool()
    
    def bool2long(self, x: Tensor, n_bits=8):
        D = n_bits

        mul = D - 1 - torch.arange(D, device=x.device)
        mul = torch.pow(torch.full_like(mul, 2), mul).long()

        x = x.view(-1, D)
        B, = x.shape
        mul = mul.view(1, D).expand(B, D)
        long = (x.long() * mul).sum(dim=-1)

        return long
    
    def float2long(self, x: Tensor, low=-1.0, high=1.0, n_bits=8):
        x = (x.clip(low, high) - low) / (high - low)
        x = x * int(2**n_bits - 1)
        return x.long()
    
    def long2float(self, x: Tensor, low=-1.0, high=1.0, n_bits=8):
        x = x.float() / (2**n_bits - 1)
        x = (high - low) * x + low
        return x


class TreeLayer(nn.Module):
    
    def __init__(self, depth, in_dim, out_dim, n_heads, bias=0):
        nn.Module.__init__(self)
        self.depth = depth
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads

        _2depth = int(2**depth)
        self.key = nn.Parameter(bias + torch.randn(n_heads, _2depth - 1, 2, in_dim), requires_grad=True)
        self.value = nn.Parameter(torch.randn(n_heads, _2depth, out_dim), requires_grad=True)
    
    def forward(self, query: Tensor, random=True):
        B, D = query.shape
        assert query.shape == (B, self.in_dim)
        H = self.n_heads

        query = query.view(B, 1, D).expand(B, H, D).reshape(-1, D)
        ix = torch.arange(H, device=query.device)
        ix = ix.view(1, H).expand(B, H).reshape(-1)

        support, value = tree_attention(query, self.depth, self.key, self.value, ix, ix, random=random)

        support = support.view(B, H)
        support = - (- support).logsumexp(dim=-1)

        return support.view(B), value