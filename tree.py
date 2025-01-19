# -*- coding: utf-8 -*-

import torch
from torch import nn

Tensor = torch.Tensor


def op_and(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    return -torch.logaddexp(-x, -y)


def op_or(x: Tensor, y: Tensor):
    assert x.shape == y.shape
    return torch.logaddexp(x, y)
    

def op_match(query: Tensor, key: Tensor):
    B, q_dim = query.shape
    assert key.shape == (B, 2, q_dim)

    k1, k2 = key.split([1, 1], 1)
    k1 = k1.view(query.shape)
    k2 = k2.view(query.shape)
    _support = op_or(op_and(k1, query), op_and(k2, -query))
    _support = -(-_support).logsumexp(dim=-1)

    return _support.view(B)


class Tree:

    def __init__(self, depth: int, key: Tensor, value: Tensor):
        self.depth = depth
        self.key = key
        self.value = value


def op_tree_attention(query: Tensor, tree: Tree):
    depth = tree.depth
    key = tree.key
    value = tree.value

    B, q_dim = query.shape
    _, _, v_dim = value.shape
    _2depth = int(2**depth)
    assert key.shape == (B, _2depth - 1, 2, q_dim)
    assert value.shape == (B, _2depth, v_dim)
    assert depth > 0

    device = query.device
    ix = torch.zeros((B,), device=device, dtype=torch.long)
    support: Tensor = None
    for d in range(depth):
        _ix: Tensor = int(2**d) - 1 + ix
        _ix = _ix.view(B, 1, 1).expand(B, 1, 2*q_dim)
        _key = key.view(B, _2depth - 1, 2*q_dim).gather(1, _ix)
        _key = _key.view(B, 2, q_dim)
        _support = op_match(query, _key)
        _ix = torch.bernoulli(_support.sigmoid().detach()).view(B)
        ix = 2 * ix + _ix.long()
        _support = (2 * _ix - 1) * _support
        if support is None:
            support = _support
        else:
            support = op_and(support, _support)

    _ix = ix.view(B, 1, 1).expand(B, 1, v_dim)
    value = value.gather(1, _ix).view(B, v_dim)

    return support, value


