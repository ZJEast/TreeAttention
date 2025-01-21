# -*- coding: utf-8 -*-

from tkinter import NO
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

    k1 = key[:, 0, :].view(query.shape)
    k2 = key[:, 1, :].view(query.shape)
    _support = op_or(op_and(k1, query), op_and(k2, -query))
    _support = -(-_support).logsumexp(dim=-1)

    return _support.view(B)


def op_tree_attention(query: Tensor, depth: int, key: Tensor, value: Tensor, k_id: Tensor=None, v_id: Tensor=None):

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
        _support = op_match(query, _key)
        _ix = torch.bernoulli(_support.sigmoid().detach()).view(B)
        ix = 2 * ix + _ix.long()
        _support = (2 * _ix - 1) * _support
        if support is None:
            support = _support
        else:
            support = op_and(support, _support)

    _ix = ix
    value = value.index_select(0, v_id * v_num + _ix).view(B, v_dim)

    return support, value


def op_equal(x: Tensor, y: Tensor):
    B, D = x.shape
    assert x.shape == y.shape
    support = - (- (2 * y.float() - 1) * x).logsumexp(dim=-1)
    return support.view(B)