# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List


def params_pairs_register(w: nn.Parameter, params_bound = None):
    assert w.shape[0] == 2
    w.requires_params_pairs_norm = True
    w.params_bound = params_bound


def params_pairs_norm(w: nn.Parameter):
    if not hasattr(w, "requires_params_pairs_norm"):
        return
    with torch.no_grad():
        w_0, w_1 = w[0], w[1]
        w_0_ = w_0 - (w_0 + w_1) / 2
        w_1_ = w_1 - (w_0 + w_1) / 2
        if w.params_bound is not None:
            w_0_ = w_0_.clamp_min_(w.params_bound)
            w_1_ = w_1_.clamp_max_(-w.params_bound)
        w_0.data = w_0_.detach()
        w_1.data = w_1_.detach()


def log_softmax(w: Tensor):
    w_0, w_1 = w[0], w[1]
    w_ = torch.logaddexp(w_0, w_1)
    w_0 = w_0 - w_
    w_1 = w_1 - w_
    return w_0, w_1


def bool2long(x: Tensor, n_bits: int, mat=None):
    if mat is None:
        mul = torch.arange(n_bits)
        mul = n_bits - 1 - mul
        mul = (2 ** mul).long()
        mat = mul.view(1, -1)

    shape = x.shape
    assert x.dtype == torch.bool

    x = x.view(-1, n_bits)
    x = (x * mat).sum(dim=-1)
    x = x.view(shape[:-1])
    return x


def long2bool(x: Tensor, n_bits: int, mat=None):
    if mat is None:
        mul = torch.arange(n_bits)
        mul = n_bits - 1 - mul
        mul = (2 ** mul).long()
        mat = mul.view(1, -1)
    
    shape = x.shape
    assert x.dtype == torch.long
    x = x.view(-1, 1)
    x = (x // mat) % 2
    x = x.view(shape + (mat, ))
    return x.bool()


def bool2float(x: Tensor, w: nn.Parameter):
    assert x.dtype == torch.bool
    w_0, w_1 = log_softmax(w)
    x_0 = torch.where(x, w_0, w_1)
    x_1 = torch.where(~x, w_0, w_1)
    return x_0, x_1


def float2bool(x_0: Tensor, x_1: Tensor):
    x = torch.bernoulli(x_0.exp())
    return x.bool()


def bce_loss(y_0: Tensor, y_1: Tensor, target: Tensor):
    loss = torch.where(target, y_0, y_1)
    loss = - loss.mean()
    return loss


def deny(x_0: Tensor, x_1: Tensor, w: Tensor):
    B, X = x_0.shape

    w_0, w_1 = log_softmax(w)
    w_0 = w_0.view(1, X)
    w_1 = w_1.view(1, X)
    y_0 = torch.logaddexp(w_0 + x_0, w_1 + x_1)
    y_1 = torch.logaddexp(w_0 + x_1, w_0 + x_1)

    return y_0, y_1


def shorter(x: Tensor, out: int):
    B, X = x.shape
    Y = out

    assert X >= Y
    if X == Y:
        return x
    
    y1 = x[:, :Y]
    y2 = x[:, Y:]
    y2 = y2.logsumexp(dim=-1, keepdim=True) - np.log(X - Y)
    y = torch.logaddexp(y1, y2)
    return y


def address(a_0: Tensor, a_1: Tensor, n_bits: int, b_0=None, b_1=None):
    if b_0 is None or b_1 is None:
        b = long2bool(torch.arange(int(2**n_bits)), n_bits).long()
        b_0 = b.float()
        b_0 = (~b).float()
    attn = torch.einsum("bd,ad->ba", a_0, b_0)
    attn += torch.einsum("bd,ad->ba", a_1, b_1)
    return attn