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
    assert key.shape == (B, int(2**depth) - 1, 2, q_dim)
    assert value.shape == (B, int(2**depth), v_dim)
    assert depth > 0

    device = query.device
    ix = torch.zeros((B,), device=device, dtype=torch.long)
    support: Tensor = None
    for d in range(depth):
        _ix: Tensor = int(2**d) - 1 + ix
        _ix = _ix.view(B, 1, 1).expand(B, 1, 2*q_dim)
        _key = key.view(B, int(2**depth), 2*q_dim).gather(1, _ix)
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


class FeedForwardTree(nn.Module):

    def __init__(self, depth, in_dim, out_dim):
        nn.Module.__init__(self)
        self.depth = depth
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.key = nn.Parameter(torch.randn((out_dim, int(2**depth) - 1, 2, in_dim)), requires_grad=True)
        self.value = nn.Parameter(torch.randn((out_dim, int(2**depth), 1)), requires_grad=True)
    
    def forward(self, query: Tensor):
        B, q_dim = query.shape
        assert q_dim == self.in_dim

        query = query.view(B, 1, self.in_dim).expand(B, self.out_dim, self.in_dim)
        query = query.reshape(-1, self.in_dim)

        key = self.key.view(1, -1).expand(B, -1)
        key = key.reshape(-1, int(2**self.depth) - 1, 2, self.in_dim)

        value = self.value.view(1, -1).expand(B, -1)
        value = value.reshape(-1, int(2**self.depth), 1)

        support, value = op_tree_attention(query, Tree(self.depth, key, value))
        support = -(-support.view(B, -1).logsumexp(dim=-1))
        value = value.view(B, self.out_dim)

        return support, value

 
def op_tree_merge(tree1: Tree, tree2: Tree):
    query = tree1.value
    B, n, q_dim = query.shape
    assert q_dim == tree2.key.shape[2]

    query = query.view(-1, q_dim)
    key = tree2.key.view(B, 1, -1).expand(B, n, -1).reshape(-1, int(2**tree2.depth) - 1, 2, q_dim)
    value = tree2.value.view(B, 1, -1).expand(B, n, -1).reshape(-1, int(2**tree2.depth), q_dim)

    support, value = op_tree_attention(query, Tree(tree2.depth, key, value))
    support = -(-support.view(B, -1).logsumexp(dim=-1))
    value = value.view(B, n, -1)
    tree3 = Tree(tree1.depth, tree1.key, value)

    return support, tree3


def loss(support: Tensor, value: Tensor, target: Tensor):
    B, = support.shape
    _, D = value.shape
    assert value.shape == (B, D)
    assert target.shape == (B, D)

    _support = - (- (2 * target - 1) * value).logsumexp(dim=-1)
    _support = op_and(_support, support)
    _support = op_or(_support, - support)

    return - nn.functional.logsigmoid(_support).mean()