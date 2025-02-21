# -*- coding: utf-8 -*-

import torch
from torch import nn

import tree

Tensor = torch.Tensor


class Critic(nn.Module):

    def __init__(self, depth, state_dim, act_dim, hidden, q_dim, n_heads):
        nn.Module.__init__(self)
        self.depth = depth
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.q_dim = q_dim
        self.n_heads = n_heads

        self.hidden_tree = tree.TreeLayer(depth, state_dim + act_dim, hidden, n_heads)
        self.q_tree = tree.TreeLayer(depth, hidden, q_dim, 1)
    
    def forward(self, state: Tensor, act: Tensor):
        query = torch.cat([state, act], dim=1)
        hidden_support, hidden = self.hidden_tree.forward(query)
        hidden = hidden.logsumexp(dim=1)
        q_support, q = self.q_tree.forward(hidden)
        support = tree.and_(hidden_support, q_support)

        return support, q
    

class Actor(nn.Module):

    def __init__(self, depth, state_dim, act_dim, hidden, n_heads1, n_heads2):
        nn.Module.__init__(self)
        self.depth = depth
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.n_heads1 = n_heads1
        self.n_heads2 = n_heads2

        self.hidden_tree = tree.TreeLayer(depth, state_dim, hidden, n_heads1)
        self.act_tree = tree.TreeLayer(depth, hidden, act_dim, n_heads2)
    
    def forward(self, state: Tensor):
        hidden_support, hidden = self.hidden_tree.forward(state)
        hidden = hidden.logsumexp(dim=1)
        act_support, act = self.act_tree.forward(hidden)
        support = tree.and_(hidden_support, act_support)

        return support, act


class Trainer:

    def __init__(self):
        self.env = None
        self.state_dim = 1
        self.act_dim = 1
    