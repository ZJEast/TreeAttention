# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from typing import List


class BaseModule(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.params_bound = None
        self.params_pairs: List[nn.Parameter] = []
        
    def build(self):
        return self
    
    def params_pairs_register(self, w_0: nn.Parameter, w_1: nn.Parameter):
        assert w_0.shape == w_1.shape
        self.params_pairs.append([w_0, w_1])
    
    def params_pairs_norm(self):
        with torch.no_grad():
            for w_0, w_1 in self.params_pairs:
                w_0_ = w_0 - (w_0 + w_1) / 2
                w_1_ = w_1 - (w_0 + w_1) / 2
                if self.params_bound is not None:
                    w_0_ = w_0_.clamp_min_(self.params_bound)
                    w_1_ = w_1_.clamp_max_(-self.params_bound)
                w_0.data = w_0_.detach()
                w_1.data = w_1_.detach()
    
    def params_pairs_forward(self, w_0: nn.Parameter, w_1: nn.Parameter):
        w_0 = w_0 - torch.logaddexp(w_0, w_1)
        w_1 = w_1 - torch.logaddexp(w_0, w_1)
        return w_0, w_1


class Coder(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.n_bit = 8
        self.params_bound = 50.0

    def build(self):
        mul = torch.arange(self.n_bit)
        mul = self.n_bit - 1 - mul
        mul = (2 ** mul).long()
        self.mul = nn.Parameter(mul.view(1, -1), requires_grad=False)

        self.w_0 = nn.Parameter(torch.tensor(self.params_bound), requires_grad=True)
        self.w_1 = nn.Parameter(torch.tensor(-self.params_bound), requires_grad=True)
        self.params_pairs_register(self.w_0, self.w_1)

        return self
    
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

    def bool2float(self, x: Tensor):
        assert x.dtype == torch.bool
        w_0, w_1 = self.params_pairs_forward(self.w_0, self.w_1)
        x_0 = torch.where(x, w_0, w_1)
        x_1 = torch.where(~x, w_0, w_1)
        return x_0, x_1
    
    def float2bool(self, x_0: Tensor, x_1: Tensor):
        x = torch.bernoulli(x_0.exp())
        return x.bool()
    
    def BCELoss(self, y_0: Tensor, y_1: Tensor, target: Tensor):
        loss = torch.where(target, y_0, y_1)
        loss = - loss.mean()
        return loss

"""
学习是否对特征取非
"""
class DenyLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.y_dim = 1
    
    def build(self):
        self.w_0 = nn.Parameter(torch.randn((self.y_dim, )), requires_grad=True)
        self.w_1 = nn.Parameter(torch.randn((self.y_dim, )), requires_grad=True)
        self.params_pairs_register(self.w_0, self.w_1)
        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        Y = self.y_dim
        assert X == Y

        w_0, w_1 = self.params_pairs_forward(self.w_0, self.w_1)
        w_0 = w_0.view(1, Y)
        w_1 = w_1.view(1, Y)
        y_0 = torch.logaddexp(w_0 + x_0, w_1 + x_1)
        y_1 = torch.logaddexp(w_0 + x_1, w_0 + x_1)

        return y_0, y_1

"""
将太长的注意力缩短
"""
class ShorterLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.y_dim = 3
    
    def build(self):
        return self
    
    def forward(self, x: Tensor):
        B, X = x.shape
        Y = self.y_dim
        assert X >= Y
        if X == Y:
            return x
        y1 = x[:, :Y]
        y2 = x[:, Y:]
        y2 = y2.logsumexp(dim=-1, keepdim=True) - np.log(X - Y)
        y = torch.logaddexp(y1, y2)
        return y

"""
对特征进行重排序
"""
class ShuffleLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.x_dim = 4
        self.y_dim = 1
        self.use_deny = True
    
    def build(self):
        n_bit = np.ceil(np.log(self.x_dim) / np.log(2))
        n_bit = int(n_bit)
        self.n_bit = n_bit
        X = int(2**self.n_bit)

        self.w_0 = nn.Parameter(
            torch.randn((self.y_dim, n_bit)),
            requires_grad=True
        )
        self.w_1 = nn.Parameter(
            torch.randn((self.y_dim, n_bit)),
            requires_grad=True
        )
        self.params_pairs_register(self.w_0, self.w_1)
        
        coder = Coder()
        coder.n_bit = self.n_bit
        coder.build()

        b = coder.long2bool(torch.arange(X)).long()
        self.b_0 = nn.Parameter(b.float(), requires_grad=False)
        self.b_1 = nn.Parameter((~b).float(), requires_grad=False)

        self.shorter = ShorterLayer()
        self.shorter.y_dim = self.x_dim
        self.shorter.build()

        if self.use_deny:
            self.deny = DenyLayer()
            self.deny.y_dim = self.y_dim
            self.deny.build()

        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        assert x_0.shape == x_1.shape
        Y = self.y_dim

        w_0, w_1 = self.params_pairs_forward(self.w_0, self.w_1) # Y, bit
        y_0 = torch.einsum("yb,xb->yx", w_0, self.b_0)
        y_1 = torch.einsum("yb,xb->yx", w_1, self.b_1)
        y = self.shorter.forward(y_0 + y_1)
        y_0 = y.view(1, Y, X) + x_0.view(B, 1, X)
        y_0 = y_0.logsumexp(dim=2)
        y_1 = y.view(1, Y, X) + x_1.view(B, 1, X)
        y_1 = y_1.logsumexp(dim=2)

        if self.use_deny:
            y_0, y_1 = self.deny.forward(y_0, y_1)

        return y_0, y_1

"""
特征重排后，对比大小
"""
class CompareLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.x_dim = 8
        self.y_dim = 128
    
    def build(self):
        self.w_0 = nn.Parameter(
            torch.randn((1, self.y_dim, self.x_dim)),
            requires_grad=True
        )
        self.w_1 = nn.Parameter(
            torch.randn((1, self.y_dim, self.x_dim)),
            requires_grad=True
        )
        self.params_pairs_register(self.w_0, self.w_1)
        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor, coder: Coder):
        B, Y, X = x_0.shape
        assert X == self.x_dim
        assert Y == self.y_dim

        w1, w2 = coder.w_0, coder.w_1
        w1, w2 = coder.params_pairs_forward(w1, w2)
        w1: Tensor = w1
        w2: Tensor  = w2 - np.log(2)
        s1: Tensor = w2.view(1, 1).expand(B, Y)
        s2: Tensor = w1.view(1, 1).expand(B, Y)
        s3: Tensor = w2.view(1, 1).expand(B, Y)

        w_0, w_1 = coder.params_pairs_forward(self.w_0, self.w_1)

        for i in range(X):
            x10 = x_0[:, :, i]
            x11 = x_1[:, :, i]
            x20 = w_0[:, :, i]
            x21 = w_1[:, :, i]
            s1 = torch.logaddexp(s1, s2 + x10 + x21)
            s3 = torch.logaddexp(s3, s2 + x11 + x20)
            s2 = s2 + torch.logaddexp(x10 + x20, x11 + x21)
        
        s1 = torch.logaddexp(s1, s2).view(B, Y)
        s3 = s3.view(B, Y)

        return s1, s3

"""
根据注意力检索value
"""
class ValueLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.x_dim = 128
        self.y_dim = 4
    
    def build(self):
        self.w_0 = nn.Parameter(
            torch.randn((1, self.x_dim, self.y_dim)),
            requires_grad=True
        )
        self.w_1 = nn.Parameter(
            torch.randn((1, self.x_dim, self.y_dim)),
            requires_grad=True
        )
        self.params_pairs_register(self.w_0, self.w_1)
        return self
    
    def forward(self, x: Tensor):
        B, X = x.shape
        assert X == self.x_dim
        Y = self.y_dim

        v_0, v_1 = self.params_pairs_forward(self.w_0, self.w_1)
        y_0 = x.view(B, X, 1) + v_0.view(1, X, Y)
        y_0 = y_0.logsumexp(1).view(B, Y)
        y_1 = x.view(B, X, 1) + v_1.view(1, X, Y)
        y_1 = y_1.logsumexp(1).view(B, Y)

        return y_0, y_1


"""
二叉树神经网络
"""
class TreeLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 8
        self.y_dim = 4
        self.depth = 8
        self.shuffle_dim = 4

    def build(self):
        shuffle_layers = []
        compare_layers = []

        for i in range(self.depth):
            n = int(2**i)
            shuffle = ShuffleLayer()
            shuffle.x_dim = self.x_dim
            shuffle.y_dim = n * self.shuffle_dim
            shuffle_layers.append(shuffle.build())

            compare = CompareLayer()
            compare.x_dim = self.shuffle_dim
            compare.y_dim = n
            compare_layers.append(compare.build())
        
        self.shuffle_layers = nn.ModuleList(shuffle_layers)
        self.compare_layers = nn.ModuleList(compare_layers)

        value = ValueLayer()
        value.x_dim = int(2**self.depth)
        value.y_dim = self.y_dim
        self.value_layer = value.build()

        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor, coder: Coder):
        B, X = x_0.shape
        assert X == self.x_dim

        attn = torch.zeros((B, 1), device=x_0.device)
        
        for i in range(self.depth):
            n = int(2**i)
            shuffle: ShuffleLayer = self.shuffle_layers[i]
            compare: CompareLayer = self.compare_layers[i]
            z_0, z_1 = shuffle.forward(x_0, x_1)
            z_0 = z_0.view(B, n, self.shuffle_dim)
            z_1 = z_1.view(B, n, self.shuffle_dim)
            attn1, attn2 = compare.forward(z_0, z_1, coder)
            attn = torch.cat([attn + attn1, attn + attn2], dim=-1)
        
        y_0, y_1 = self.value_layer.forward(attn)

        return y_0, y_1


"""
多通道的二叉树
"""
class ChannelTree(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.tree = TreeLayer()
        self.n_channel = 4
        self.n_bit = 4
    
    def build(self):
        self.w_0 = nn.Parameter(
            torch.randn((self.n_channel, self.n_bit)),
            requires_grad=True
        )
        self.w_1 = nn.Parameter(
            torch.randn((self.n_channel, self.n_bit)),
            requires_grad=True
        )
        self.params_pairs_register(self.w_0, self.w_1)

        self.tree.build()

        return self
    
    def get_query(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        assert x_0.shape == x_1.shape
        C = self.n_channel
        D = self.n_bit

        w_0, w_1 = self.params_pairs_forward(self.w_0, self.w_1)
        x_0 = x_0.view(B, 1, X).expand(B, C, X).reshape(B, C, X)
        x_1 = x_1.view(B, 1, X).expand(B, C, X).reshape(B, C, X)
        w_0 = w_0.view(1, C, D).expand(B, C, D).reshape(B, C, D)
        w_1 = w_1.view(1, C, D).expand(B, C, D).reshape(B, C, D)
        q_0 = torch.cat([w_0, x_0], dim=-1)
        q_1 = torch.cat([w_1, x_1], dim=-1)

        return q_0, q_1

    def forward(self, x_0: Tensor, x_1: Tensor, coder: Coder):
        B, X = x_0.shape
        assert x_0.shape == x_1.shape

        q_0, q_1 = self.get_query(x_0, x_1)
        v_0, v_1 = self.tree.forward(q_0, q_1, coder)

        return v_0.view(B, -1), v_1.view(B, -1)


class WorkingMemory:

    @classmethod
    def init_memory(cls, m0: Tensor, coder: Coder):
        assert m0.dtype == torch.bool
        B, A, C = m0.shape
        m_0, m_1 = coder.bool2float(m0)
        return m_0, m_1