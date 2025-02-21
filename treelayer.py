# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from typing import List

"""
编码相关
"""
class Coder(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.n_bit = 8
        self.inf = 100.0
    
    def build(self):
        mul = torch.arange(self.n_bit)
        mul = self.n_bit - 1 - mul
        mul = (2 ** mul).long()
        self.mul = nn.Parameter(mul.view(1, -1), requires_grad=False)
        self.w = nn.Parameter(torch.tensor([0.0, -100.0]), requires_grad=True)
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
    
    def check(self):
        w1, w2 = self.w[0].item(), self.w[1].item()
        w1 = max(0.0, w1)
        w2 = min(-self.inf, w2)
        self.w.data = torch.tensor([w1, w2], device=self.w.device)

    def bool2float(self, x: Tensor):
        assert x.dtype == torch.bool
        w = self.w.log_softmax(-1)
        x1 = torch.where(x, w[0], w[1])
        x2 = torch.where(~x, w[0], w[1])
        x1 = x1.view(x.shape + (1, ))
        x2 = x2.view(x.shape + (1, ))
        x = torch.cat([x1, x2], dim=-1)
        return x
    
    def float2bool(self, x: Tensor):
        assert x.shape[-1] == 2
        shape = x.shape[:-1]
        x = Categorical(logits=x.view(-1, 2)).sample()
        return (~ x.bool()).view(shape)
    
    def BCELoss(self, y1: Tensor, y2: Tensor):
        y1 = y1.view(-1)
        y2 = y2.view(-1, 2)
        loss = torch.where(y1, y2[:, 0], y2[:, 1])
        loss = - loss.mean()
        return loss

"""
学习是否对特征取非
"""
class DenyLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.y_dim = 1
    
    def build(self):
        self.w = nn.Parameter(
            torch.randn((self.y_dim, 2)),
            requires_grad=True
        )

        return self
    
    def forward(self, x: Tensor):
        Y = self.y_dim
        y = self.w.log_softmax(-1)
        y = y.view(1, Y, 2)
        y1 = (x + y).logsumexp(dim=-1, keepdim=True)
        y2 = (x + y.flip([2])).logsumexp(dim=-1, keepdim=True)
        y = torch.cat([y1, y2], dim=-1)
        return y

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
class ShuffleLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 4
        self.y_dim = 1
        self.use_deny = True
    
    def build(self):
        n_bit = np.ceil(np.log(self.x_dim) / np.log(2))
        n_bit = int(n_bit)
        self.n_bit = n_bit
        X = int(2**self.n_bit)

        self.w = nn.Parameter(
            torch.randn((self.y_dim, n_bit, 2)),
            requires_grad=True
        )
        
        coder = Coder()
        coder.n_bit = self.n_bit
        coder.build()
        self.coder = coder

        b = coder.long2bool(torch.arange(X)).long()
        ix = torch.arange(n_bit).view(1, n_bit) * 2 + b.view(X, n_bit)
        self.ix = nn.Parameter(ix.view(-1), requires_grad=False)

        self.shorter = ShorterLayer()
        self.shorter.y_dim = self.x_dim
        self.shorter.build()

        if self.use_deny:
            self.deny = DenyLayer()
            self.deny.y_dim = self.y_dim
            self.deny.build()

        return self
    
    def forward(self, x: Tensor):
        B, X, two = x.shape
        assert two == 2
        Y = self.y_dim

        y = self.w.log_softmax(-1)
        y = y.view(Y, -1).index_select(1, self.ix)
        y = y.view(Y, int(2**self.n_bit), -1).sum(-1)
        y = self.shorter.forward(y)
        y = y.view(1, Y, X, 1) + x.view(B, 1, X, 2)
        y = y.logsumexp(dim=2).view(B, Y, 2)
        if self.use_deny:
            y = self.deny.forward(y)

        return y


class ShuffleTestData(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 8
        self.y_dim = 2
        self.deny = [False, True]
        self.ix = [7, 3]
        self.batch_size = 32
    
    def build(self):
        self.deny = nn.Parameter(
            torch.tensor(self.deny, dtype=torch.bool),
            requires_grad=False
        )
        self.ix = nn.Parameter(
            torch.tensor(self.ix, dtype=torch.long),
            requires_grad=False
        )

        return self

    def sample(self):
        B = self.batch_size
        X = self.x_dim
        Y = self.y_dim
        x = torch.bernoulli(torch.full((B, X), 0.5, device=self.deny.device)).bool()
        y = x.index_select(1, self.ix)
        d = self.deny.view(1, -1).expand(B, -1)
        y = torch.where(d, ~y, y)
        return x, y


class ShuffleTestTrainer:

    def __init__(self):
        self.data = ShuffleTestData()
        self.device = torch.device("cuda")
        self.lr = 1e-2

    def build(self):
        self.data = self.data.build().to(self.device)

        shuffle = ShuffleLayer()
        shuffle.x_dim = self.data.x_dim
        shuffle.y_dim = self.data.y_dim

        self.model = shuffle.build().to(self.device)
        self.coder = Coder().build().to(self.device)
        parameters = list(self.model.parameters()) + list(self.coder.parameters())

        self.opt = torch.optim.Adam(parameters, lr=self.lr)

        return self
    
    def train(self):
        while True:
            x, y1 = self.data.sample()
            x = self.coder.bool2float(x)
            y2: Tensor = self.model.forward(x)
            loss = self.coder.BCELoss(y1, y2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.coder.check()

            print(loss.item())
            if loss.isnan():
                break

"""
特征重排后，对比大小
"""
class CompareLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 8
        self.y_dim = 128
    
    def build(self):
        self.w = nn.Parameter(
            torch.randn((1, self.y_dim, self.x_dim, 2)),
            requires_grad=True
        )
        return self
    
    def forward(self, x: Tensor, coder: Coder):
        B, Y, X, two = x.shape
        assert two == 2
        assert X == self.x_dim
        assert Y == self.y_dim

        w = coder.w.log_softmax(-1)
        w1: Tensor = w[0]
        w2: Tensor  = w[1] - np.log(2)
        s1: Tensor = w2.view(1, 1).expand(B, Y)
        s2: Tensor = w1.view(1, 1).expand(B, Y)
        s3: Tensor = w2.view(1, 1).expand(B, Y)

        w = self.w.log_softmax(-1)

        for i in range(X):
            x10 = x[:, :, i, 0]
            x11 = x[:, :, i, 1]
            x20 = w[:, :, i, 0]
            x21 = w[:, :, i, 1]
            s1 = torch.logaddexp(s1, s2 + x10 + x21)
            s3 = torch.logaddexp(s3, s2 + x11 + x20)
            s2 = s2 + torch.logaddexp(x10 + x20, x11 + x21)
        
        s1 = torch.logaddexp(s1, s2).view(B, Y)
        s3 = s3.view(B, Y)

        return s1, s3


"""
根据注意力检索value
"""
class ValueLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 128
        self.y_dim = 4
    
    def build(self):
        self.w = nn.Parameter(
            torch.randn((1, self.x_dim, self.y_dim, 2)),
            requires_grad=True
        )
        return self
    
    def forward(self, x: Tensor):
        B, X = x.shape
        assert X == self.x_dim
        Y = self.y_dim

        v = self.w.log_softmax(-1)
        y = x.view(B, X, 1, 1) + v.view(1, X, Y, 2)
        y = y.logsumexp(1).view(B, Y, 2)

        return y


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
    
    def forward(self, x: Tensor, coder: Coder):
        B, X, two = x.shape
        assert two == 2
        assert X == self.x_dim

        attn = torch.zeros((B, 1), device=x.device)
        
        for i in range(self.depth):
            n = int(2**i)
            shuffle: ShuffleLayer = self.shuffle_layers[i]
            compare: CompareLayer = self.compare_layers[i]
            z = shuffle.forward(x).view(B, n, self.shuffle_dim, 2)
            attn1, attn2 = compare.forward(z, coder)
            attn = torch.cat([attn + attn1, attn + attn2], dim=-1)
        
        y = self.value_layer.forward(attn)

        return y


class TreeTestData(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.x_dim = 4
        self.batch_size = 32
    
    def build(self):
        y = torch.bernoulli(torch.full((int(2**self.x_dim), 1), 0.5)).bool()
        self.y = nn.Parameter(y, requires_grad=False)

        coder = Coder()
        coder.n_bit = self.x_dim
        self.coder = coder.build()

        return self
    
    def sample(self):
        device = self.y.device
        B = self.batch_size
        x = torch.randint(0, int(2**self.x_dim), size=(B,), device=device)
        y = self.y.index_select(0, x)
        x = self.coder.long2bool(x)
        return x, y


class TreeTestTrainer:

    def __init__(self):
        self.data = TreeTestData()
        self.device = torch.device("cuda")
        self.lr = 1e-2

    def build(self):
        self.data = self.data.build().to(self.device)

        tree = TreeLayer()
        tree.x_dim = self.data.x_dim
        tree.shuffle_dim = 1
        tree.y_dim = 1
        tree.depth = self.data.x_dim + 1
        self.model = tree.build().to(self.device)
        self.coder = Coder().build().to(self.device)
        parameters = list(self.model.parameters()) + list(self.coder.parameters())

        self.opt = torch.optim.Adam(parameters, lr=self.lr)

        return self
    
    def train(self):
        while True:
            x, y1 = self.data.sample()
            x = self.coder.bool2float(x)
            y2: Tensor = self.model.forward(x, self.coder)
            loss = self.coder.BCELoss(y1, y2)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.coder.check()

            print(loss.item())
            if loss.isnan():
                break


if __name__ == "__main__":
    # trainer = ShuffleTestTrainer().build()
    # trainer.train()
    trainer = TreeTestTrainer().build()
    trainer.train()