# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List


class BaseModule(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        
    def build(self):
        return self
    
    def params_pairs_register(self, w: nn.Parameter, params_bound = None):
        assert w.shape[0] == 2
        w.requires_params_pairs_norm = True
        w.params_bound = params_bound
    
    @classmethod
    def params_pairs_norm(cls, w: nn.Parameter):
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
    
    @classmethod
    def log_softmax(cls, w: nn.Parameter):
        w_0, w_1 = w[0], w[1]
        w_ = torch.logaddexp(w_0, w_1)
        w_0 = w_0 - w_
        w_1 = w_1 - w_
        return w_0, w_1


class Coder(BaseModule):

    BOUND = 50.0

    def __init__(self):
        BaseModule.__init__(self)
        self.n_bits = 8

    def build(self):
        mul = torch.arange(self.n_bits)
        mul = self.n_bits - 1 - mul
        mul = (2 ** mul).long()
        self.mul = nn.Parameter(mul.view(1, -1), requires_grad=False)

        self.w = nn.Parameter(torch.tensor([Coder.BOUND, -Coder.BOUND]), requires_grad=True)
        self.params_pairs_register(self.w, Coder.BOUND)

        return self
    
    def bool2long(self, x: Tensor):
        shape = x.shape
        assert x.dtype == torch.bool
        x = x.view(-1, self.n_bits)
        x = (x * self.mul).sum(dim=-1)
        x = x.view(shape[:-1])
        return x

    def long2bool(self, x: Tensor):
        shape = x.shape
        assert x.dtype == torch.long
        x = x.view(-1, 1)
        x = (x // self.mul) % 2
        x = x.view(shape + (self.n_bits, ))
        return x.bool()

    def bool2float(self, x: Tensor):
        assert x.dtype == torch.bool
        w_0, w_1 = self.log_softmax(self.w)
        x_0 = torch.where(x, w_0, w_1)
        x_1 = torch.where(~x, w_0, w_1)
        return x_0, x_1
    
    def float2bool(self, x_0: Tensor, x_1: Tensor):
        x = torch.bernoulli(x_0.exp())
        return x.bool()
    
    @classmethod
    def BCELoss(cls, y_0: Tensor, y_1: Tensor, target: Tensor):
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
        self.w = nn.Parameter(torch.randn((2, self.y_dim, )), requires_grad=True)
        self.params_pairs_register(self.w)
        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        Y = self.y_dim
        assert X == Y

        w_0, w_1 = self.log_softmax(self.w)
        w_0 = w_0.view(1, Y)
        w_1 = w_1.view(1, Y)
        y_0 = torch.logaddexp(w_0 + x_0, w_1 + x_1)
        y_1 = torch.logaddexp(w_0 + x_1, w_0 + x_1)

        return y_0, y_1

"""
将太长的注意力缩短
"""
class ShorterLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
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
将地址转为注意力
"""
class AddressLayer(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.n_bits = 3
    
    def build(self):
        coder = Coder()
        coder.n_bits = self.n_bits
        coder.build()

        b = coder.long2bool(torch.arange(int(2**self.n_bits))).long()
        self.b_0 = nn.Parameter(b.float(), requires_grad=False)
        self.b_1 = nn.Parameter((~b).float(), requires_grad=False)

        return self
    
    def forward(self, a_0: Tensor, a_1: Tensor):
        attn = torch.einsum("bd,ad->ba", a_0, self.b_0)
        attn += torch.einsum("bd,ad->ba", a_1, self.b_1)
        return attn

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
        n_bits = np.ceil(np.log(self.x_dim) / np.log(2))
        n_bits = int(n_bits)
        self.n_bits = n_bits

        self.w = nn.Parameter(
            torch.randn((2, self.y_dim, n_bits)),
            requires_grad=True
        )
        self.params_pairs_register(self.w)
        
        coder = Coder()
        coder.n_bits = self.n_bits
        coder.build()

        self.address = AddressLayer()
        self.address.n_bits = self.n_bits
        self.address.build()

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

        w_0, w_1 = self.log_softmax(self.w) # Y, bit
        y = self.address.forward(w_0, w_1)
        y = self.shorter.forward(y)
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
        self.coder = Coder()
    
    def build(self):
        self.w = nn.Parameter(
            torch.randn((2, 1, self.y_dim, self.x_dim)),
            requires_grad=True
        )
        self.params_pairs_register(self.w)
        self.coder.build()
        return self
    
    def forward(self, x_0: Tensor, x_1: Tensor):
        B, Y, X = x_0.shape
        assert X == self.x_dim
        assert Y == self.y_dim
        coder = self.coder

        w1, w2 = coder.log_softmax(coder.w)
        w1: Tensor = w1
        w2: Tensor  = w2 - np.log(2)
        s1: Tensor = w2.view(1, 1).expand(B, Y)
        s2: Tensor = w1.view(1, 1).expand(B, Y)
        s3: Tensor = w2.view(1, 1).expand(B, Y)

        w_0, w_1 = coder.log_softmax(self.w)

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
        self.w = nn.Parameter(
            torch.randn((2, 1, self.x_dim, self.y_dim)),
            requires_grad=True
        )
        self.params_pairs_register(self.w)
        return self
    
    def forward(self, x: Tensor):
        B, X = x.shape
        assert X == self.x_dim
        Y = self.y_dim

        v_0, v_1 = self.log_softmax(self.w)
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
    
    def forward(self, x_0: Tensor, x_1: Tensor):
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
            attn1, attn2 = compare.forward(z_0, z_1)
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
        self.n_bits = 4
    
    def build(self):
        self.w = nn.Parameter(
            torch.randn((2, self.n_channel, self.n_bits)),
            requires_grad=True
        )
        self.params_pairs_register(self.w)

        self.tree.build()

        return self
    
    def get_query(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        assert x_0.shape == x_1.shape
        C = self.n_channel
        D = self.n_bits

        w_0, w_1 = self.log_softmax(self.w)
        x_0 = x_0.view(B, 1, X).expand(B, C, X).reshape(B, C, X)
        x_1 = x_1.view(B, 1, X).expand(B, C, X).reshape(B, C, X)
        w_0 = w_0.view(1, C, D).expand(B, C, D).reshape(B, C, D)
        w_1 = w_1.view(1, C, D).expand(B, C, D).reshape(B, C, D)
        q_0 = torch.cat([w_0, x_0], dim=-1)
        q_1 = torch.cat([w_1, x_1], dim=-1)

        return q_0, q_1

    def forward(self, x_0: Tensor, x_1: Tensor):
        B, X = x_0.shape
        assert x_0.shape == x_1.shape

        q_0, q_1 = self.get_query(x_0, x_1)
        v_0, v_1 = self.tree.forward(q_0, q_1)
        v_0, v_1 = v_0.view(B, -1), v_1.view(B, -1)

        return v_0, v_1


"""
模拟工作记忆
"""
class WorkingMemory(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.shorter = ShorterLayer()
        self.n_char_bits = 4
        self.n_address_bits = 4
    
    def build(self):
        
        self.adress = AddressLayer()
        self.adress.n_bits = self.n_address_bits
        self.adress.build()

        return self
    
    def read(self, m_0: Tensor, m_1: Tensor, a_0: Tensor, a_1: Tensor):
        A = self.shorter.y_dim
        C = self.n_char_bits
        m_0 = m_0.view(-1, A, C)
        m_1 = m_1.view(-1, A, C)
        B, A, C = m_0.shape
        assert self.shorter.y_dim == A
        assert self.n_char_bits == C

        B, D = a_0.shape
        assert a_0.shape == a_1.shape
        assert D == self.n_address_bits

        attn = self.adress.forward(a_0, a_1)
        attn = self.shorter.forward(attn)

        m_0 = attn.view(B, A, 1) + m_0
        m_0 = m_0.logsumexp(dim=1)
        m_1 = attn.view(B, A, 1) + m_1
        m_1 = m_1.logsumexp(dim=1)

        return m_0, m_1

    def write(self, 
            m_0: Tensor, m_1: Tensor, 
            a_0: Tensor, a_1: Tensor, 
            v_0: Tensor, v_1: Tensor):
        A = self.shorter.y_dim
        C = self.n_char_bits
        m_0 = m_0.view(-1, A, C)
        m_1 = m_1.view(-1, A, C)
        B, A, C = m_0.shape
        assert self.shorter.y_dim == A
        assert self.n_char_bits == C

        B, D = a_0.shape
        assert a_0.shape == a_1.shape
        assert D == self.n_address_bits

        attn = self.adress.forward(a_0, a_1)
        attn = self.shorter.forward(attn)

        nattn1 = attn[:, :-1].logcumsumexp(1)
        nattn1 = F.pad(nattn1, (1, 0), "constant", -torch.inf)
        nattn2 = attn[:, 1:].flip([1]).logcumsumexp(1).flip([1])
        nattn2 = F.pad(nattn2, (0, 1), "constant", -torch.inf)

        nattn = torch.logaddexp(nattn1, nattn2)

        m_0 = torch.logaddexp(
            m_0 + nattn.view(B, A, 1), 
            v_0.view(B, 1, C) + attn.view(B, A, 1))
        m_1 = torch.logaddexp(
            m_1 + nattn.view(B, A, 1), 
            v_1.view(B, 1, C) + attn.view(B, A, 1))
        
        return m_0, m_1

"""
将词转为向量
"""
class EmbeddingLayer(BaseModule):

    def __init__(self):
        BaseModule.__init__(self)
        self.n_token = 128
        self.n_bits = 4
    
    def build(self):
        self.w = nn.Parameter(torch.randn((2, self.n_token, self.n_bits)), requires_grad=True)
        self.params_pairs_register(self.w)
        return self
    
    def forward(self, x: Tensor):
        assert x.dtype == torch.long
        w_0, w_1 = self.log_softmax(self.w)
        x_0 = w_0.index_select(0, x.view(-1))
        x_1 = w_1.index_select(0, x.view(-1))
        shape = x.shape + (self.n_bits, )
        return x_0.view(shape), x_1.view(shape)

"""
图灵机模型
"""
class TuringMachine(BaseModule):
    
    def __init__(self):
        BaseModule.__init__(self)
        self.tree = ChannelTree()
        self.rw_memory = WorkingMemory()
        self.r_memory = WorkingMemory()
        self.n_r_head0 = 4 # rw_memory
        self.n_r_head1 = 4 # r_memory
        self.n_w_head = 4
        self.state_dim = 4
        self.n_loop = 4
        self.out_dim = 4

    def build(self):
        tree_in = self.state_dim + self.n_r_head0 * self.rw_memory.n_char_bits + \
            self.n_r_head0 * self.rw_memory.n_char_bits + \
            self.n_r_head1 * self.r_memory.n_char_bits + \
            self.tree.n_bits
        self.tree.tree.x_dim = tree_in
        tree_out = self.tree.tree.y_dim * self.tree.n_bits

        self.tree.build()
        self.rw_memory.build()
        self.r_memory.build()

        self.state_head = ShuffleLayer()
        self.state_head.x_dim = tree_out
        self.state_head.y_dim = self.state_dim
        self.state_head.build()

        r_head0 = []
        for i in range(self.n_r_head0):
            head = ShuffleLayer()
            head.x_dim = tree_out
            head.y_dim = self.rw_memory.n_address_bits
            r_head0.append(head.build())
        self.r_head0 = nn.ModuleList(r_head0)

        r_head1 = []
        for i in range(self.n_r_head1):
            head = ShuffleLayer()
            head.x_dim = tree_out
            head.y_dim = self.r_memory.n_address_bits
            r_head1.append(head.build())
        self.r_head1 = nn.ModuleList(r_head1)

        w_head = []
        for i in range(self.n_w_head):
            head = ShuffleLayer()
            head.x_dim = tree_out
            head.y_dim = self.rw_memory.n_address_bits
            w_head.append(head.build())
        self.w_head = nn.ModuleList(w_head)

        v_head = []
        for i in range(self.n_w_head):
            head = ShuffleLayer()
            head.x_dim = tree_out
            head.y_dim = self.rw_memory.n_char_bits
            v_head.append(head.build())
        self.v_head = nn.ModuleList(v_head)

        self.rw_memory0 = nn.Parameter(
            torch.randn(2, 1, self.rw_memory.shorter.y_dim, self.rw_memory.n_char_bits),
            requires_grad=True
        )
        self.params_pairs_register(self.rw_memory0)

        self.state0 = nn.Parameter(
            torch.randn(2, self.state_dim),
            requires_grad=True
        )
        self.params_pairs_register(self.state0)

        self.r_ptr0 = nn.Parameter(
            torch.randn(2, self.n_r_head0, self.rw_memory.n_address_bits),
            requires_grad=True
        )
        self.params_pairs_register(self.r_ptr0)

        self.r_ptr1 = nn.Parameter(
            torch.randn(2, self.n_r_head1, self.r_memory.n_address_bits),
            requires_grad=True
        )
        self.params_pairs_register(self.r_ptr1)

        self.out_head = ShuffleLayer()
        self.out_head.x_dim = self.rw_memory.shorter.y_dim * self.rw_memory.n_char_bits
        self.out_head.y_dim = self.out_dim
        self.out_head.build()

    def forward(self, rm_0: Tensor, rm_1: Tensor):
        B, L, A, C = rm_0.shape
        assert rm_0.shape == rm_1.shape
        assert self.r_memory.shorter.y_dim == A
        assert self.r_memory.n_char_bits == C

        rwm_0, rwm_1 = self.log_softmax(self.rw_memory0)
        rwm_0 = rwm_0.expand(B, -1, -1).reshape(B, -1, -1)
        rwm_1 = rwm_1.expand(B, -1, -1).reshape(B, -1, -1)

        r_ptr0_0, r_ptr0_1 = self.log_softmax(self.r_ptr0)
        r_ptr0_0 = [r_ptr0_0[i].view(1, -1).expand(B, -1) for i in range(self.n_r_head0)]
        r_ptr0_1 = [r_ptr0_1[i].view(1, -1).expand(B, -1) for i in range(self.n_r_head1)]

        r_ptr1_0, r_ptr1_1 = self.log_softmax(self.r_ptr1)
        r_ptr1_0 = [r_ptr1_0[i].view(1, -1).expand(B, -1) for i in range(self.n_r_head0)]
        r_ptr1_1 = [r_ptr1_1[i].view(1, -1).expand(B, -1) for i in range(self.n_r_head1)]

        state_0, state_1 = self.log_softmax(self.state0)
        state_0 = state_0.view(1, -1).expand(B, -1)
        state_1 = state_1.view(1, -1).expand(B, -1)

        out_0 = [rwm_0]
        out_1 = [rwm_1]

        for i in range(L):
            rm_0_ = rm_0[:, i, :, :]
            rm_1_ = rm_1[:, i, :, :]
            for j in range(self.n_loop):
                q_0 = [state_0]
                q_1 = [state_1]

                for k in range(self.n_r_head0):
                    v_0, v_1 = self.rw_memory.read(rm_0_, rm_1_, r_ptr0_0[k], r_ptr0_1[k])
                    q_0.append(v_0)
                    q_1.append(v_1)

                for k in range(self.n_r_head1):
                    v_0, v_1 = self.r_memory.read(rm_0_, rm_1_, r_ptr1_0[k], r_ptr1_1[k])
                    q_0.append(v_0)
                    q_1.append(v_1)

                q_0 = torch.cat(q_0, dim=-1)
                q_1 = torch.cat(q_1, dim=-1)
                v_0, v_1 = self.tree.forward(q_0, q_1)

                for k in range(self.n_w_head):
                    w_head: ShuffleLayer = self.w_head[k]
                    a_0, a_1 = w_head.forward(v_0, v_1)
                    v_head: ShuffleLayer = self.v_head[k]
                    wv_0, wv_1 = v_head.forward(v_0, v_1)
                    rwm_0, rwm_1 = self.rw_memory.write(rwm_0, rwm_1, a_0, a_1, wv_0, wv_1)

                for k in range(self.n_r_head0):
                    head: ShuffleLayer = self.r_head0[k]
                    r_ptr0_0[k], r_ptr0_1[k] = head.forward(v_0, v_1)

                for k in range(self.n_r_head1):
                    head: ShuffleLayer = self.r_head1[k]
                    r_ptr1_0[k], r_ptr1_1[k] = head.forward(v_0, v_1)

                state_0, state_1 = self.state_head.forward(v_0, v_1)

            _d = self.out_head.x_dim
            _out_0, _out_1 = self.out_head.forward(rwm_0.view(-1, _d), rwm_1.view(-1, _d))
            out_0.append(_out_0)
            out_1.append(_out_1)

        return out_0, out_1