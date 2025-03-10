# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.distributions import Categorical
import tree
from torch import optim

Tensor = torch.Tensor

class TreeModel(nn.Module):

    def __init__(self, depth, q_dim, v_dim):
        nn.Module.__init__(self)
        self.depth = depth
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.key = nn.Parameter(torch.randn((int(2**depth) - 1, 2, q_dim)), requires_grad=True)
        self.value = nn.Parameter(torch.randn((int(2**depth), v_dim)), requires_grad=True)
    
    def forward(self, query: Tensor):
        B, _ = query.shape
        assert query.shape == (B, self.q_dim)

        return tree.tree_attention(query, self.depth, self.key, self.value)


class Coder(nn.Module):

    def __init__(self, lower_bound=0.1):
        nn.Module.__init__(self)
        self.lower_bound = lower_bound
        self.abs = nn.Parameter(torch.tensor(lower_bound), requires_grad=True)
    
    def encode(self, x: Tensor):
        return (2 * x.float() - 1) * self.abs
    
    def decode(self, x: Tensor):
        x = torch.bernoulli(x.sigmoid())
        return x.bool()
    
    def check(self):
        data = self.abs.data
        self.abs.data = torch.maximum(data, torch.full_like(data, self.lower_bound))
        

class RandomTreeModel:

    def __init__(self, t: TreeModel):
        self.template = t
        self.weights = torch.tensor([1, 3, 3, 10]).log()
        self.inf = 1000.0
    
    def sample_tree(self):
        t1 = self.template
        t2 = TreeModel(t1.depth, t1.q_dim, t1.v_dim)
        _2depth = int(2**t1.depth)

        logits = self.weights.log().view(1, -1).expand((_2depth - 1) * t1.q_dim, -1)
        key = Categorical(logits=logits).sample().view((_2depth - 1), t1.q_dim)
        k1 = (key % 2).float()
        k2 = (key // 2).float()
        t2.key.data[0, :, 0, :] = (2 * k1 - 1) * self.inf
        t2.key.data[0, :, 1, :] = (2 * k2 - 1) * self.inf

        value = torch.full_like(t2.value, 0.5)
        value = torch.bernoulli(value) * 2 - 1
        t2.value.data = value * self.inf

        device = t1.key.device
        t2 = t2.to(device)
        coder = Coder(self.inf).to(device)
        return t2, coder
    
    def sample_query(self, batch_size):
        t1 = self.template
        B = batch_size
        query = torch.full((B, t1.q_dim), 0.5)
        query = torch.bernoulli(query).bool()

        query = query.to(t1.key.device)
        return query
    

def target_loss(support: Tensor, value: Tensor, target: Tensor):
    B, = support.shape
    _, D = value.shape
    assert value.shape == (B, D)
    assert target.shape == (B, D)

    _support = tree.equal(value, target)
    _support = tree.and_(_support, support)
    _support = tree.or_(_support, - support)

    return - nn.functional.logsigmoid(_support).mean()


def test_train(t1: TreeModel):
    device = torch.device("cuda")
    t1 = t1.to(device)
    c1 = Coder(0.1).to(device)

    random = RandomTreeModel(t1)
    t2, c2 = random.sample_tree()

    opt = optim.Adam(list(t1.parameters()) + list(c1.parameters()), lr=0.1)

    t2.eval()
    c2.eval()
    B = 512
    step = 0
    while True:
        query = random.sample_query(B)

        _, target = t2.forward(c2.encode(query))
        target = c2.decode(target)

        support, value = t1.forward(c1.encode(query))
        loss = target_loss(support, value, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        c1.check()
        
        print(f"{step}: {loss.item()}")
        step += 1

        if loss.item() < 1e-4:
            print(c1.abs.data)
            print(t1.key.data)
            print(t1.value.data)
            print(c2.abs.data)
            print(t2.key.data)
            print(t2.value.data)
            break


if __name__ == "__main__":
    test_train(TreeModel(5, 5, 1))