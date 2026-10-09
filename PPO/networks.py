import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# 函数：初始化layer，layer.weight是半正交矩阵，layer.bias初始为0
def init_layer(layer, std=np.sqrt(2), bias=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


# class ActorCritic,继承nn.Moudle,初始化Actor,Critic网络，函数get_value(self,x)返回当前局面打分，
# 函数get_action_and_value(self,x,action=None)返回动作、该动作的对数概率p1、动作概率分布的信息熵、当前局面打分
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            init_layer(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            init_layer(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            init_layer(nn.Linear(64, 1), std=1.0),
        )

    def get_critics(self, state_x):
        return self.critic(state_x)

    def get_action_and_value(self, state_x, action=None):
        logits = self.actor(state_x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.get_critics(state_x),
        )
