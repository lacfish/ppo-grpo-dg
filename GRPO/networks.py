import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

    def get_action_logprob_and_entropy(self, state_x, action=None):
        logits = self.actor(state_x)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return (action, distribution.log_prob(action), distribution.entropy())
