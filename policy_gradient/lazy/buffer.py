import torch
from collections import deque


class RollOutBuffer:
    def __init__(self, device):
        self.device = device

        self.rewards = []
        self.logprobs = []

    def insert(self, reward, logprob):
        self.rewards.append(reward)
        self.logprobs.append(logprob)

    def compute_logprobs_and_advantages(self, gamma=0.99, eps=1e-8):
        discounted_return = 0
        returns = deque()
        for r in self.rewards[::-1]:
            discounted_return = discounted_return * gamma + r
            returns.appendleft(discounted_return)

        returns = torch.tensor(returns, device=self.device)
        baseline = returns.mean()
        advantages = (returns - baseline) / (returns.std() + eps)
        return self.logprobs, advantages

    def reset(self):
        del self.rewards[:]
        del self.logprobs[:]
