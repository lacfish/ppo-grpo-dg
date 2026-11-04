import numpy as np


class RollOutBuffer:
    def __init__(self):
        self.rewards = []
        self.actions = []
        self.caches = []

    def insert(self, reward, action, cache):
        self.rewards.append(reward)
        self.actions.append(action)
        self.caches.append(cache)

    def reset(self):
        self.rewards.clear()
        self.actions.clear()
        self.caches.clear()

    def compute_advantages(self, gamma=0.99):
        returns = []
        discounted_return = 0
        for r in reversed(self.rewards):
            discounted_return = discounted_return * gamma + r
            returns.insert(0, discounted_return)
        returns = np.array(returns, dtype=np.float32)

        baseline = np.mean(returns)
        advantages = (returns - baseline) / (np.std(returns) + 1e-8)
        return advantages
