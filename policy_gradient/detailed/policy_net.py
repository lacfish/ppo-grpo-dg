import numpy as np


class Policy_net:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros(action_dim)

    def forward(self, state):
        z1 = (self.W1.T @ state[:, None]).squeeze() + self.b1
        h1 = np.maximum(z1, 0)
        z2 = (self.W2.T @ h1[:, None]).squeeze() + self.b2
        exp_z2 = np.exp(z2 - np.max(z2))
        probs = (exp_z2) / (np.sum(exp_z2))

        cache = (state, z1, h1, probs)
        return probs, cache

    def manual_backward(self, cache, actions, advantages):
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)

        for cache, action, adv in zip(cache, actions, advantages):
            state, z1, h1, probs = cache

            dz2 = probs.copy()
            dz2[action] -= 1
            dz2 = dz2 * adv

            db2 += dz2
            dW2 += h1[:, None] @ dz2[None, :]

            dh1 = (dz2[None, :] @ self.W2.T).squeeze()
            dz1 = dh1.copy()
            for i in range(z1.size):
                if z1[i] <= 0:
                    dz1[i] = 0

            db1 += dz1
            dW1 += state[:, None] @ dz1[None, :]
        return dW1, db1, dW2, db2
