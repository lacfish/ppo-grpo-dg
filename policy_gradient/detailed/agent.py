import numpy as np
from buffer import RollOutBuffer
from policy_net import Policy_net


class REINFORCEagent:
    def __init__(self, state_dim, hidden_dim, action_dim, lr=0.01):
        self.policy = Policy_net(state_dim, hidden_dim, action_dim)
        self.buffer = RollOutBuffer()
        self.lr = lr

    def select_action(self, state):
        probs, cache = self.policy.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, cache

    def update(self):
        adv = self.buffer.compute_advantages()
        dW1, db1, dW2, db2 = self.policy.manual_backward(
            self.buffer.caches, self.buffer.actions, adv
        )
        self.policy.W1 -= dW1 * self.lr
        self.policy.b1 -= db1 * self.lr
        self.policy.W2 -= dW2 * self.lr
        self.policy.b2 -= db2 * self.lr

        self.buffer.reset()
