import torch
import torch.optim as optim
from torch.distributions import Categorical
from policy_net import policyNet
from buffer import RollOutBuffer


class REINFORCEagent:
    def __init__(self, policy_net, device, lr=1e-2):
        self.device = device
        self.policy = policyNet().to(device)
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.RollOutBuffer = RollOutBuffer(device)

    def select_action_and_return_logprob(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def update(self, gamma=0.99):
        log_probs, advantages = self.RollOutBuffer.compute_logprobs_and_advantages()
        policy_loss = []
        for logprob, advantage in zip(log_probs, advantages):
            policy_loss.append(-logprob * advantage)

        loss = torch.cat(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.reset()
