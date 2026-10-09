import torch


class RollOutBuffer:
    def __init__(self, num_steps, num_envs, device, state_dim):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.masks = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    def insert(self, states, actions, logprobs, rewards, masks):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.masks[self.step] = masks
        self.step = self.step + 1

    # 为了保证能够在get_flatten_batches中使用self.step做切片
    def reset(self):
        self.step = 0

    def compute_returns_and_advantages(self, eps=1e-8):
        rewards = self.rewards[: self.step]
        masks = self.masks[: self.step]

        returns = (rewards * masks).sum(dim=0)
        group_advantages = (returns - returns.mean()) / (
            returns.std(unbiased=False) + eps
        )
        self.advantages = group_advantages.unsqueeze(0).expand_as(masks)

        lengths = masks.sum(dim=0)
        group_weights = masks.sum() / (self.num_envs * lengths + eps)
        self.weights = group_weights.unsqueeze(0).expand_as(masks)

    def get_flatten_batches(self):
        masks = self.masks[: self.step]
        valid = masks.reshape(-1).bool()

        return (
            self.states[: self.step].reshape(-1, self.states.shape[-1])[valid],
            self.actions[: self.step].reshape(-1)[valid],
            self.logprobs[: self.step].reshape(-1)[valid],
            self.advantages.reshape(-1)[valid],
            self.weights.reshape(-1)[valid],
        )
