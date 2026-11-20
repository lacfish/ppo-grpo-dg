import torch


class RollOutBuffer:
    def __init__(self, num_steps, num_envs, device, state_dim, action_dim):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        # add:q(a|s)，inference engine对所有动作的概率分布
        self.q_probs = torch.zeros(
            (num_steps, num_envs, action_dim), device=device
        )
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    def insert(self, states, actions, logprobs, q_probs, rewards, dones, values):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.q_probs[self.step] = q_probs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.step = (self.step + 1) % self.num_steps

    def compute_returns_and_advantages(
        self, next_value, next_done, gamma=0.99, gae_lambda=0.95
    ):
        self.advantages = torch.zeros_like(self.rewards, device=self.device)
        gae_tmp = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnotterminal = 1 - next_done
                value_tplus1 = next_value
            else:
                nextnotterminal = 1 - self.dones[t + 1]
                value_tplus1 = self.values[t + 1]
            delta = (
                self.rewards[t]
                + nextnotterminal * gamma * value_tplus1
                - self.values[t]
            )
            self.advantages[t] = delta + nextnotterminal * gamma * gae_lambda * gae_tmp
            gae_tmp = delta + nextnotterminal * gamma * gae_lambda * gae_tmp
        self.returns = self.advantages + self.values

    def get_flatten_batches(self):
        return (
            self.states.reshape(-1, self.states.shape[-1]),
            self.actions.reshape(-1),
            self.logprobs.reshape(-1),
            self.q_probs.reshape(-1, self.q_probs.shape[-1]),
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
            self.values.reshape(-1),
        )
