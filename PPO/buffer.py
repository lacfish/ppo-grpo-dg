import torch


# RollOutBuffer类，存储状态state,动作actions,当前状态下动作的对数概率logprobs，
# 环境给的即时奖励rewards,回合是否结束dones,critic对于当前状态的打分values
# 初始化num_steps,num_envs,device,六个存储值，以(num_steps,num_envs)的矩阵存储，位置为device，step作为当前步数的指针
class RollOutBuffer:
    def __init__(self, num_steps, num_envs, device, state_dim):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.states = torch.zeros((num_steps, num_envs, state_dim), device=device)
        self.actions = torch.zeros((num_steps, num_envs), device=device)
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    # insert函数，对于六个存储值赋值，step+1
    def insert(self, states, actions, logprobs, rewards, dones, values):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.step = (self.step + 1) % self.num_steps

    # compute_returns_and_advantages，初始化advantages数据结构，与rewards相同，从后往前便利并处理t==num_steps-1的边界情况
    # (定义nextnonterminal并外界传入next_value,next_done)，TD 误差 delta = r + gamma * V(s') - V(s)，
    # 优势 A_t = delta + lambda * gamma * A_{t+1},用advantage+values算出return,step归零
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

    # get_flatten_batches，将self.states
    def get_flatten_batches(self):
        return (
            self.states.reshape(-1, self.states.shape[-1]),
            self.actions.reshape(-1),
            self.logprobs.reshape(-1),
            self.advantages.reshape(-1),
            self.returns.reshape(-1),
            self.values.reshape(-1),
        )
