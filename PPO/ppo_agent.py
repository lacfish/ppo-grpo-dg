import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


# 定义类PPO,传入network，初始化optimizer,learning rate, clip, value function, entropy coefficient,max_grad_norm
class PPO:
    def __init__(
        self,
        network,
        lr=2.5e-4,
        clip_coef=0.2,
        critic_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.network = network
        self.lr = lr
        self.clip_coef = clip_coef
        self.critic_coef = critic_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

    # update函数，传入buffer中的数据，batch_size,minibatch_size,update_epochs（单个batch重复训练次数），两层循环，第一层
    # 根据训练次数循环，为每个step赋予一个index，打乱index，第二层以minibatch_size为步长，调用self.network.get_action_and_value
    # 获取newlogprob, entropy, newvalue，计算actor_loss,critic_loss,entropy_loss，再计算loss,并使用optimizer更新参数
    def update(
        self,
        b_state,
        b_actions,
        b_logprobs,
        b_returns,
        b_advantages,
        b_values,
        batch_size,
        minibatch_size,
        update_epochs,
    ):
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    b_state[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                actor_loss1 = -ratio * mb_adv
                actor_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                actor_loss = torch.max(actor_loss1, actor_loss2).mean()

                critic_loss = (
                    0.5 * ((newvalue.reshape(-1) - b_returns[mb_inds]) ** 2).mean()
                )

                entropy_loss = entropy.mean()

                loss = (
                    actor_loss
                    + critic_loss * self.critic_coef
                    - entropy_loss * self.ent_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
