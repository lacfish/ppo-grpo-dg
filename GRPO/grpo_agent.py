import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class GRPO:
    def __init__(
        self,
        network,
        lr=2.5e-4,
        clip_coef=0.2,
        beta=0.0,
        ent_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.network = network
        self.lr = lr
        self.clip_coef = clip_coef
        self.beta = beta
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)

        # 加入self.ref_network，若self.beta>0,则需要KL,复制当前network作为ref_network
        self.ref_network = None
        if self.beta > 0:
            self.ref_network = copy.deepcopy(network)
            self.ref_network.eval()
            for param in self.ref_network.parameters():
                param.requires_grad_(False)

    def update(
        self,
        b_state,
        b_actions,
        b_logprobs,
        b_advantages,
        b_weights,
        batch_size,
        minibatch_size,
        update_epochs,
    ):
        # 在开始计算ref_network的logprob
        if self.beta > 0:
            with torch.no_grad():
                _, ref_log_prob, _ = self.ref_network.get_action_logprob_and_entropy(
                    b_state, b_actions
                )
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start : start + minibatch_size]
                mb_weights = b_weights[mb_inds]
                _, newlogprob, entropy = self.network.get_action_logprob_and_entropy(
                    b_state[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                actor_loss1 = -ratio * mb_adv
                actor_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                actor_loss = (torch.max(actor_loss1, actor_loss2) * mb_weights).mean()

                entropy_loss = (entropy * mb_weights).mean()

                kl_loss = 0.0
                if self.beta > 0:
                    u_t = ref_log_prob[mb_inds] - newlogprob
                    kl = u_t.exp() - u_t - 1
                    kl_loss = (mb_weights * kl).mean()

                loss = actor_loss + kl_loss * self.beta - entropy_loss * self.ent_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
