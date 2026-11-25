# PPO 与 Score Centering 比较

## 各任务族得分

差值为 `PPO+SC − PPO`

| 任务族              |      PPO |   PPO+SC |      差值 |
| ------------------- | -------: | -------: | --------: |
| bandit              | 0.947714 | 0.947714 |  0.000000 |
| bandit_noise        | 0.565791 | 0.565791 |  0.000000 |
| bandit_scale        | 0.896395 | 0.896396 | +0.000001 |
| cartpole            | 0.636019 | 0.632232 | -0.003788 |
| cartpole_noise      | 0.564125 | 0.509164 | -0.054961 |
| cartpole_scale      | 0.316225 | 0.291522 | -0.024703 |
| cartpole_swingup    | 0.000000 | 0.000000 |  0.000000 |
| catch               | 0.764325 | 0.764750 | +0.000425 |
| catch_noise         | 0.221266 | 0.221255 | -0.000011 |
| catch_scale         | 0.270442 | 0.270403 | -0.000038 |
| deep_sea            | 0.000000 | 0.000000 |  0.000000 |
| deep_sea_stochastic | 0.000000 | 0.000000 |  0.000000 |
| discounting_chain   | 0.251720 | 0.250640 | -0.001080 |
| memory_len          | 0.043478 | 0.043478 |  0.000000 |
| memory_size         | 0.000000 | 0.000000 |  0.000000 |
| mnist               | 0.568144 | 0.570856 | +0.002711 |
| mnist_noise         | 0.238558 | 0.238162 | -0.000396 |
| mnist_scale         | 0.232919 | 0.234171 | +0.001253 |
| mountain_car        | 0.104638 | 0.103190 | -0.001448 |
| mountain_car_noise  | 0.100589 | 0.100361 | -0.000228 |
| mountain_car_scale  | 0.101639 | 0.100886 | -0.000753 |
| umbrella_distract   | 0.034783 | 0.026087 | -0.008696 |
| umbrella_length     | 0.278261 | 0.278261 |  0.000000 |

因此，本次完整实验的结论是：**PPO + Score Centering 与 PPO 的整体表现基本相同，没有显示出明显的总体提升**

# 分析1：PPO不应与score_centering结合，而是应该独立编码

设在状态（或上下文）$x$ 下：

- $q_a=q(a\mid x)$：采集 rollout 的旧策略概率；
- $p_a=p_\theta(a\mid x)$：当前训练策略概率；
- $r_a=p_a/q_a$：PPO importance ratio；
- $s_a=\nabla_\theta\log p_a$：动作 $a$ 的 score；
- $A$：当前样本的 advantage；
- $\epsilon$：PPO clipping 系数。

## 1. 论文中的基础 Score Centering

论文首先定义采样策略 $q$ 下的期望 score：

$$
\bar{s}
=\mathbb E_{a\sim q}[s_a]
=\sum_a q_a\nabla_\theta\log p_a.
$$

当 $p\neq q$ 时，该期望通常不为 0。基础 Score Centering 将样本 score 替换为：

$$
\tilde{s}_{a_t}
=s_{a_t}-\bar{s},
\qquad a_t\sim q.
$$

因此策略梯度样本变为：

$$
g_{\mathrm{SC}}
=A\left(
s_{a_t}-\sum_a q_a s_a
\right).
$$

## 2. 与 PPO importance weighting 结合

 **首先，假设没有任何裁剪的发生**

PPO 的梯度带有 importance ratio：

$$
g_{\mathrm{PPO}}
=A r_{a_t}s_{a_t}.
$$

PPO+score centering的梯度变为：

$$
g
=A\left(
r_{a_t}s_{a_t}
-\sum_a q_a r_a s_a
\right).
$$

考虑修正项：

$$
\sum_a q_a r_a s_a
=\sum_a q_a\frac{p_a}{q_a}s_a
=\sum_a p_a s_a.
$$

又因为：

$$
\sum_a p_a s_a
=\sum_a p_a\nabla_\theta\log p_a
=\sum_a\nabla_\theta p_a
=\nabla_\theta\sum_a p_a
=\nabla_\theta 1
=0,
$$

所以假如没有发生裁剪，importance weighting 已经使加权期望 score 为 0。此时 Score Centering 的修正就没有意义。

# 分析2：本实验中的 mismatch 很小

mismatch 指采集动作的旧策略 $q$ 与当前正在训练的策略 $p$ 不一致：

$$
q(a\mid x)\neq p(a\mid x).
$$

论文中的 Score Centering 主要用于处理以下几种较强且持续存在的 mismatch：

1. **采样端量化。** 例如，推理端为了加速，使用 INT8 或 FP8 生成 rollout，而训练端使用 BF16。低精度计算会改变动作概率，所以即使两端加载的是同一份模型参数，采样策略 $q$ 和训练策略 $p$ 也可能从 rollout 生成时就不相同。
2. **训练引擎和推理引擎不一致。** 采样和训练可能使用不同的程序、计算 kernel、batch 形状或浮点运算顺序。微小的数值差异会使两端输出不同的动作概率，并在持续训练中逐渐积累。
3. **严重 staleness。** 在异步训练中，采样器可能隔几十个训练 step 才同步一次参数。此时采样器仍使用很久以前的旧模型 $q$ 生成数据，而训练器 $p$ 已经更新了很多次，二者可能明显不同。

本实验没有主动制造这些条件：

- rollout 采样和梯度训练使用同一个 PyTorch 网络实现，不存在两套不同的训练、推理引擎；
- 采样端和训练端使用相同数值精度，没有只对采样端进行 INT8、FP8 等量化；
- 没有异步采样器，采样策略不会落后当前策略几十个训练 step；

本实验中的 mismatch 只在同一批数据被重复更新的短时间内出现：

```text
采集一批 rollout：p = q
        ↓
冻结旧策略 q，执行第 1 次更新
        ↓
当前策略 p 发生小幅变化：p != q
        ↓
继续复用这一批数据，最多更新到第 4 个 epoch
        ↓
使用更新后的策略采集下一批数据：重新令 p = q
```

因此，本实验中的 $p\neq q$ 预计持续时间短、差异也较小。

# 后续实验计划：以Unclipped PG为baseline,将不加 Clipping 的独立 Score Centering 与 PPO 对比

下一步实现不与 PPO clipping 组合的独立 Score Centering，并以unclipped PG为baseline与标准 PPO 比较。

独立 Score Centering 的策略梯度为：

$$
g_{\mathrm{SC}}
=A\left[
\nabla_\theta\log p(a_t\mid x)
-\sum_a q(a\mid x)\nabla_\theta\log p(a\mid x)
\right],
\qquad a_t\sim q.
$$

该算法不使用 PPO clipping，也不在采样动作项前乘 importance ratio $p(a_t\mid x)/q(a_t\mid x)$。独立 Score Centering 优于 Unclipped PG，能够说明收益来自 Score Centering 本身,以PG为基线可以与PPO展开比较。
