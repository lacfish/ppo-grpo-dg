# PPO-GRPO-DG

本仓库用于学习、实现和比较 Policy Gradient、PPO、PPO + Score-Centering、GRPO 与 DG 相关的强化学习算法。其中，Policy Gradient 部分实现了基础的 REINFORCE 算法，并提供 PyTorch 版本和手动推导版本，便于对照理解自动求导背后的计算过程。

> 项目目前处于开发阶段。仓库中包含 Policy Gradient、PPO、PPO + Score-Centering、DG 实现和 GRPO 原型。

## 项目结构

```text
ppo-grpo-dg/
|-- PPO/
|   |-- buffer.py      # 轨迹缓存与 GAE 计算
|   |-- networks.py    # Actor-Critic 网络
|   |-- ppo_agent.py   # PPO 参数更新
|   `-- train.py       # CartPole 训练入口
|-- score_centering/
|   |-- buffer.py                  # 轨迹缓存、采样策略分布与 GAE 计算
|   |-- networks.py                # Actor-Critic 网络与完整动作分布
|   |-- score_centering_agent.py   # PPO + Score-Centering 参数更新
|   `-- train.py                   # CartPole 训练入口
|-- GRPO/
|   |-- buffer.py      # 组回报、相对优势与样本权重
|   |-- networks.py    # Actor 网络
|   |-- grpo_agent.py  # GRPO 参数更新与 KL 正则
|   `-- train.py       # CartPole 训练入口
|-- DG/
|   |-- agent.py       # DG 动作采样与参数更新
|   |-- buffer.py      # 回报、优势、动作与前向缓存
|   |-- policy_net.py  # NumPy 策略网络与手动反向传播
|   `-- train.py       # 多臂老虎机训练入口
|-- policy_gradient/
|   |-- lazy/          # 基于 PyTorch 自动求导的 REINFORCE 实现
|   |   |-- agent.py       # 动作采样与策略更新
|   |   |-- buffer.py      # 回报、优势与 log probability 缓存
|   |   |-- policy_net.py  # PyTorch 策略网络
|   |   `-- train.py       # CartPole 训练入口
|   `-- detailed/      # 在 lazy 版本基础上展开的手动实现
|       |-- agent.py       # 手动梯度更新
|       |-- buffer.py      # 回报、动作与前向缓存
|       |-- policy_net.py  # NumPy 前向传播与反向传播
|       `-- train.py       # 多臂老虎机训练入口
|-- .gitignore
`-- README.md
```

## Policy Gradient 实现

`policy_gradient` 目录包含两种 REINFORCE 实现。两者遵循相同的基本流程：使用策略网络输出动作概率、按概率采样动作、根据采样轨迹计算折扣回报与优势，最后利用策略梯度更新网络参数。

- `lazy`：使用 PyTorch 实现策略网络，借助 `torch.distributions.Categorical` 采样动作，并通过自动求导和 Adam 优化器完成参数更新。该版本保留较高层的实现方式，代码更精简。
- `detailed`：以 `lazy` 版本的算法流程为基础，将其中由 PyTorch 隐式完成的步骤展开。该版本使用 NumPy 手动实现线性层、ReLU、Softmax、策略梯度反向传播及梯度下降，主要用于理解 REINFORCE 的底层计算过程。

目前两个版本使用不同的示例环境：`lazy` 在 Gymnasium 的 `CartPole-v1` 上训练，`detailed` 使用内置的多臂老虎机环境演示手动梯度计算。

## PPO + Score-Centering 实现

`score_centering` 目录以 `PPO` 实现为基础，保留相同的 Actor-Critic 网络、GAE、优势标准化、PPO clipped objective、critic loss、entropy bonus 和优化器配置，并在策略损失中加入 score-centering 修正。

采样时，buffer 额外保存行为策略对所有动作的概率分布 `q_probs`。更新时，当前策略重新计算 `p_theta_probs`，并在 PPO clipping 对应的有效动作区域内计算期望 score，从 PPO actor loss 中消除由采样策略与当前策略不一致产生的梯度漂移。

该目录用于在尽量控制网络结构、采样规模和训练超参数一致的条件下，与 `PPO` 目录进行对比实验。

## DG 实现

`DG` 目录实现了 Delightful Policy Gradient。该实现使用 NumPy 构建两层策略网络，并手动完成前向传播和梯度更新。

更新时，算法根据动作的 surprisal、优势值和温度参数 `eta` 计算 delight，并使用对应的 sigmoid 权重调整策略梯度。轨迹缓存负责计算折扣回报、均值 baseline 和标准化优势。

当前实现使用五臂老虎机环境，训练过程中输出各拉杆的真实中奖概率、策略预测概率和当前偏好的拉杆。

## 环境要求

- Python 3.10 或更高版本
- PyTorch
- Gymnasium
- NumPy

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch gymnasium numpy
```

## 运行

在仓库根目录运行 PPO：

```powershell
python PPO\train.py
```

运行 PPO + Score-Centering：

```powershell
python score_centering\train.py
```

运行 GRPO 原型：

```powershell
python GRPO\train.py
```

运行 DG：

```powershell
python DG\train.py
```

运行 PyTorch 版 Policy Gradient：

```powershell
python policy_gradient\lazy\train.py
```

运行手动实现的 Policy Gradient：

```powershell
python policy_gradient\detailed\train.py
```

基于 PyTorch 的训练脚本会自动选择 CUDA；如果 CUDA 不可用，则使用 CPU。`DG` 和 `detailed` 版本仅依赖 NumPy，在 CPU 上运行。

## 后续计划

添加PPO、PPO + Score-Centering对比实验
