# PPO-GRPO-DG

本仓库用于学习、实现和比较 PPO、GRPO 与 DG 相关的强化学习算法。当前代码以 Gymnasium 的 `CartPole-v1` 为实验环境，使用 PyTorch 构建 Actor-Critic 网络。

> 项目目前处于开发阶段。仓库中包含 PPO 基础实现和 GRPO 原型，DG 模块尚待补充。

## 项目结构

```text
ppo-grpo-dg/
|-- PPO/
|   |-- buffer.py      # 轨迹缓存与 GAE 计算
|   |-- networks.py    # Actor-Critic 网络
|   |-- ppo_agent.py   # PPO 参数更新
|   `-- train.py       # CartPole 训练入口
|-- GRPO/
|   |-- buffer.py      # 组回报、相对优势与样本权重
|   |-- networks.py    # Actor 网络
|   |-- grpo_agent.py  # GRPO 参数更新与 KL 正则
|   `-- train.py       # CartPole 训练入口
|-- .gitignore
`-- README.md
```

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

运行 GRPO 原型：

```powershell
python GRPO\train.py
```

训练脚本会自动选择 CUDA；如果 CUDA 不可用，则使用 CPU。

## 后续计划

- 添加 DG 模块
- 添加测试及 PPO、GRPO、DG 对比实验
