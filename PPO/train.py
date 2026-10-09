import gymnasium as gym
import torch
from networks import ActorCritic
from buffer import RollOutBuffer
from ppo_agent import PPO

# 设置超参数，num_envs,num_steps,batch_size,minibatch_size,update_epochs,total_iterations,device
num_envs = 4
num_steps = 128
batch_size = num_envs * num_steps
minibatch_size = 32
update_epochs = 4
total_iterations = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
)
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n

# 初始化network,agent,buffer
network = ActorCritic(state_dim, action_dim).to(device)
agent = PPO(network)
buffer = RollOutBuffer(num_steps, num_envs, device, state_dim)

# 初始化游戏，为每个环境副本获取最初状态以及设立游戏是否结束的flag
next_state, _ = envs.reset()
next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
next_done = torch.zeros(num_envs, device=device)

# 一层循环total_iterations次，二层循环num_steps次，先用networks.py产生动作，再与环境交互返回state,reward,
# terminations,truncations，用insert存入buffer，一轮采样结束后，buffer.compute_returns_and_advantages(next_value, next_done)计算优势值
# 最后agent_update进行迭代
for i in range(total_iterations):
    for t in range(num_steps):
        with torch.no_grad():
            action, logprob, _, value = network.get_action_and_value(next_state)
            state, reward, terminations, truncations, _ = envs.step(
                action.cpu().numpy()
            )
            value = value.reshape(-1)

            buffer.insert(
                next_state,
                action,
                logprob,
                torch.tensor(reward, dtype=torch.float32, device=device),
                next_done,
                value,
            )
            next_done = terminations | truncations
            next_state = torch.tensor(state, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done, dtype=torch.float32, device=device)

    with torch.no_grad():
        _, _, _, next_value = network.get_action_and_value(next_state)
    next_value = next_value.reshape(-1)
    buffer.compute_returns_and_advantages(next_value, next_done)

    b_state, b_actions, b_logprobs, b_returns, b_advantages, b_values = (
        buffer.get_flatten_batches()
    )
    agent.update(
        b_state,
        b_actions,
        b_logprobs,
        b_returns,
        b_advantages,
        b_values,
        batch_size,
        minibatch_size,
        update_epochs,
    )

envs.close()
