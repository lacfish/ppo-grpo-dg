import gymnasium as gym
import torch
from networks import Actor
from buffer import RollOutBuffer
from grpo_agent import GRPO

num_envs = 4
num_steps = 128
minibatch_size = 32
update_epochs = 4
total_iterations = 200
beta = 0.04
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envs = gym.vector.SyncVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]
)
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.n

network = Actor(state_dim, action_dim).to(device)
agent = GRPO(network, beta)
buffer = RollOutBuffer(num_steps, num_envs, device, state_dim)

next_state, _ = envs.reset()
next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

for i in range(total_iterations):
    active_masks = torch.ones(num_envs, device=device)
    next_state, _ = envs.reset()
    next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

    for t in range(num_steps):
        with torch.no_grad():
            action, logprob, _ = network.get_action_logprob_and_entropy(next_state)
            state, reward, terminations, truncations, _ = envs.step(
                action.cpu().numpy()
            )

            buffer.insert(
                next_state,
                action,
                logprob,
                torch.tensor(reward, dtype=torch.float32, device=device),
                active_masks,
            )
            next_done = terminations | truncations
            active_masks = active_masks * (
                1 - torch.tensor(next_done, dtype=torch.float32, device=device)
            )
            next_state = torch.tensor(state, dtype=torch.float32, device=device)
            if active_masks.sum() == 0:
                break

    buffer.compute_returns_and_advantages()

    b_state, b_actions, b_logprobs, b_advantages, b_weights = (
        buffer.get_flatten_batches()
    )
    batch_size = len(b_weights)
    buffer.reset()

    agent.update(
        b_state,
        b_actions,
        b_logprobs,
        b_advantages,
        b_weights,
        batch_size,
        minibatch_size,
        update_epochs,
    )

envs.close()
