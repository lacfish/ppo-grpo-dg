import numpy as np
from agent import REINFORCEagent


class BanditEnv:
    def __init__(self, k_arms=5):
        self.k_arms = k_arms
        self.probs = np.random.uniform(0.1, 0.9, size=self.k_arms)
        self.best_arm = np.argmax(self.probs)

    def reset(self):
        return np.array([1.0], dtype=np.float32)

    def step(self, action):
        random_num = np.random.rand()
        reward = 0
        if random_num < self.probs[action]:
            reward = 1
        done = True
        return reward, done


def train():
    k_arms = 5
    env = BanditEnv(k_arms)
    print("=" * 50)
    print(f"老虎机每个拉杆的中奖概率: {np.round(env.probs, 3)}")
    print(f"理论最优拉杆编号: {env.best_arm} (概率: {env.probs[env.best_arm]:.3f})")
    print("=" * 50)

    total_iterations = 10000
    batch_size = 30
    agent = REINFORCEagent(1, 32, k_arms, 0.01)
    for i in range(total_iterations):
        state = env.reset()
        action, cache = agent.select_action(state)
        reward, done = env.step(action)
        agent.buffer.insert(reward, action, cache)
        if (i + 1) % batch_size == 0:
            agent.update()
        if (i + 1) % 200 == 0:
            final_probs, _ = agent.policy.forward(state)
            predicted_arm = np.argmax(final_probs)
            print(
                f"训练次数 {i + 1:4d} | 各臂预测概率: {np.round(final_probs, 3)} | 偏好拉杆: {predicted_arm}"
            )

if __name__ == "__main__":
    train()
