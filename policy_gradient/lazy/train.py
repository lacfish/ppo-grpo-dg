import gymnasium as gym
import torch
from agent import REINFORCEagent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')
    agent = REINFORCEagent(device=device, lr=1e-2)
    
    for i in range(1,10000):
        state,_=env.reset()
        while True:
            action,logprob=agent.select_action_and_return_logprob(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.buffer.insert(reward, logprob)
            state=next_state
            
            if terminated or truncated:
                break
        agent.update()




if __name__ == '__main__':
    main()