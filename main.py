from dqn_agent import DQNAgent
from environment import ChromeDinoEnvironment
import config

def train():
    env = ChromeDinoEnvironment()
    agent = DQNAgent()
    # agent.load_checkpoint('checkpoint_0.pth') #load previous model
    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn() 
            total_reward += reward
            state = next_state
            if done:
                break
        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if episode % 20 == 0:
            agent.save_checkpoint(f'checkpoint_{episode}.pth')
    env.close()

    

if __name__ == "__main__":
    train()
