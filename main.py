from dqn_agent import DQNAgent
from environment import ChromeDinoEnvironment
import config

def train():
    env = ChromeDinoEnvironment()
    agent = DQNAgent()

    for episode in range(config.NUM_EPISODES): 
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()  # this uses experience to learn
            total_reward += reward
            state = next_state
            # if done:
            #     print(f"Episode: {episode}, Total reward: {total_reward}")
            #     break
    env.close()

if __name__ == "__main__":
    train()
