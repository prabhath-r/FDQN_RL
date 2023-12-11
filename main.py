# main.py
import argparse
import os
import re
from dqn_agent import DQNAgent
import config
from environment import ChromeDinoEnvironment, BreakoutEnvironment, PongEnvironment
from torch.utils.tensorboard import SummaryWriter

enable_rendering = False

def find_latest_checkpoint(env_name):
    folder_name = f"{env_name.lower()}_model"
    if not os.path.exists(folder_name):
        return None  

    checkpoints = os.listdir(folder_name)
    checkpoint_regex = re.compile(rf"{env_name}_checkpoint_(\d+).pth")
    checkpoints = [file for file in checkpoints if checkpoint_regex.match(file)]

    if not checkpoints:
        return None
    
    episodes = [int(checkpoint_regex.match(file).group(1)) for file in checkpoints]
    latest_episode = max(episodes)
    return latest_episode

def delete_old_checkpoints(env_name, keep_last_n=1):
    folder_name = f"{env_name.lower()}_model"
    if not os.path.exists(folder_name):
        return

    checkpoint_regex = re.compile(rf"{env_name}_checkpoint_(\d+).pth")
    checkpoints = os.listdir(folder_name)
    checkpoints = [file for file in checkpoints if checkpoint_regex.match(file)]

    if len(checkpoints) <= keep_last_n:
        return

    checkpoints.sort(key=lambda x: int(checkpoint_regex.match(x).group(1)))
    for checkpoint in checkpoints[:-keep_last_n]:
        os.remove(os.path.join(folder_name, checkpoint))

def train():
    log_dir = f"runs/{config.GAME_ENV}"
    writer = SummaryWriter(log_dir)

    if config.GAME_ENV == 'ChromeDino':
        config.ACTION_SIZE = 2
        env = ChromeDinoEnvironment()
    elif config.GAME_ENV == 'Breakout':
        config.ACTION_SIZE = 4
        env = BreakoutEnvironment()
    elif config.GAME_ENV == 'Pong':
        config.ACTION_SIZE = 6
        env = PongEnvironment()
    else:
        raise ValueError("Select from the list [Breakout, ChromeDino, Pong]")

    agent = DQNAgent()

    latest_checkpoint = find_latest_checkpoint(config.GAME_ENV)
    start_episode = 0
    if latest_checkpoint is not None:
        print(f"Resuming from episode {latest_checkpoint}")
        agent.load_checkpoint(latest_checkpoint)
        start_episode = latest_checkpoint+1

    for episode in range(start_episode, config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_loss = 0
        while True:
            # env.visualize_state(state)
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            loss = agent.learn()
            if loss is not None:
                episode_loss += loss
                writer.add_scalar('Loss/train', loss, episode)
            writer.add_scalar('Reward/train', total_reward, episode)

            if enable_rendering:
                if config.GAME_ENV == 'ChromeDino':
                    pass  # no rendering for Chrome Dino, we will see that using chromedriver for that
                else:
                    env.render()

            if done:
                break

        print(f"Episode: {episode}, Total Reward: {total_reward}")
        if episode % 50 == 0:
            delete_old_checkpoints(config.GAME_ENV, keep_last_n=1)  # Delete old checkpoints
            agent.save_checkpoint(episode)

    env.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_env", type=str, required=True, 
                        choices=["Breakout", "ChromeDino", "Pong"],
                        help="Specify the game environment: 'Breakout' or 'ChromeDino' or 'Pong")
    args = parser.parse_args()

    config.GAME_ENV = args.game_env
    train()
