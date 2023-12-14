import os
import re

def find_latest_checkpoint(env_name):
    folder_name = f"models/{env_name.lower()}_model"
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
    folder_name = f"models/{env_name.lower()}_model"
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
