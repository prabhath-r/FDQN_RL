import gym
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

class GymWrapper:
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    
class ChromeDinoEnvironment(GymWrapper):
    def __init__(self):
        chrome_options = Options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("https://6574dd7dc999e52870826b07--extraordinary-custard-4155ae.netlify.app/")
        self.dino = self.driver.find_element(By.TAG_NAME, "body")

    def get_state(self):
        image = self.driver.get_screenshot_as_png() # ss of np array
        image = Image.open(io.BytesIO(image))
        image = image.convert('L') # image to grayscale
        image = image.resize((80, 80), Image.Resampling.LANCZOS) 
        state = np.array(image)
        return state
    
    def check_game_over(self):
        game_over_script = "return Runner.instance_.crashed"  # this checks the instance in js, which says when the game is done.
        game_over = self.driver.execute_script(game_over_script)
        return game_over
    
    def reset(self):
        self.driver.refresh()
        time.sleep(1)
        self.dino = self.driver.find_element(By.TAG_NAME, "body") 
        self.dino.send_keys(Keys.SPACE) 
        return self.get_state()
    
    def step(self, action):
        if action == 1:
            self.dino.send_keys(Keys.SPACE)  # jump
        time.sleep(0.01)  # time between actions
        next_state = self.get_state()
        reward = 0.1
        done = self.check_game_over()
        if done:
            print("Game Over")
        return next_state, reward, done
    
    def close(self):
        self.driver.close()

    # def render(self):
    #     pass

    # To debug the current state as seen by the model
    # def visualize_state(self, state):
    #     if state.ndim == 4:
    #         # If state has a batch dimension, we remove it
    #         state = state.squeeze(0)
    #     plt.imshow(state, cmap='gray')  # No need to squeeze if already 2D or 3D
    #     plt.title("Current State - ChromeDino")
    #     plt.show()
    
class BaseEnvironment(GymWrapper):
    def __init__(self, game_id, render_mode=None):
        self.env = gym.make(game_id, render_mode=render_mode)
        self.state = self.env.reset()

    def reset(self):
        observation = self.env.reset()
        return self.process_observation(observation)

    def step(self, action):
        next_observation, reward, done, *_ = self.env.step(action)
        return self.process_observation(next_observation), reward, done

    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        image = Image.fromarray(observation)
        image = image.convert('L')
        image = image.resize((80, 80), Image.Resampling.LANCZOS)
        return np.array(image)

    def close(self):
        self.env.close()

class BreakoutEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Breakout-v5', render_mode='human')

class PongEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Pong-v5', render_mode='human')

class SpaceInvadersEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/SpaceInvaders-v5', render_mode='human')

class MsPacmanEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/MsPacman-v5', render_mode='human')

class FrostbiteEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Frostbite-v5', render_mode='human')

class QbertEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Qbert-v5', render_mode='human')

class AssaultEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Assault-v5', render_mode='human')

class EnduroEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Enduro-v5', render_mode='human')

class SeaquestEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Seaquest-v5', render_mode='human')

class RiverRaidEnvironment(BaseEnvironment):
    def __init__(self):
        super().__init__('ALE/Riverraid-v5', render_mode='human')



# class BreakoutEnvironment(GymWrapper):
#     def __init__(self):
#         # self.env = gym.make('ALE/Breakout-v5', render_mode = 'human') # if the environment needs to be rendered
#         self.env = gym.make('ALE/Breakout-v5') 

#         self.env.reset()

#     def reset(self):
#         observation = self.env.reset()  
#         return self.process_observation(observation)

#     def step(self, action):
#         next_observation, reward, done, _, _ = self.env.step(action)
#         next_state = self.process_observation(next_observation)
#         return next_state, reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         if isinstance(observation, np.ndarray):
#             image = Image.fromarray(observation)
#         else:
#             raise TypeError("Unsupported observation format")

#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)
    
#     def close(self):
#         self.env.close()

#     # def render(self):
#     #     self.env.render()

#     # To debug the current state as seen by the model
#     # def visualize_state(self, state):
#     #     if state.ndim == 4:
#     #         # If state has a batch dimension, we remove it
#     #         state = state.squeeze(0)
#     #     plt.imshow(state, cmap='gray')  # No need to squeeze if already 2D or 3D
#     #     plt.title("Current State - Breakout")
#     #     plt.show()

# class PongEnvironment(GymWrapper):
#     def __init__(self):
#         # self.env = gym.make('ALE/Pong-v5', render_mode='human') # if the environment needs to be rendered
#         self.env = gym.make('ALE/Pong-v5')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class SpaceInvadersEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/SpaceInvaders-v5', render_mode = 'human')
#         self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         next_observation, reward, done, info, _ = self.env.step(action)

#         next_state = self.process_observation(next_observation)
#         return next_state, reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]  # Assuming the first element is the observation array
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     # def render(self, mode='human'):
#     #     self.env.render(mode=mode)

#     def close(self):
#         self.env.close()

# class MsPacmanEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/MsPacman-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

#     # def render(self):
#     #     self.env.render()

# class FrostbiteEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Frostbite-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class QbertEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Qbert-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class AssaultEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Assault-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class EnduroEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Enduro-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class SeaquestEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Seaquest-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()

# class RiverRaidEnvironment(GymWrapper):
#     def __init__(self):
#         self.env = gym.make('ALE/Riverraid-v5', render_mode = 'human')
#         self.state = self.env.reset()

#     def reset(self):
#         observation = self.env.reset()
#         return self.process_observation(observation)

#     def step(self, action):
#         results = self.env.step(action)
#         next_observation, reward, done, *_ = results
#         return self.process_observation(next_observation), reward, done

#     def process_observation(self, observation):
#         if isinstance(observation, tuple):
#             observation = observation[0]
#         image = Image.fromarray(observation)
#         image = image.convert('L')
#         image = image.resize((80, 80), Image.Resampling.LANCZOS)
#         return np.array(image)

#     def close(self):
#         self.env.close()
