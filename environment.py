from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import numpy as np
from PIL import Image
import io

class ChromeDinoEnvironment:
    def __init__(self):
        chrome_options = Options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("file:////Users/prabhathr/Desktop/Projects/DQN_Framwork_RL-Game/chrome_dino.html")
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
        time.sleep(0.1)  # time between actions
        next_state = self.get_state()
        reward = 0.1
        done = self.check_game_over()
        if done:
            print("Game Over")

        return next_state, reward, done


    def close(self):
        self.driver.close()
