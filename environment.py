from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import numpy as np
from PIL import Image
import io
# import base64

class ChromeDinoEnvironment:
    def __init__(self):
        chrome_options = Options()
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.get("file:////Users/prabhathr/Desktop/Projects/DQN_RL_Game/chrome_dino.html") # abs file path containing the html file
        self.dino = self.driver.find_element(By.TAG_NAME, "body")

    def get_state(self):
        image = self.driver.get_screenshot_as_png() #to capture ss as a np.array
        image = Image.open(io.BytesIO(image))
        image = image.convert('L')  #convert ss to grayscale
        image = image.resize((80, 80), Image.Resampling.LANCZOS)  # resize to smaller size, for faster processing
        state = np.array(image)
        return state

    def reset(self):
        self.driver.refresh()
        time.sleep(2)  # time to wait for game to load before taking action
        self.dino = self.driver.find_element(By.TAG_NAME, "body") 
        self.dino.send_keys(Keys.SPACE)  # start the game
        return self.get_state()

    def step(self, action):
        if action == 1:
            self.dino.send_keys(Keys.SPACE)  # jump
        time.sleep(0.1)  # time between actions in sec
        next_state = self.get_state()
        reward = 0.1
        done = False
        return next_state, reward, done

    def close(self):
        self.driver.close()
