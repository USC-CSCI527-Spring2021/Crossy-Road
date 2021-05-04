from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2 as cv

GAME_URL = 'file:///Users/shuweizhang/Documents/Studies_Local/527/CrossyRoad/web/index.html'
WINDOW_SIZE_W = 1200
WINDOW_SIZE_H = 600


class Game:
    def __init__(self, on_cloud=True):
        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(chrome_options=chrome_options)
        self.driver = webdriver.Chrome()
        self.driver.set_window_size(WINDOW_SIZE_W, WINDOW_SIZE_H)
        self.driver.get(GAME_URL)
        self._paused = False
        self.element = self.driver.find_element_by_id('retry')

    def forward(self):
        self.driver.find_element_by_id("forward").send_keys(Keys.UP)

    def backward(self):
        self.driver.find_element_by_id("backward").send_keys(Keys.DOWN)

    def left(self):
        self.driver.find_element_by_id('left').send_keys(Keys.LEFT)

    def right(self):
        self.driver.find_element_by_id('right').send_keys(Keys.RIGHT)

    def get_score(self):
        text_score = self.driver.find_element_by_id('counter').text
        return int(text_score)

    def get_crashed(self):
        retry_element = self.driver.find_element_by_id('retry')
        return retry_element.is_displayed()

    def restart_game(self):
        WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, "retry"))).click()

    def end(self):
        self.driver.close()

    def pause_or_resume(self):
        action = ActionChains(self.driver)
        action.send_keys(Keys.ENTER).perform()
        self._paused = not self._paused

    def pause(self):
        if not self._paused:
            self.pause_or_resume()

    def resume(self):
        if self._paused:
            self.pause_or_resume()

    def get_paused(self):
        return self._paused

    def end_game(self):
        self.driver.close()


class Agent_Setting:
    def __init__(self,
                 agent_id,
                 score_dep=True,
                 delta_score=True,
                 move_list=[0.1, 0.1, 0.1, 0.1],
                 reward_weights=[1, 1, 1, 1],
                 dead_punishment=-10,
                 resize_w=80,
                 resize_h=80,
                 canny_th1=100,
                 canny_th2=200,
                 learning_rate=1e-4,
                 initial_epsilon=0.2,
                 final_epsilon=0.001,
                 observe=10000,
                 explore=10000,
                 replay_memory=5000,
                 batch=32,
                 gamma=0.99):
        self.agent_id = agent_id
        self.score_dep = score_dep
        self.delta_score = delta_score
        self.move_list = move_list
        self.reward_weights = reward_weights
        self.dead_punishment = dead_punishment
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.observe = observe
        self.explore = explore
        self.replay_memory = replay_memory
        self.batch = batch
        self.gamma = gamma


class Game_State:
    def __init__(self, game, reward_setting):
        self.game = game
        self.reward_setting = reward_setting

    def processing_img(self):
        img_64 = self.game.driver.find_element_by_id('imgURL').text
        screen = np.array(Image.open(BytesIO(base64.b64decode(img_64))))
        img = cv.cvtColor(screen, cv.COLOR_RGBA2GRAY)
        img = img[200:400, 650:850]
        img = cv.resize(img, (self.reward_setting.resize_w, self.reward_setting.resize_h))
        img = cv.Canny(img,
                       threshold1=self.reward_setting.canny_th1,
                       threshold2=self.reward_setting.canny_th2)

        return img

    def get_reward(self, action, is_dead, old_score):
        if action == -1:
            return 0

        if is_dead:
            return self.reward_setting.dead_punishment

        if not self.reward_setting.score_dep:
            return self.reward_setting.move_list[action]

        score = self.game.get_score()

        if self.reward_setting.delta_score:
            score -= old_score

        return score * self.reward_setting.reward_weights[action]

    def get_state(self, action):
        old_score = self.game.get_score()

        if action == 0:
            self.game.forward()
        elif action == 1:
            self.game.left()
        elif action == 2:
            self.game.right()
        elif action == 3:
            self.game.backward()

        image = self.processing_img()
        is_dead = self.game.get_crashed()
        new_score = self.game.get_score()
        reward = self.get_reward(action, is_dead, old_score)

        if is_dead:
            self.game.restart_game()

        return image, reward, is_dead, new_score


# class Chicken:
#     def __init__(self, game):
#         self._game = game
#
#     def forward(self):
#         self._game.press_up()
#
#     def backward(self):
#         self._game.press_down()
#
#     def left(self):
#         self._game.press_left()
#
#     def right(self):
#         self._game.press_right()
#
#     def is_crashed(self):
#         return self._game.get_crashed()
#
#     def pause_or_resume(self):
#         self._game.pause_or_resume()
