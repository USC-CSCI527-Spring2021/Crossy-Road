import random
import os
import pandas as pd
import numpy as np

from agent import Game
from agent import Agent_Setting
from agent import Game_State
from util import save_obj
from util import load_obj
from util import init_cache
from model import build_model

from keras.optimizers import Adam

ACTIONS = 4

MODEL_FILE_PATH = '../objects/model.h5'
LOSS_FILE_PATH = '../objects/loss_df.csv'
ACTIONS_FILE_PATH = '../objects/actions_df.csv'
Q_VALUES_FILE_PATH = '../objects/q_values.csv'
SCORES_FILE_PATH = '../objects/scores_df.csv'


class Task:
    def __init__(self):
        self.loss_df = pd.read_csv(LOSS_FILE_PATH) if os.path.isfile(LOSS_FILE_PATH) else pd.DataFrame(columns=['loss'])
        self.q_values_df = pd.read_csv(Q_VALUES_FILE_PATH) if os.path.isfile(Q_VALUES_FILE_PATH) else pd.DataFrame(
            columns=['qvalues'])
        self.game_state = Game_State(Game(), Agent_Setting(1))
        self.model = build_model()

    def train_nn(self):
        init_cache()
        memo = load_obj('memory')
        do_noting = -1
        t = load_obj('time')
        epsilon = load_obj('epsilon')

        if os.path.isfile(MODEL_FILE_PATH):
            self.model.load_weights(MODEL_FILE_PATH)
            adam = Adam(learning_rate=self.game_state.reward_setting.learning_rate)
            self.model.compile(loss='mse', optimizer=adam)

        image_t, reward_0, is_dead, game_score = self.game_state.get_state(do_noting)
        state_t = np.stack((image_t, image_t, image_t, image_t), axis=2)
        state_t = state_t.reshape(1, state_t.shape[0], state_t.shape[1], state_t.shape[2])

        while True:
            loss = 0
            Q_sa = 0
            reward_t = 0
            action_t = do_noting

            self.game_state.game.pause()

            if t < self.game_state.reward_setting.observe:
                action_t = random.randrange(ACTIONS)
            else:
                pred = self.model.predict(state_t)
                action_t = np.argmax(pred)

            self.game_state.game.resume()

            if epsilon > self.game_state.reward_setting.final_epsilon and t > self.game_state.reward_setting.observe:
                epsilon -= \
                    (self.game_state.reward_setting.initial_epsilon - self.game_state.reward_setting.final_epsilon) \
                    / self.game_state.reward_setting.explore

            image_t1, reward_t1, is_dead, game_score = self.game_state.get_state(action_t)
            image_t1 = image_t1.reshape(1, image_t1.shape[0], image_t1.shape[1], 1)
            state_t1 = np.append(image_t1, state_t[:, :, :, :3], axis=3)

            memo.append((state_t, action_t, reward_t, state_t1, is_dead))

            self.game_state.game.pause()

            if t > self.game_state.reward_setting.observe:
                minibatch = random.sample(memo, self.game_state.reward_setting.batch)
                inputs = np.zeros(
                    (self.game_state.reward_setting.batch, state_t.shape[1], state_t.shape[2], state_t.shape[3]))
                targets = np.zeros((inputs.shape[0], ACTIONS))

                for i in range(len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    is_dead = minibatch[i][4]

                    inputs[i:i+1] = state_t
                    targets[i] = self.model.predict(state_t)
                    Q_sa = self.model.predict(state_t1)

                    if is_dead:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + self.game_state.reward_setting.gamma * np.max(Q_sa)

                loss += self.model.train_on_batch(inputs, targets)
                self.loss_df.loc[len(self.loss_df)] = loss
                self.q_values_df.loc[len(self.q_values_df)] = np.max(Q_sa)

            state_t = state_t1
            t += 1

            if t % 200 == 0:
                print('Now we save model')
                self.model.save_weights(MODEL_FILE_PATH, overwrite=True)
                save_obj(memo, 'memory')
                save_obj(t, 'time')
                save_obj(epsilon, 'epsilon')
                self.loss_df.to_csv(LOSS_FILE_PATH, index=False)
                self.q_values_df.to_csv(Q_VALUES_FILE_PATH, index=False)
                print('Finished Saving')

            state = 'train'
            if t <= self.game_state.reward_setting.observe:
                state = 'observe'
            elif self.game_state.reward_setting.observe < t \
                    <= self.game_state.reward_setting.observe + self.game_state.reward_setting.explore:
                state = 'explore'

            print("TIMESTEP", t,
                  "/ STATE", state,
                  "/ EPSILON", epsilon,
                  "/ ACTION", action_t,
                  "/ REWARD", reward_t,
                  "/ SCORE", game_score,
                  "/ Q_MAX ", np.max(Q_sa),
                  "/ Loss ", loss,
                  "/ Is Dead", is_dead)

            self.game_state.game.resume()
