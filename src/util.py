from collections import deque
import os
import pickle
from marshmallow import Schema, fields, post_load
from agent import Agent_Setting


def save_obj(obj, name):
    with open('../Objects/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('../Objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Setting_Schema(Schema):
    agent_id = fields.Str()
    score_dep = fields.Bool()
    move_list = fields.List
    reward_weights = fields.List
    dead_punishment = fields.Int()
    resize_w = fields.Int()
    resize_h = fields.Int()
    canny_th1 = fields.Int()
    canny_th2 = fields.Int()

    @post_load
    def read_setting(self, data):
        return Agent_Setting(**data)


def init_cache():
    if not os.path.isfile('time.pkl'):
        initial_epsilon = 0.4
        save_obj(initial_epsilon, 'epsilon')
        t = 0
        save_obj(t, 'time')
        memo = deque()
        save_obj(memo, "memory")
