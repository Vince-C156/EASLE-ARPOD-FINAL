from reward_shaping import reward_formulation
from visualizer import write2text
#write2text(chaser, data_dir, file_name, step)
import numpy as np
import pdb

class ARPOD:

    def __init__(self, chaser):
        self.r_form = reward_formulation(chaser)
        self.chaser = chaser
        self.info = {}

    def step(self, action):
        reward = 0
        done = False
        obs = self.chaser.get_next(action)
        self.chaser.update_state(obs)

        p, done = self.r_form.terminal_conditions()
        reward += p

        if done:
            return obs, reward, done, self.info

        r, done = self.r_form.win_conditions()
        reward += r

        if done:
            return obs, reward, done, self.info

        p = self.r_form.soft_penalities()
        reward += p

        r = self.r_form.soft_rewards()
        reward += r

        return obs, reward, done, self.info

    def write_data(self, file_name):
        print('writing data to text')
        print(f'current step {self.chaser.current_step}')
        write2text(self.chaser, 'runs', file_name, self.chaser.current_step)


    def reset(self):
        self.chaser.reset()
        self.chaser.update_state(self.chaser.state)
        print('reseting environment')
        return np.array(self.chaser.state, copy=True)
