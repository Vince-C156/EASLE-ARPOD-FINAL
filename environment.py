from reward_shaping import reward_formulation
from visualizer import write2text
#write2text(chaser, data_dir, file_name, step)
import numpy as np
import pdb

class ARPOD:

    def __init__(self, chaser):
        self.r_form = reward_formulation(chaser)
        self.chaser = chaser
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}

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
        self.info['time in los'] = self.info['time in los'] + self.r_form.time_inlos
        self.info['time in validslowzone'] = self.info['time in validslowzone'] + self.r_form.time_slowzone
        self.info['time in los and slowzone'] = self.info['time in los and slowzone'] + self.r_form.time_inlos_slowzone
        self.info['time in los and phase 3'] = self.info['time in los and phase3'] + self.r_form.time_inlos_phase3
        self.info['episode time'] = self.info['episode time'] + 1
        self.r_form.reset_counts()
        return obs, reward, done, self.info

    def write_data(self, file_name):
        #print('writing data to text')
        #print(f'current step {self.chaser.current_step}')
        write2text(self.chaser, 'runs', file_name, self.chaser.current_step)


    def reset(self):
        self.chaser.reset()
        self.chaser.update_state(self.chaser.state)
        self.info = {'time in los' : 0,
                     'time in validslowzone' : 0,
                     'episode time' : 0,
                     'time in los and slowzone' : 0,
                     'time in los and phase3' : 0}

        print('reseting environment')
        return np.array(self.chaser.state, copy=True)
