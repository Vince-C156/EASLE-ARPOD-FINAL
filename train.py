import numpy as np
from PPO2 import Agent
from environment import ARPOD
from dynamics import chaser_discrete
import numpy as np
from time import sleep
import torch as T
"""
initilaize dynamics and environment
"""

chaser = chaser_discrete(True, False)
env = ARPOD(chaser)

#rbar_chaser = chaser_discrete(False, True)
#env_rbar = ARPOD(rbar_chaser)

"""
update iterval 
"""
N = 3800
batch_size = 1600
n_epochs = 3
alpha = 0.004

"""
init agent
"""
n_actions = (3,)
obs_shape = (6,)
agent = Agent()
#agent.load_checkpoint_fromfile('tmp/ppo/ppo_ckpt/ppo4')

n_games = 200000
best_score = 0
score_history = []


learn_iters = 0
avg_score = 0
n_steps = 0
"""
initalizations = 10
print('initalizing rms')
print('------------------')
for n in range(initalizations):
    obs = env.reset()
    done = False

    while not done:
        action, prob, entropy, val, _ = agent.get_actions_and_value(obs, action=None, rms_update=True)
        clipped_action = np.clip(action, -10.0, 10.0)
        #print(clipped_action)
        obs_, reward, done, info = env.step(clipped_action)
"""
print('starting training')
print('--------------------')


for i in range(n_games):
    obs = env.reset()
    done = False
    data_file_name = f'vbar1/chaser{i}.txt'
    #accumilative score
    score = 0
    episode_step = 0
    Ux = []
    Uy = []
    Uz = []
    while not done:
        action, prob, entropy, val, _ = agent.get_actions_and_value(obs, action=None, rms_update=True)
        Ux.append(action[0][0])
        Uy.append(action[0][1])
        Uz.append(action[0][2])
        clipped_action = np.clip(action, -10.0, 10.0)
        #clipped_action += 10.0
        #print(f'raw action {action}')
        #action *= 0.004
        #normalized_action = action / np.linalg.norm(action) #[-1, 1]
        #scalars = np.array([0.005, 0.005, 0.005], dtype=np.float64)
        #print(f'normalized_action {normalized_action}')
        #scaled_action = clipped_action
        #print(f'clipped_action {clipped_action}')
        #print('------------------------')
        obs_, reward, done, info = env.step(clipped_action)
        env.write_data(data_file_name)
        """
        print(f'state from action {obs_}')
        print('-------------------------')
        print(f'reward {reward}')
        print(f'value {val}')
        print(f'action {normalized_action}')
        print('------------------------')
        """
        n_steps += 1
        score += reward
        agent.remember(obs, action, prob, val, reward, done)
        episode_step += 1
        """
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        """
        obs = obs_

    print(f'vbar info {info}')
    print('-------------------')
    print('running rbar')
    done = False
    obs = env.reset()
    data_file2_name = f'vbar2/chaser{i}.txt'

    while not done:
        action, prob, entropy, val, _ = agent.get_actions_and_value(obs, action=None, rms_update=True)
        Ux.append(action[0][0])
        Uy.append(action[0][1])
        Uz.append(action[0][2])
        clipped_action = np.clip(action, -10.0, 10.0)
        #clipped_action += 10.0
        #print(f'raw action {action}')
        #action *= 0.004
        #normalized_action = action / np.linalg.norm(action) #[-1, 1]
        #scalars = np.array([0.005, 0.005, 0.005], dtype=np.float64)
        #print(f'normalized_action {normalized_action}')
        #scaled_action = clipped_action
        #print(f'clipped_action {clipped_action}')
        #print('------------------------')
        obs_, reward, done, info = env.step(clipped_action)
        env.write_data(data_file2_name)
        """
        print(f'state from action {obs_}')
        print('-------------------------')
        print(f'reward {reward}')
        print(f'value {val}')
        print(f'action {normalized_action}')
        print('------------------------')
        """
        n_steps += 1
        score += reward
        agent.remember(obs, action, prob, val, reward, done)
        episode_step += 1
        """
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        """
        obs = obs_

    print(f'r_bar info {info}')
    print(f'learning from combined rollout')
    agent.learn()
    score_history.append(score)
    agent.writer.add_scalar("environment_metrics/reward", score, i+1)
    agent.writer.add_scalar("environment_metrics/steps_in_los", info['time in los'], i+1)
    #agent.writer.add_scalar("environment_metrics/steps_met_slowzone_conditions", info['time in validslowzone'], i+1)
    agent.writer.add_scalar("environment_metrics/episode length", info['episode time'], i+1)

    Ux, Uz, Uy = np.asarray(Ux), np.asarray(Uz), np.asarray(Uy)
    agent.writer.add_scalar("simulation/mean u(x)", np.mean(Ux), i+1)
    agent.writer.add_scalar("simulation/mean u(y)", np.mean(Uy), i+1)
    agent.writer.add_scalar("simulation/mean u(z)", np.mean(Uz), i+1)

    avg_score = np.mean(score_history[-100:])


    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    if learn_iters <= 1:
        agent.save_models()
    learn_iters += 1

    print("==================")
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
    print("=================")
    #sleep(0.5)
agent.writer.close()
