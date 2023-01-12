import numpy as np
from PPO import Agent
from environment import ARPOD
from dynamics import chaser_discrete
import numpy as np
from time import sleep

"""
initilaize dynamics and environment
"""

chaser = chaser_discrete(True, False)
env = ARPOD(chaser)

"""
update iterval 
"""
N = 3800
batch_size = 100
n_epochs = 1
alpha = 0.0005

"""
init agent
"""
n_actions = 3
obs_shape = 6
agent = Agent(n_actions=n_actions, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=obs_shape)


n_games = 200000
best_score = 0
score_history = []


learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    obs = env.reset()
    done = False
    data_file_name = f'chaser{i}.txt'
    #accumilative score
    score = 0
    while not done:
        action, prob, val = agent.choose_action(obs)
        #action *= 0.004
        normalized_action = action / np.linalg.norm(action) #[-1, 1]
        scaled_action = action*10.0
        #print('------------------------')
        obs_, reward, done, info = env.step(scaled_action)
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
        """
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        """
        obs = obs_
    print(info)
    agent.learn()
    learn_iters += 1
    score_history.append(score)
    agent.writer.add_scalar("environment_metrics/reward", score, i+1)
    agent.writer.add_scalar("environment_metrics/steps_in_los", info['time in los'], i+1)
    #agent.writer.add_scalar("environment_metrics/steps_met_slowzone_conditions", info['time in validslowzone'], i+1)
    agent.writer.add_scalar("environment_metrics/episode length", info['episode time'], i+1)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()
    print("==================")
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
    print("=================")
    #sleep(0.5)
agent.writer.close()
