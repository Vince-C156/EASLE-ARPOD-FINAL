import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
"""
Classes:
    -PPOMemory (self, batch_size)
    -Actor (self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, ckpt_dir='tmp/ppo")
    -Critic (self, input_dims, alpha, fc1_dims=256, fc2_dims=256, ckpt_dir='tmp/ppo')
    -Agent (self, n_actions, gamma=0.99, alpha=0.003, 
        policy_clip=0.1, batch_size=64, N=2048, n_epochs=10)
"""

class PPOMemory:
    def __init__(self, batch_size):
        """
        class members tracking information over time
        """
        self.states = []
        self.logprobs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def gen_batches(self):
        """
        shuffling steps from 0-N and returning batches of randomly assorted steps in
        the environment
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        #for i (all starting indicies) create arrays from i to i+batchsize then store
        #array into array. (array of batches)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        #(states, actions, logprobs, rewards, dones, batches
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.logprobs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, logprobs, vals, rewards, done):
        """
        writes (state, action, logprobs, vals, rewards, done) to respected class member
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.vals.append(vals)
        self.rewards.append(rewards)
        self.dones.append(done)


    def reset_memory(self):
        self.states = []
        self.logprobs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []



class Actor(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, ckpt_dir='tmp/ppo'):
        """
        input:
            -n_actions: number of possible control inputs
            -input_dims:
            -alpha: learning rate alpha
            -fc1_dims and fc2_dims: fully connected dimensions for fully connected layer 1 and 2 
        """
        super(Actor, self).__init__()

        dir_path = os.getcwd()
        ckpt_path = os.path.join(dir_path, ckpt_dir, 'actor_ckpts') 

        
        if os.path.exists(ckpt_path) == False:
            print("check point path doesnt exist attempting to create :", ckpt_path)
            os.makedirs(ckpt_path)

        model_num = len(os.listdir(ckpt_path))
        model_name = f"actor_ppo{model_num}"
        self.checkpoint_file = os.path.join(ckpt_path, model_name)

        self.actor_network = nn.Sequential(
            self.layer_init(nn.Linear(input_dims, 130).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(130, 88).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(88, 60).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(60, n_actions).double(), std=0.01)
        )



        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"using cuda: {T.cuda.is_available()}")
        self.to(self.device)
        self.cov_var = T.full(size=(n_actions,), fill_value=0.5, dtype=T.double, device=self.device)
        self.cov_mat = T.diag(self.cov_var).double()

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        T.nn.init.orthogonal_(layer.weight, std).double()
        T.nn.init.constant_(layer.bias, bias_const).double()
        return layer

    def forward(self, state):
        """
        given state, get the distribution of actions by passing state into actor network
        """

        mean = self.actor_network(state).double()
        distribution = MultivariateNormal(mean, self.cov_mat)

        return distribution

    def save_checkpoint(self):
        """
        saves current training checkpoint, no return
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        loads last saved checkpoint
        """

        self.load_state_dict(T.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, ckpt_dir='tmp/ppo'):
        super(Critic, self).__init__()

        dir_path = os.getcwd()
        ckpt_path = os.path.join(dir_path, ckpt_dir, 'critic_ckpts') 


        if os.path.exists(ckpt_path) == False:
            print("check point path doesnt exist attempting to create :", ckpt_path)
            os.makedirs(ckpt_path)

        model_num = len(os.listdir(ckpt_path))
        model_name = f"critic_ppo{model_num}"
        self.checkpoint_file = os.path.join(ckpt_path, model_name)

        self.critic_network = nn.Sequential(
            self.layer_init(nn.Linear(input_dims, 130).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(130, 25).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(25, 5).double()),
            nn.Tanh(),
            self.layer_init(nn.Linear(5, 1).double(), std=1.0) 
        )


        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        T.nn.init.orthogonal_(layer.weight, std).double()
        T.nn.init.constant_(layer.bias, bias_const).double()
        return layer

    def forward(self, state):
        """
        given a state the value of the state is outputed
        """
        value = self.critic_network(state)
        return value

    def save_checkpoint(self):
        """
        saves current training checkpoint, no return
        """
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """
        loads last saved checkpoint
        """

        self.load_state_dict(T.load(self.checkpoint_file))



class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.003, gae_lambda=0.95, 
                 policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(n_actions, input_dims, alpha)
        self.critic = Critic(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        #sending tensor to device (float64)
        state = T.tensor([observation], dtype=T.float64).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample().double()
        action = T.clamp(action, min=-1.0, max=1.0)
        #log_probs = dist.log_prob(action)
        #log_probs = log_probs.cpu().detach().numpy().item()
        #log_probs = T.squeeze(dist.log_prob(action)).item()
        #print('log probs')
        #print(log_probs)
        log_probs_sum = dist.log_prob(action).sum(0).cpu().detach()
        action = action.cpu().detach().numpy()
        action = action[0]
        #action = action.item()
        value = T.squeeze(value).item()

        return action, log_probs_sum, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.gen_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float64)

            for t in range(len(reward_arr) - 1):
                discount = 1 #discount factor
                a_t = 0 #advantage at t (initalizes at 0)
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(done_arr[k])) - values[k])
                    discount *= (self.gamma * self.gae_lambda)
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.double).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                #
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                #back prop loss

                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        #clear memory at end of all epochs
        self.memory.reset_memory()
