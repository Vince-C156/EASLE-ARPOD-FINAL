import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
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
        #np.random.shuffle(indices)
        #for i (all starting indicies) create arrays from i to i+batchsize then store
        #array into array. (array of batches)
        #batches = [indices[i:i+self.batch_size] for i in batch_start]

        #(states, actions, logprobs, rewards, dones, batches
        return np.asarray(self.states),\
                np.asarray(self.actions),\
                np.asarray(self.logprobs),\
                np.asarray(self.vals),\
                np.asarray(self.rewards),\
                np.asarray(self.dones) #batches

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

        self.actor_logstd = nn.Parameter(T.zeros(1, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f"using cuda: {T.cuda.is_available()}")
        self.to(self.device)
        #self.cov_var = T.full(size=(n_actions,), fill_value=1.0, dtype=T.double, device=self.device)
        #self.cov_mat = T.diag(self.cov_var).double()

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        T.nn.init.orthogonal_(layer.weight, std).double()
        T.nn.init.constant_(layer.bias, bias_const).double()
        return layer

    def forward(self, state):
        """
        given state, get the distribution of actions by passing state into actor network
        """

        mean = self.actor_network(state).double()
        action_logstd = self.actor_logstd.expand_as(mean)
        action_std = T.exp(action_logstd)
        probs = Normal(mean, action_std)
        #distribution = MultivariateNormal(mean, self.cov_mat)

        return probs

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
        if os.path.exists('logs') == False:
            os.mkdir('logs')
        log_id = len(os.listdir('logs'))
        self.writer = SummaryWriter(f"logs/ppo{log_id}")
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = Actor(n_actions, input_dims, alpha)
        self.critic = Critic(input_dims, alpha)
        self.actor_scheduler = CosineAnnealingLR(self.actor.optimizer, T_max = 200000, eta_min = 1e-6)
        self.critic_scheduler = CosineAnnealingLR(self.critic.optimizer, T_max = 200000, eta_min = 1e-6)
        self.memory = PPOMemory(batch_size)

        self.episode = 0

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
        probs = self.actor(state)
        value = self.critic(state)
        action = probs.sample()

        log_probs = probs.log_prob(action).sum(1).cpu().detach().numpy()
        log_probs = log_probs[0]

        action = action.cpu().detach().numpy()
        action = action[0]

        value = T.squeeze(value).item()

        return action, log_probs, value

    def learn(self):
        self.episode += 1
        for _ in range(self.n_epochs):
            #batches
            state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr = self.memory.gen_batches()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float64)

            for t in range(len(reward_arr) - 1):
                discount = 0.9 #discount factor
                a_t = 0 #advantage at t (initalizes at 0)
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(done_arr[k])) - values[k])
                    discount *= (self.gamma * self.gae_lambda)
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)

            states = T.tensor(state_arr, dtype=T.double).to(self.actor.device)
            old_probs = T.tensor(old_probs_arr).to(self.actor.device)
            actions = T.tensor(action_arr).to(self.actor.device)

            dist = self.actor(states)
            entropy = dist.entropy()
            #print(f'entropy {entropy}')
            entropy_loss = entropy.mean()
            critic_value = self.critic(states)

            critic_value = T.squeeze(critic_value)
            #print(f'critic value {critic_value}')
            #print(f'log probs {dist.log_prob(actions).sum(1)}')
            new_probs = dist.log_prob(actions).sum(1)
            #print(f'new probs {new_probs}')
            #print(f'old probs {old_probs}')
            #prob_ratio = new_probs.exp() / old_probs.exp()
            log_ratio = new_probs - old_probs
            prob_ratio = log_ratio.exp()
            #normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            #
            weighted_probs = advantage * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            
            returns = advantage + values
            critic_loss = (returns-critic_value)**2.0
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.01*entropy_loss + 0.5*critic_loss
            #print(f'advantage {advantage}')
            print(f'actor loss {actor_loss}')
            print(f'entropy loss {0.01*entropy_loss}')
            print(f'critic loss {0.5*critic_loss}')
            print(f'total loss {total_loss}')
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            #back prop loss

            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()

            """
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.double).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                entropy = dist.entropy()
                entropy_loss = entropy.mean()
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions).sum(1)
                prob_ratio = new_probs.exp() / old_probs.exp()

                #
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss - 0.01*entropy_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                #back prop loss

                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
            """
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.writer.add_scalar("charts/actor_learning_rate", self.actor_scheduler.get_last_lr()[0], self.episode)
        self.writer.add_scalar("charts/actor_learning_rate", self.critic_scheduler.get_last_lr()[0], self.episode)
        self.writer.add_scalar("losses/critic_loss", critic_loss, self.episode)
        self.writer.add_scalar("losses/actor_loss", actor_loss, self.episode)
        self.writer.add_scalar("losses/ppo loss", total_loss, self.episode)
        self.writer.add_scalar("losses/entropy", entropy_loss, self.episode)
        print(f'actor LR {self.actor_scheduler.get_last_lr()}')
        print(f'critic LR {self.critic_scheduler.get_last_lr()}')
        #clear memory at end of all epochs
        self.memory.reset_memory()
