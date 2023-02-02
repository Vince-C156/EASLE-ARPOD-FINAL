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
from numba import njit, prange, cuda

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        assert x.shape[1] >= 1, 'invalid obs dimension, must be atleast 2 dimensions'

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

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
        self.r_rms = RunningMeanStd(shape=())
        self.gamma = 0.95
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")


        self.batches_generated = 0
        #self.device = device
        #self.states = T.tensor([]).to(device)
        #self.actions = T.tensor([]).to(device)
        #self.logprobs = T.tensor([]).to(device)
        #self.rewards = T.tensor([]).to(device)
        #self.dones = T.tensor([]).to(device)
        #self.vals = T.tensor([]).to(device)

        self.batch_size = batch_size

    def gen_batches(self):
        """
        shuffling steps from 0-N and returning batches of randomly assorted steps in
        the environment
        """

        """
        calculate scaled rewards (rewards / all_returns_std)
        """


        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)

        #self.batches_generated += 1
        np.random.shuffle(indices)
        #for i (all starting indicies) create arrays from i to i+batchsize then store
        #array into array. (array of batches)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        #(states, actions, logprobs, rewards, dones, batches
        return np.asarray(self.states),\
                np.asarray(self.actions),\
                np.asarray(self.logprobs),\
                np.asarray(self.vals),\
                np.asarray(self.rewards),\
                np.asarray(self.dones),\
                batches

    def get_scaled_rewards(self):
        """
        calculate scaled rewards (rewards / all_returns_std)
        """
        @njit(parallel=True, cache=True)
        def calc_returns(rewards, vals, dones, gamma):
            returns = np.zeros(len(rewards), dtype=np.float64)
            #T = list(range(14400, 0, -1))
            for ts in prange(len(rewards) - 1):
                t = len(rewards) - ts
                #replacing reward at last step with value estimate
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    next_return = 1.0
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            return returns

        reward_arr = np.asarray(self.rewards)
        done_arr = np.asarray(self.dones)
        values = np.asarray(self.vals) 

        returns = calc_returns(reward_arr, values, done_arr, self.gamma)
        returns = np.atleast_2d(returns)
        self.r_rms.update(returns)

        #reward clipped between -10 and 10
        #scaled_rew = np.clip( reward_arr / np.sqrt(self.r_rms.var + 1e-4) , -10.0, 10.0)
        scaled_rew = reward_arr / np.sqrt(self.r_rms.var + 1e-4)
        return np.asarray(scaled_rew)

    def store_memory(self, state, action, logprobs, vals, rewards, done):
        """
        writes (state, action, logprobs, vals, rewards, done) to respected class member
        """
        #self.states = T.cat((self.states, state))
        #self.actions = T.cat((self.actions, action))
        #self.logprobs = T.cat((self.logprobs, logprobs))
        #self.vals = T.cat((self.vals, vals))
        #self.rewards = T.cat((self.rewards, rewards))
        #self.dones = T.cat((self.dones, done))

        self.states.append(state)
        self.actions.append(np.squeeze(action))
        self.logprobs.append(np.squeeze(logprobs))
        self.vals.append(np.squeeze(vals))
        self.rewards.append(np.squeeze(rewards))
        self.dones.append(done)


    def reset_memory(self):
        self.states = []
        self.logprobs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batches_generated = 0
        #self.r_rms = RunningMeanStd(shape=())


class Agent(nn.Module):

    def __init__(self, ckpt_dir='tmp/ppo'):
        super(Agent, self).__init__()

        dir_path = os.getcwd()
        ckpt_path = os.path.join(dir_path, ckpt_dir, 'ppo_ckpt')
        if os.path.exists(ckpt_path) == False:
            print("check point path doesnt exist attempting to create :", ckpt_path)
            os.makedirs(ckpt_path)

        model_num = len(os.listdir(ckpt_path))
        model_name = f"ppo{model_num}"
        self.checkpoint_file = os.path.join(ckpt_path, model_name)
        log_id = len(os.listdir('logs'))
        self.writer = SummaryWriter(f"logs/ppo{log_id}")


        self.action_dims = (3,)

        self.input_dims = (6,)
        self.obs_rms = RunningMeanStd(shape=self.input_dims)

        self.use_gae = False
        self.gae_lambda = 0.95

        self.batch_size = 800

        self.gamma = 0.95
        self.policy_clip = 1.0
        self.clip_vloss = True
        self.epochs = 15
        self.entropy_coef = 0.02 #[0.1, 0.01]
        #LR scheduler parameters
        lr_actor = 3e-4
        actor_lrmin = 3.125e-6
        lr_critic = 5e-4
        critic_lrmin = 1.5625e-5
        annealing_time = 20000


        #episode counter
        self.episode = 0

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        """
        Actor and critic networks
        """
        self.actor_network = nn.Sequential(
            self.layer_init(nn.Linear(*self.input_dims, 128).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(128, 88).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(88,60).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(60, np.prod(self.action_dims)).double(), std=0.01),
        )

        self.actor_logstd = nn.Parameter(T.zeros(1, np.prod(self.action_dims)).double() )


        self.critic_network = nn.Sequential(
            self.layer_init(nn.Linear(*self.input_dims, 256).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(256, 150).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(150, 50).double()),
            nn.LeakyReLU().double(),
            self.layer_init(nn.Linear(50,1).double(), std=1.0)
        )

        self.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr_critic, eps=1e-5)

        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max = annealing_time, eta_min = actor_lrmin)
        self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max = annealing_time, eta_min = critic_lrmin)

        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.memory = PPOMemory(self.batch_size)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        T.nn.init.orthogonal_(layer.weight, std).double()
        T.nn.init.constant_(layer.bias, bias_const).double()
        return layer

    def save_checkpoint(self):
        """
        saves current training checkpoint, no return
        """
        T.save(self.state_dict(), self.checkpoint_file)


    def load_checkpoint_fromfile(self, ckpt_file):
        """
        loads last saved checkpoint file

        sets file of this model to that aswell
        """

        print(f'loading from {ckpt_file}')
        print(f'saving new checkpoints in {self.checkpoint_file}')
        #self.checkpoint_file = ckpt_file
        self.load_state_dict(T.load(ckpt_file))


    def load_checkpoint(self):
        """
        loads last saved checkpoint
        """
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_models(self):
        print('... saving model ...')
        self.save_checkpoint()
        #self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.load_checkpoint()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def get_value(self, x, rms_update=False):
        """
        do not input more than one observation for this function

        input: a numpy array representing the state either [x y z xdot ydot zdot] or [[x y z xdot ydot zdot]]

        output: a value representing the estiamted reward for the input state.
        """
        x = np.atleast_2d(x)
        if rms_update == True:
            self.obs_rms.update(x)
        x = np.clip((x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-4), -10.0, 10.0)
        x = T.tensor(x, dtype=T.float64).to(self.device)
        return T.squeeze(self.critic_network(x)).item()

    def get_actions_and_value(self, x, action=None, rms_update=False):
        x = np.atleast_2d(x)
        if rms_update == True:
            self.obs_rms.update(x)
        x = np.clip((x - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-4), -10.0, 10.0)
        self.writer.add_scalar("norms/state_rms x", x[-1][0], self.episode)
        self.writer.add_scalar("norms/state_rms y", x[-1][1], self.episode)
        self.writer.add_scalar("norms/state_rms z", x[-1][2], self.episode)

        self.writer.add_scalar("norms/state_rms xdot", x[-1][3], self.episode)
        self.writer.add_scalar("norms/state_rms ydot", x[-1][4], self.episode)
        self.writer.add_scalar("norms/state_rms zdot", x[-1][5], self.episode)

        state = T.tensor(x, dtype=T.float64).to(self.device)
        #print(f'normalized state {state}')
        #print(f'state mean {self.obs_rms.mean}')
        #print(f'state var {self.obs_rms.var}')
        #state = T.atleast_2d(state)

        act_means = self.actor_network(state)
        act_logstds = self.actor_logstd.expand_as(act_means)
        action_stds = T.exp(act_logstds)
        #logp_u(u) = logp_params(x) - logtanh'(x) where x = tanh^-1(x)

        policy_distributions = Normal(act_means, action_stds)

        if action is None:
            action = policy_distributions.sample()

        return action.detach().cpu().numpy(), policy_distributions.log_prob(action).sum(1).detach().cpu().numpy(), policy_distributions.entropy().sum(1), self.critic_network(state).detach().cpu().numpy(), act_logstds


    def learn(self):

        @njit(parallel=True, cache=True)
        def calc_advantage(rewards, vals, dones, gamma, next_value):
            returns = np.zeros(len(rewards), dtype=np.float64)
            #T = list(range(14400, 0, -1))
            for ts in prange(len(rewards) - 1):
                t = len(rewards) - ts
                #replacing reward at last step with value estimate
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones[t]
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - vals
            return advantages, returns

        self.episode += 1

        #calculating clipped scaled reward based on discounted sum of rewards for whole episode
        #scaled_rews = self.memory.get_scaled_rewards()


        for _ in range(self.epochs):
            with T.no_grad():
                #calculate advantage and returns
                state_arr, action_arr, old_probs_arr, vals_arr, reward_arr, done_arr, batches = self.memory.gen_batches()
                values = vals_arr
                """
                calculating advantage and new returns over the whole episode using clipped normalized rewards
                """
                scaled_rews = self.memory.get_scaled_rewards()
                last_state_value = self.get_value(state_arr.flatten()[-1])
                advantages_arr, returns_arr = calc_advantage(scaled_rews, values, done_arr, self.gamma, last_state_value)

            #advantages_arr = T.tensor(advantages).to(self.device)
            #returns_arr = T.tensor(returns).to(self.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.double).to(self.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                advantages = T.tensor(advantages_arr[batch]).to(self.device)
                returns = T.tensor(returns_arr[batch]).to(self.device)
                bvals = T.tensor(values[batch]).to(self.device)

                _, newlogprobs, entropy, newvalues, logstds = self.get_actions_and_value(states.detach().cpu().numpy())

                newvalues = T.tensor(newvalues, dtype=T.float64).to(self.device)
                logratio = T.tensor(newlogprobs).to(self.device) - old_probs
                ratio = logratio.exp()
                clipfracs = []
                with T.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.policy_clip).float().mean().item()]


                #advantage normalization
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * T.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                pg_loss = T.max(pg_loss1, pg_loss2).mean()


                # Value loss
                newvalues = newvalues.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalues - returns) ** 2
                    v_clipped = bvals + T.clamp(
                        newvalues - bvals,
                        -self.policy_clip,
                        self.policy_clip,
                    )
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss_max = T.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalues - returns) ** 2).mean()

                entropy_loss = entropy.mean()
                #entropy_loss = 0.5 * T.sum(logstds + T.tensor(np.log(2*np.pi *np.e)))

                loss = pg_loss - self.entropy_coef*entropy_loss + 0.5*v_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                loss.backward()
                #clipping params for both networks
                #nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                #self.entropy_coef*entropy_loss
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
        self.writer.add_scalar("charts/actor_learning_rate", self.actor_scheduler.get_last_lr()[0], self.episode)
        self.writer.add_scalar("charts/actor_learning_rate", self.critic_scheduler.get_last_lr()[0], self.episode)
        self.writer.add_scalar("losses/critic_loss", v_loss, self.episode)
        self.writer.add_scalar("losses/actor_loss", pg_loss, self.episode)
        self.writer.add_scalar("losses/ppo loss", loss, self.episode)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.episode)
        self.writer.add_scalar("simulation/mean x", self.obs_rms.mean[0], self.episode)
        self.writer.add_scalar("simulation/var x", self.obs_rms.var[0] , self.episode)
        self.writer.add_scalar("simulation/mean y", self.obs_rms.mean[1], self.episode)
        self.writer.add_scalar("simulation/var y", self.obs_rms.var[1] , self.episode)
        self.writer.add_scalar("simulation/mean z", self.obs_rms.mean[2], self.episode)
        self.writer.add_scalar("simulation/var z", self.obs_rms.var[2] , self.episode)
        #self.writer.add_scalar("simulation/returns var", self.memory.r_rms.var[:,-1], self.episode)
        #self.writer.add_scalar("simulation/returns mean", self.memory.r_rms.mean[:,-1], self.episode)

        print(f'actor LR {self.actor_scheduler.get_last_lr()}')
        print(f'critic LR {self.critic_scheduler.get_last_lr()}')
        #clear memory at end of all epochs
        self.memory.reset_memory()
        #self.obs_rms = RunningMeanStd(shape=self.input_dims)


