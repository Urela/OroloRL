import gym
import random
import pybullet_envs
import collections

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

'''
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
'''

class ActorCritic(nn.Module):
  def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
  def __init__(self, in_space, out_space, lr=0.0005):
    super(ActorCritic, self).__init__()
    self.fc_actor = nn.Sequential(
      self.layer_init(nn.Linear(np.array(in_space.shape[0]).prod(), 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, out_space.shape[0]), std=0.01),
      nn.Softmax(dim=-1)
    )

    self.fc_critic = nn.Sequential(
      self.layer_init(nn.Linear(np.array(in_space.shape[0]).prod(), 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 1), std=1.0),
    )

    self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(out_space.shape))) 
    self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

  def actor(self, x):
    mean = self.fc_actor(x)
    return mean

  def critic(self, x):
    value = self.fc_critic(x)
    return value

class PPO:  
  def __init__(self, in_space, out_space, batch_size=4, num_steps=128, num_envs=4):
    self.lr    = 2.5e-4   # 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2

    self.AC = ActorCritic(in_space, out_space, self.lr)

    self.size, self.bsize, self.idx = num_steps, batch_size, 0
    self.states  = np.zeros((num_steps, num_envs)+in_space.shape,  dtype=np.float32)
    self.actions = np.zeros((num_steps, num_envs)+out_space.shape, dtype=np.int32)
    self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
    self.probs   = np.zeros((num_steps, num_envs), dtype=np.float32)
    self.values  = np.zeros((num_steps, num_envs), dtype=np.float32)
    self.dones   = np.zeros((num_steps, num_envs), dtype=np.int32)

  def store(self, state, action, reward, probs, vals, done):
    idx = self.idx % self.size
    self.idx += 1
    self.states  [idx] = state
    self.actions [idx] = action
    self.rewards [idx] = reward
    self.probs   [idx] = probs
    self.values  [idx] = vals
    self.dones   [idx] = 1-(done)

  def get_action(self, obs):
    obs   = torch.from_numpy(obs).float().to('cpu')
    value = self.AC.critic(obs)
    mean  = self.AC.actor(obs)
    log_std = self.AC.actor_logstd.expand_as(mean)
    std = log_std.exp()

    dist = Normal(mean, std)
    action = dist.sample()

    probs  = torch.squeeze(dist.log_prob(action)).detach().numpy().sum(1)
    action = torch.squeeze(action).detach().numpy()
    value  = torch.squeeze(value).detach().numpy()
    return action, probs, value

  def train(self):
    self.idx=0
    pass

time_step = 0
num_envs  = 4       # number of parallel environments
env = gym.vector.SyncVectorEnv(
    [lambda: gym.make('HopperBulletEnv-v0') for _ in range(num_envs) ]
)
env = gym.wrappers.RecordEpisodeStatistics(env)

#agent = PPO( env.observation_space, env.action_space )
#agent = PPO(env.single_observation_space, env.single_action_space, )
agent = PPO(env.single_observation_space, env.single_action_space, batch_size=4, num_steps=128, num_envs=num_envs)

#env.render(mode='human')
#env.render() scores = []
scores = []
for epi in range(1000):
  obs = env.reset()
  while True:
    action, probs, vals = agent.get_action(obs)
    #action, probs, vals = env.action_space.sample(), 0 ,0
    _obs, reward, done, info = env.step(action)
    agent.store(obs, action, reward, probs, vals, done)

    time_step+=1
    #agent.train()

    obs = _obs
    for item in info:
      if "episode" in item.keys():
        scores.append(int(item['episode']['r']))
        avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
        print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
        break
