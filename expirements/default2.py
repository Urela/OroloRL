import gym
import pybullet_envs
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([[x[2]] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([x[4] for x in batch])
    return states, actions, rewards, nstates, dones


env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
#agent = DQN( env.observation_space, env.action_space )

update_frequency = 500
time_step = 0

#env.render(mode='human')
env.render()
scores = []
for epi in range(1000):
  obs = env.reset()
  while True:
    #action = agent.get_action(obs)
    action = env.action_space.sample()
    _obs, reward, done, info = env.step(action)
    #agent.memory.append((obs, action, reward, _obs, done))
    time_step+=1
    #agent.train()

    obs = _obs
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
