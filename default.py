import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
batch_size   = 32
buffer_limit = 50000

class ReplayBuffer:
  def __init__(self, size=buffer_limit):
    self.memory  = collections.deque(maxlen=size)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, bath_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch] , dtype=torch.float)
    actions = torch.tensor([[x[1]] for x in batch] , dtype=torch.int64)
    rewards = torch.tensor([x[2] for x in batch] , dtype=torch.float)
    nstates = torch.tensor([x[3] for x in batch] , dtype=torch.float)
    dones   = torch.tensor([1-(x[4]) for x in batch]  )
    return states, actions, rewards, nstates, dones



env = gym.make('Pendulum-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
memory = ReplayBuffer()

scores = []
for epi in range(100):
  obs = env.reset()
  while True:

    action = env.action_space.sample()
    _obs, reward, done, info = env.step([action])

    memory.store((obs, action, reward/100.0, _obs, done))

    obs = _obs
    if "episode" in info.keys():
      scores.append( info['episode']['r'] )
      print(f"Episode {epi}, Return: {info['episode']['r']}")
      break

env.close()

y = scores 
x = np.arange(len(y))

from bokeh.plotting import figure, show
p = figure(title="TODO", x_axis_label="Episodes", y_axis_label="Scores")
p.line(x, y,  legend_label="Scores", line_color="blue", line_width=2)
show(p) 
