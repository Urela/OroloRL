""" https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py """
import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

class ReplayBuffer:
  def __init__(self, size=buffer_limit):
    self.memory  = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, bath_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch] , dtype=torch.float)
    actions = torch.tensor([[x[1]] for x in batch] ).float()
    rewards = torch.tensor([[x[2]] for x in batch] ).float()
    nstates = torch.tensor([x[3] for x in batch] , dtype=torch.float)
    dones   = torch.tensor([[1-(x[4])] for x in batch]  )
    return states, actions, rewards, nstates, dones



class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, out_space.shape[0])
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
    mu = torch.tanh(self.fc3(x))*2 
    return mu

class Critic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Critic, self).__init__()
    self.fc_obs = nn.Linear(in_space.shape[0], 64)
    self.fc_act = nn.Linear(out_space.shape[0], 64)
    self.fc3 = nn.Linear(128, 32)
    self.fc4 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    z_s = F.relu(self.fc_obs(s))
    z_a = F.relu(self.fc_act(a)) 
    x = torch.cat([z_s,z_a], dim=1)
    x = self.fc3(x)
    x = self.fc4(x)
    return x

class OrnsteinUhlenbeckNoise:
  def __init__(self, mu):
    self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
    self.mu = mu
    self.x_prev = np.zeros_like(mu)

  def __call__(self):
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
        self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x


class DDPG:
  def __init__(self, in_space, out_space):
    self.memory = ReplayBuffer()
    self.actor  = Actor(in_space, out_space)
    self.critic = Critic(in_space, out_space)
    self.targ_actor  = Actor(in_space, out_space)
    self.targ_critic = Critic(in_space, out_space)

    #self.ou_noise = OrnsteinUhlenbeckNoise(mu=out_space.shape[0])
    self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

  def store(self,exp): self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.from_numpy(obs).float()
    mu  = self.actor(obs) 
    return mu.item() + self.ou_noise()[0]

  def train(self):
    #print(len(self.memory))
    if len(self.memory) >= 2000:
      print( "training" )
      for i in range(10):
        states, actions, rewards, nstates, dones = self.memory.sample(batch_size)

        q = self.critic(states, actions)
        a_targ = self.targ_actor(nstates)
        q_targ = self.targ_critic( nstates, a_targ )
        target = rewards + gamma*q_targ *dones
        crit_loss = F.smooth_l1_loss( q, target.detach() )

        self.critic.optimizer.zero_grad()
        crit_loss.backward()
        self.critic.optimizer.step()

        a_pred = self.actor(states)
        act_loss = - self.critic(states, a_pred).mean()
        self.actor.optimizer.zero_grad()
        act_loss.backward()
        self.actor.optimizer.step()
    pass

  def update_targets(self):
    network, target = self.actor, self.targ_actor
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)
    




env = gym.make('Pendulum-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
#print( env.observation_space, "dddd", env.action_space )
agent = DDPG( env.observation_space, env.action_space )

score = []
for epi in range(10000):
  obs = env.reset()
  while True:

    env.render()
    action = agent.get_action(obs)
    _obs, reward, done, info = env.step([action])

    agent.store((obs, action, reward/100.0, _obs, done))

    obs = _obs
    if "episode" in info.keys():
      #score.append( info['episode']['r'] )
      agent.train()
      print(f"Episode {epi}, Return: {info['episode']['r']}")
      break
  agent.update_targets()

env.close()
