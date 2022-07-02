import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = 'cpu'
gamma = 0.99
tau   = 0.005 # for target network soft update

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.rewards = np.zeros(length)
    self.nstates = np.zeros((length, obs_dim))
    self.dones   = np.zeros(length, dtype=bool)
    self.size = length
    self.idx  = 0

  def __len__(self): return self.idx

  def store(self, obs, action, reward, next_obs, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.nstates[idx] = next_obs
    self.dones[idx] = done

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(device)
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).to(device)
    nstates = torch.tensor( self.nstates[indices], dtype=torch.float).to(device)
    dones   = torch.tensor( self.dones[indices] ).float()
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_dims, out_dims, lr=5e-4):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_dims, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, out_dims)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = torch.tanh(self.fc3(x))
    return x

class Critic(nn.Module):
  def __init__(self, in_dims, out_dims, lr=1e-3):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_dims+out_dims, 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, o, a):
    x = torch.cat([o, a], dim=1)
    #x = torch.cat(o+a, dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class DDPG():
  def __init__(self, in_dims, out_dims):
    self.out_dims = out_dims
    self.memory = ReplayBuffer(in_dims, out_dims)

    self.actor  = Actor(in_dims,  out_dims).to(device)
    self.critic = Critic(in_dims, out_dims).to(device)
    self.targ_actor  = Actor(in_dims,  out_dims).to(device) # target critic
    self.targ_critic = Critic(in_dims, out_dims).to(device) # target actor

    # intialize the targets to match their networks
    self.targ_critic.load_state_dict(self.critic.state_dict())
    self.targ_actor.load_state_dict(self.actor.state_dict())

  def store(self, obs,action,reward,_obs,done):
    self.memory.store(obs,action,reward,_obs,done)

  def get_action(self, obs):
    action = self.actor(torch.from_numpy(obs).float()) 
    # DDPG is a deterministic, so in order to explore we need use
    # epsilon greedy or we can add noise the makes the policy stochasic
    # https://soeren-kirchner.medium.com/deep-deterministic-policy-gradient-ddpg-with-and-without-ornstein-uhlenbeck-process-e6d272adfc3
    # Normal distribution is easier to implement than OrnsteinUhlenbeckNoise
    normal_scalar = .25
    action = action.item() + np.random.randn(self.out_dims) * normal_scalar
    return action.item()


  def train(self):
    if(len(self.memory) >= 2000):
      for i in range(10):
        states, actions, rewards, nstates, dones = self.memory.sample(32)
        q = self.critic(states, actions).squeeze()
        a_targ = self.targ_actor(nstates)
        q_targ = self.targ_critic(nstates, a_targ).squeeze()
        q_targ = rewards + gamma*q_targ * (1-dones)
        critic_loss = F.smooth_l1_loss(q, q_targ )

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        a_pred = self.actor(states)
        actor_loss = -self.critic(states, a_pred).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
      
    self.update_targets()

  def update_targets(self):
    # soft update https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    network, target = self.actor, self.targ_actor
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)

    network, target = self.critic, self.targ_critic
    for param_targ, param in zip(target.parameters(), network.parameters()):
      param_targ.data.copy_(param_targ.data * (1.0 - tau) + param.data * tau)


env = gym.make('Pendulum-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0])

scores = []
for epi in range(100):
  obs = env.reset()
  while True:

    action = agent.get_action(obs)
    _obs, reward, done, info = env.step([action])
    agent.store(obs,action,reward,_obs,done)
    obs = _obs
    agent.train()
  
    if "episode" in info.keys():
      scores.append(info['episode']['r'])
      #avg_scores = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {info['episode']['r']}")
      break
env.close()

y = scores 
x = np.arange(len(y))
from bokeh.plotting import figure, show
p = figure(title="TODO", x_axis_label="Episodes", y_axis_label="Scores")
p.line(x, y,  legend_label="Scores", line_color="blue", line_width=2)
show(p) 
