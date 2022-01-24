import gym
import pybullet_envs
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([[x[2]] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([int(x[4]) for x in batch])
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc_mean = nn.Linear(64, out_space.shape[0])
    self.fc_std  = nn.Linear(64, out_space.shape[0])

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.reparam_noise = 1e-6
    self.max_action=out_space.high

  def forward(self, x, reparmi=False):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    mean = self.fc_mean(x)
    std = self.fc_std(x)
    # we want to bound our distbuitons so that they are not massive
    std = torch.clamp(std, min=self.reparam_noise, max=1) 

    dist = Normal(mean, std)
    actions = dist.sample().to('cpu') if reparmi else dist.rsample().to('cpu')

    action = torch.tanh(actions)*torch.tensor(self.max_action).to('cpu')
    log_probs = dist.log_prob(actions)
    log_probs -= torch.log(1-torch.tanh(action).pow(2) + self.reparam_noise)
    #log_probs = log_probs.sum(1, keepdim=True)           # ????? TODO
    return action, log_probs

class Critic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.shape[0], 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s,a], dim=1)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

class Value(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Value, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 128)
    self.fc2 = nn.Linear(128, 32)
    self.fc3 = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

class SAC:
  def __init__(self, in_space, out_space):
    self.tau = 0.005
    self.gamma = 0.99
    self.scale = 2 
    self.memory = ReplayBuffer()

    self.actor   = Actor(in_space, out_space).to('cpu')
    self.critic1 = Critic(in_space, out_space).to('cpu')
    self.critic2 = Critic(in_space, out_space).to('cpu')
    self.value      = Value(in_space, out_space).to('cpu')
    self.targ_value = Value(in_space, out_space).to('cpu')

  def store(self,exp): 
    self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.tensor(obs).float()
    action, log_prob = self.actor(obs)
    return action.detach().numpy(), log_prob 

  def train(self, batch_size=256 ):
    if len(self.memory) >= batch_size:
      #print( "training" )
      states, actions, rewards, nstates, dones = self.memory.sample(batch_size)

      value  = self.value(states).view(-1)
      nvalue = self.targ_value(nstates).view(-1)
      nvalue[dones] = 0.01

      # update value network
      _actions, log_probs = self.actor(states)
      log_probs = log_probs.view(-1)
      q1 = self.critic1.forward(states, _actions)
      q2 = self.critic2.forward(states, _actions)
      critic = torch.min(q1 , q2).view(-1)

      self.value.optimizer.zero_grad()
      target = critic - log_probs
      loss = 0.5 * F.mse_loss(value, target)
      loss.backward(retain_graph=True)
      self.value.optimizer.step()

      # update policy network
      _actions, log_probs = self.actor(states, True)
      log_probs = log_probs.view(-1)
      q1 = self.critic1.forward(states, _actions)
      q2 = self.critic2.forward(states, _actions)
      critic = torch.min(q1 , q2).view(-1)

      loss = log_probs - critic
      loss = torch.mean(loss)
      self.actor.optimizer.zero_grad()
      loss.backward(retain_graph=True)
      self.actor.optimizer.step()

      #update critics
      self.critic1.optimizer.zero_grad()
      self.critic2.optimizer.zero_grad()

      #print( states.shape, actions.shape)
      q_hat = self.scale*rewards + self.gamma*nvalue
      q1 = self.critic1.forward(states, actions).view(-1)
      q2 = self.critic2.forward(states, actions).view(-1)
      c1_loss = 0.5 * F.mse_loss(q1, q_hat)
      c2_loss = 0.5 * F.mse_loss(q2, q_hat)

      loss = c1_loss + c2_loss
      loss.backward()
      self.critic1.optimizer.step()
      self.critic2.optimizer.step()

      self.update_network_parameters()
    pass

  def update_network_parameters(self, tau=None):
    if tau is None:
      tau = self.tau

    target_value_params = self.targ_value.named_parameters()
    value_params = self.value.named_parameters()

    target_value_state_dict = dict(target_value_params)
    value_state_dict = dict(value_params)

    for name in value_state_dict:
      value_state_dict[name] = tau*value_state_dict[name].clone() + \
        (1-tau)*target_value_state_dict[name].clone()
    self.targ_value.load_state_dict(value_state_dict)

#env = gym.make('Pendulum-v0')
env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = SAC( env.observation_space, env.action_space )

update_frequency = 500
time_step = 0

#env.render(mode='human')
#env.render()
scores = []
for epi in range(1000):
  obs = env.reset()
  while True:
    #env.render()
    action, log_prob = agent.get_action(obs)
    _obs, reward, done, info = env.step(action)
    agent.store((obs, action, reward, _obs, done))
    time_step+=1
    agent.train()

    obs = _obs
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
