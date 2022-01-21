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

  def sample(self, batch_size=32):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
    actions = torch.tensor([x[1] for x in batch]).float()
    rewards = torch.tensor([x[2] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([1-int(x[4]) for x in batch])
    return states, actions, rewards, nstates, dones

# Initialize Policy weights
def layer_init(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)

""" Implementing a diagonal gaussian policy - Denis Yates"""
class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 256) # Better result with slightly wider networks.
    self.fc2 = nn.Linear(256, 128)
    self.fc_mean = nn.Linear(128, out_space.shape[0])
    self.fc_std  = nn.Linear(128, out_space.shape[0])
    self.LOG_STD_MAX, self.LOG_STD_MIN = -5, 2

    # action rescaling
    self.action_scale = torch.FloatTensor((out_space.high - out_space.low)/2.)
    self.action_bias  = torch.FloatTensor((out_space.high + out_space.low)/2.)

    self.apply(layer_init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    mean = self.fc_mean(x)
    log_std = self.fc_std(x)
    log_std = torch.tanh(log_std)
    log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
    return mean, log_std

  def evaluate(self, obs):
    mean, log_std = self.forward(obs)
    std = log_std.exp()
    dist = Normal(mean, std)
    x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)
    action = y_t * self.action_scale + self.action_bias
    log_probs = dist.log_prob(x_t)
    #print( self.action_scale.shape, y_t.shape )
    log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
    log_probs = log_probs.sum(1, keepdim=True)
    mean = torch.tanh(mean) * self.action_scale + self.action_bias
    return action, log_probs

class Critic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.shape[0], 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
  def forward(self, s, a):
    x = torch.cat([s,a], dim=1)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

learning_start = 5e3
class SAC:
  def __init__(self, in_space, out_space):
    self.tau = 0.005
    self.gamma = 0.99
    self.memory = ReplayBuffer()

    self.actor = Actor(in_space,  out_space, lr=3e-4).to('cpu')
    self.critic1 = Critic(in_space, out_space, lr=1e-4).to('cpu')
    self.critic2 = Critic(in_space, out_space, lr=1e-4).to('cpu')
    self.targ_critic1 = Critic(in_space, out_space, lr=1e-4).to('cpu')
    self.targ_critic2 = Critic(in_space, out_space, lr=1e-4).to('cpu')

    self.delay_step    = 2   # Denis Yarats' implementation delays this by 2.
    self.update_step   = 0
    self.target_update = 2 

    # entropy temperature
    #self.alpha = 0.2
    # TODO automatic entropy tuning
    self.target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to('cpu')).item()
    self.log_alpha = torch.zeros(1, requires_grad=True, device='cpu')
    self.alpha = self.log_alpha.exp().item()
    self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-4)

  def store(self,exp): 
    self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.tensor([obs]).float().to('cpu')
    action, _ = self.actor.evaluate(obs)
    return action.tolist()[0]
    #return action.detach().numpy()

  def train(self, batch_size=256):
    self.update_step += 1
    self.target_update += 1
    if len(self.memory) >= learning_start:
      states, actions, rewards, nstates, dones = self.memory.sample(batch_size)
      with torch.no_grad():
        # train critics
        nactions, nlog_probs = self.actor.evaluate(nstates)
        nq1 = self.targ_critic1.forward(nstates, nactions)
        nq2 = self.targ_critic2.forward(nstates, nactions)
        q_targ = torch.min(nq1 , nq2) - self.alpha * nlog_probs
        q_targ = rewards - self.gamma * q_targ.view(-1) * dones

      q1 = self.critic1.forward(states, actions).view(-1)
      q2 = self.critic2.forward(states, actions).view(-1)
      q1_loss = F.mse_loss(q1, q_targ)
      q2_loss = F.mse_loss(q2, q_targ)
      #print ( q2_loss, q1_loss)

      self.critic1.optimizer.zero_grad()
      q1_loss.backward()
      self.critic1.optimizer.step()

      self.critic2.optimizer.zero_grad()
      q2_loss.backward()
      self.critic2.optimizer.step()

      # train policy using TD 3 Delayed update
      #if self.update_step % self.delay_step == 0: 
      if self.update_step % self.delay_step == 0: 
        for _ in range(self.delay_step): # compensate for the delay by doing 'actor_update_interval' instead of 1
          _actions, log_probs = self.actor.evaluate(states)
          q1 = self.critic1.forward(states, _actions)
          q2 = self.critic2.forward(states, _actions)
          critic = torch.min(q1 , q2).view(-1)
          actor_loss = (self.alpha * log_probs - critic).mean()
          #print( actor_loss )

          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()

          with torch.no_grad():
            _, log_probs = self.actor.evaluate(states)
          alpha_loss = ( -self.log_alpha * (log_probs + self.target_entropy)).mean()
          #print( alpha_loss )

          self.a_optimizer.zero_grad()
          alpha_loss.backward()
          self.a_optimizer.step()
          self.alpha = self.log_alpha.exp().item()

      # update target networks
      if self.update_step % 1 == 0: # TD 3 Delayed update support
        for target_param, param in zip(self.targ_critic1.parameters(), self.critic1.parameters()):
          target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.targ_critic2.parameters(), self.critic2.parameters()):
          target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
    pass

env = gym.make('Pendulum-v0')
#env = gym.make('HopperBulletEnv-v0')
#env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = SAC( env.observation_space, env.action_space )

time_step = 0
#env.render(mode='human')
#env.render()
scores = []
for epi in range(5000):
  obs = env.reset()
  while True:
    #env.render()
    if time_step < learning_start:
      action = env.action_space.sample()
    else: action = agent.get_action(obs)

    _obs, reward, done, info = env.step(action)
    agent.store((obs, action, reward, _obs, done))
    time_step+=1
    agent.train()

    obs = _obs
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode: {epi}, Time step: {time_step}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
