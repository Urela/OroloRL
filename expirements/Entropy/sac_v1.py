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
    actions = torch.tensor([x[1] for x in batch])
    rewards = torch.tensor([[x[2]] for x in batch]).float()
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
    dones   = torch.tensor([int(x[4]) for x in batch])
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc_mean = nn.Linear(256, out_space.shape[0])
    self.fc_std  = nn.Linear(256, out_space.shape[0])

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.max_action=out_space.high

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    mean = self.fc_mean(x)
    log_std = self.fc_std(x)
    log_std = torch.clamp(log_std, min=-20, max=2)  # bound std and its distbuiton
    return mean, log_std

  def evaluate(self, obs, reparameterize=False, epsilon=1e-6):
    mean, log_std = self.forward(obs)
    std = log_std.exp()
    dist = Normal(mean, std)

    if reparameterize: z = dist.rsample().to('cpu') 
    else: z = dist.sample().to('cpu') 
    action = torch.tanh(z) 

    action = torch.tanh(action)*torch.tensor(self.max_action).to('cpu')
    log_probs = dist.log_prob(action)
    log_probs -= torch.log(1 - action.pow(2) + epsilon)
    #log_probs = log_probs.sum(1, keepdim=True)           # ????? TODO
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

class SAC:
  def __init__(self, in_space, out_space):
    self.tau = 0.005
    self.gamma = 0.99
    self.scale = 2 
    self.memory = ReplayBuffer()

    self.actor = Actor(in_space,  out_space, lr=3e-4).to('cpu')
    self.critic1 = Critic(in_space, out_space, lr=3e-4).to('cpu')
    self.critic2 = Critic(in_space, out_space, lr=3e-4).to('cpu')
    self.targ_critic1 = Critic(in_space, out_space, lr=3e-4).to('cpu')
    self.targ_critic2 = Critic(in_space, out_space, lr=3e-4).to('cpu')

    #self.copy_networks(self.critic1, self.targ_critic1)
    #self.copy_networks(self.critic2, self.targ_critic2)

    # entropy temperature
    self.alpha = 0.2
    self.target_entropy = - torch.Tensor(out_space.shape).prod().item()
    self.log_alpha = torch.zeros(1, requires_grad=True)
    self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-3)

    self.update_step = 0
    self.delay_step = 2

  def copy_networks(self, org_net, dest_net):
    for dest_param, param in zip(dest_net.parameters(), org_net.parameters()):
      dest_param.data.copy_(param.data)

  def store(self,exp): 
    self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.tensor(obs).float()
    action, _ = self.actor.evaluate(obs)
    return action.detach().numpy()

  def train(self, batch_size=128):
    if len(self.memory) >= batch_size:
      #print( "training" )
      states, actions, rewards, nstates, dones = self.memory.sample(batch_size)

      # train critics
      q1 = self.critic1.forward(states, actions)
      q2 = self.critic2.forward(states, actions)

      nactions, nlog_probs = self.actor(nstates)
      nq1 = self.targ_critic1.forward(nstates, nactions)
      nq2 = self.targ_critic2.forward(nstates, nactions)

      q_targ = torch.min(nq1 , nq2) - self.alpha * nlog_probs
      q_targ[dones] = 0.0  # set all terminal states' value to zero
      q_targ = rewards - self.gamma * q_targ

      q1_loss = F.mse_loss(q1, q_targ.detach())
      q2_loss = F.mse_loss(q2, q_targ.detach())
      #print("q1_loss",  q1_loss)
      #print("q2_loss",  q1_loss)

      self.critic1.optimizer.zero_grad()
      q1_loss.backward()
      self.critic1.optimizer.step()

      self.critic2.optimizer.zero_grad()
      q2_loss.backward()
      self.critic2.optimizer.step()

      #train policy
      _actions, log_probs = self.actor.evaluate(states)
      if self.update_step % self.delay_step == 0:
        q1 = self.critic1.forward(states, _actions)
        q2 = self.critic2.forward(states, _actions)
        critic = torch.min(q1 , q2).view(-1)
        actor_loss = (self.alpha * log_probs - critic).mean()
        #print("actor_loss",  actor_loss)

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # target networks
        for target_param, param in zip(self.targ_critic1.parameters(), self.critic1.parameters()):
          target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        for target_param, param in zip(self.targ_critic2.parameters(), self.critic2.parameters()):
          target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

      # update temperature
      alpha_loss = (self.log_alpha * (-log_probs - self.target_entropy).detach()).mean()
      #print("alpha  _loss",  alpha_loss)

      self.alpha_optim.zero_grad()
      alpha_loss.backward()
      self.alpha_optim.step()
      self.alpha = self.log_alpha.exp()
      self.update_step += 1
    pass

#env = gym.make('Pendulum-v0')
env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = SAC( env.observation_space, env.action_space )

time_step = 0

#env.render(mode='human')
#env.render()
scores = []
for epi in range(1000):
  obs = env.reset()
  while True:
    #env.render()
    action = agent.get_action(obs)
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
