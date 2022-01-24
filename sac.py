import gym
import pybullet_envs
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float).to('cpu')
    actions = torch.tensor([x[1] for x in batch], dtype=torch.float).to('cpu')
    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float).to('cpu')
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float).to('cpu')
    dones   = torch.tensor([1-int(x[4]) for x in batch]).to('cpu')
    #dones   = torch.tensor([int(x[4]) for x in batch])
    return states, actions, rewards, nstates, dones

# Initialize Policy weights
def layer_init(m):
  if isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)

time_step = 0          # global time step logger
learning_start = 5e3   # timestep to start learning
LOG_STD_MAX, LOG_STD_MIN = 2, -5 

""" Implementing a diagonal gaussian policy """
class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 256) # Better result with slightly wider networks.
    self.fc2 = nn.Linear(256, 128)
    self.fc_mean = nn.Linear(128, out_space.shape[0])
    self.fc_std  = nn.Linear(128, out_space.shape[0])
    self.apply(layer_init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

    # action rescaling
    self.action_scale = torch.FloatTensor((out_space.high - out_space.low)/2.)
    self.action_bias  = torch.FloatTensor((out_space.high + out_space.low)/2.)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    mean = self.fc_mean(x)
    log_std = self.fc_std(x)
    log_std = torch.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    return mean, log_std

  def evaluate(self, obs):
    mean, log_std = self.forward(obs)
    std = log_std.exp()
    dist = Normal(mean, std)
    x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)
    action = y_t * self.action_scale + self.action_bias
    log_probs = dist.log_prob(x_t)
    log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
    log_probs = log_probs.sum(1, keepdim=True)
    return action, log_probs

  def to(self, device):
    self.action_scale = self.action_scale.to(device)
    self.action_bias = self.action_bias.to(device)
    return super(Actor, self).to(device)

class Critic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.shape[0], 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)
    self.apply(layer_init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s,a], dim=1)
    x = F.relu(self.fc1(x)) 
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

class SAC:
  def __init__(self, in_space, out_space):

    self.tau    = 0.005 # for target network soft update
    #self.alpha  = 0.2   # initial entropy 
    self.gamma  = 0.90  # discount
    self.memory = ReplayBuffer(size=1000000)

    self.delay_step    = 1   # Denis Yarats' implementation delays this by 2.
    self.target_update = 1

    self.actor = Actor(in_space, out_space, lr=3e-4).to('cpu')
    self.critic1 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.critic2 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.targ_critic1 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.targ_critic2 = Critic(in_space, out_space, lr=1e-3).to('cpu')

    self.targ_critic1.load_state_dict(self.critic1.state_dict())
    self.targ_critic2.load_state_dict(self.critic2.state_dict())

    self.value_optimizer = optim.Adam(list(self.critic1.parameters()) 
                                      + list(self.critic2.parameters()), lr=1e-3)
    self.loss_fn = nn.MSELoss()

    # automatic entropy tuning
    self.target_entropy = - torch.prod(torch.Tensor(out_space.shape).to('cpu')).item()
    self.log_alpha = torch.zeros(1, requires_grad=True, device='cpu')
    self.alpha = self.log_alpha.exp().item()
    self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

  def store(self,exp): 
    self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.tensor([obs]).float().to('cpu')
    action, _ = self.actor.evaluate(obs)
    return action.tolist()[0]
    #return action.cpu().detach().numpy()[0]

  def train(self, batch_size=256):
    if len(self.memory) >= learning_start:
      states, actions, rewards, nstates, dones = self.memory.sample(batch_size)

      with torch.no_grad():
        nactions, nlog_probs = self.actor.evaluate(nstates)
        nq1 = self.targ_critic1.forward(nstates, nactions)
        nq2 = self.targ_critic2.forward(nstates, nactions)
        q_targ = torch.min(nq1 , nq2) - self.alpha * nlog_probs
        q_targ = rewards + self.gamma * q_targ.view(-1) * dones
        #q_targ = rewards + (1-dones) * self.gamma* q_targ.view(-1) 

      q1 = self.critic1.forward(states, actions).view(-1)
      q2 = self.critic2.forward(states, actions).view(-1)
      q2_loss = self.loss_fn(q2, q_targ)
      q1_loss = self.loss_fn(q1, q_targ)

      self.critic1.optimizer.zero_grad()
      q1_loss.backward()
      self.critic1.optimizer.step()

      self.critic2.optimizer.zero_grad()
      q2_loss.backward()
      self.critic2.optimizer.step()

      #print(q1_loss , q2_loss) 
      #value_loss = (q2_loss + q2_loss)/2
      ##print( value_loss)
      #self.value_optimizer.zero_grad()
      #value_loss.backward()
      #self.value_optimizer.step()

      if time_step % self.delay_step == 0: 
        for _ in range(self.delay_step): 
          _actions, log_probs = self.actor.evaluate(states)
          #print(type(_actions))
          q1 = self.critic1.forward(states, _actions)
          q2 = self.critic2.forward(states, _actions)
          critic = torch.min(q1 , q2).view(-1)
          actor_loss = (self.alpha * log_probs - critic).mean()
          #print( actor_loss)

          self.actor.optimizer.zero_grad()
          actor_loss.backward()
          self.actor.optimizer.step()

          # auto tune temp
          with torch.no_grad():
            _d, log_probs = self.actor.evaluate(states)
          alpha_loss = ( -self.log_alpha * (log_probs + self.target_entropy)).mean()
          self.a_optimizer.zero_grad()
          alpha_loss.backward()
          self.a_optimizer.step()
          self.alpha = self.log_alpha.exp().item()

      # update target networks
      if time_step % self.target_update == 0: # TD 3 Delayed update support
        for target_param, param in zip(self.targ_critic1.parameters(), self.critic1.parameters()):
          target_param.data.copy_(self.tau * param.data+ (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.targ_critic2.parameters(), self.critic2.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

      # update the target network
      #if time_step % self.target_update == 0: # TD 3 Delayed update support
      #  for param, target_param in zip(self.critic1.parameters(), self.targ_critic1.parameters()):
      #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
      #  for param, target_param in zip(self.critic2.parameters(), self.targ_critic2.parameters()):
      #    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    pass


#env = gym.make('Pendulum-v0')
env = gym.make('HopperBulletEnv-v0')
#env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)


agent = SAC( env.observation_space, env.action_space )

#env.render()
scores = []
for epi in range(2000):
  obs = env.reset()
  while True:
    if time_step < learning_start:
      action = env.action_space.sample()
    else: 
      action = agent.get_action(obs)

    _obs, reward, done, info = env.step(action)
    agent.store((obs, action, reward, _obs, done))

    agent.train()
    obs = _obs
    time_step+=1
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      #print(f"Episode: {epi}, Time step: {time_step}, Return: {scores[-1]}, Avg return: {avg_score}")
      if epi % 10 ==0:
        print(f"global_step={time_step}, episode_reward={int(info['episode']['r'])}")
      break
env.close()
