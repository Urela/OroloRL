"""
Continous PPO
"""
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
    self.lr    = 1e-5   # 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 10
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
    self.actions [idx] = action .detach().numpy()
    self.rewards [idx] = reward
    self.probs   [idx] = probs  .detach().numpy()
    self.values  [idx] = vals   .detach().numpy()
    self.dones   [idx] = 1-(done)

  def get_action(self, obs):
    value = self.AC.critic(obs)
    mean  = self.AC.actor(obs)
    log_std = self.AC.actor_logstd.expand_as(mean)
    std = log_std.exp()
    dist = Normal(mean, std)

    action = dist.sample()
    probs  = torch.squeeze(dist.log_prob(action)).sum(1)
    action = torch.squeeze(action)
    value  = torch.squeeze(value)
    #print( action.shape, probs.shape, value.shape)
    return action, probs, value

  def train(self):
    for epi in range( self.epoch):
      # finding Advantages using gamma returns
      nvalues = np.concatenate([self.values[1:] ,[self.values[-1]]])
      delta = self.rewards + self.gamma*nvalues* self.dones - self.values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu').reshape(-1)
      values    = torch.tensor(self.values.reshape(-1)).to('cpu')
      
      # create mini batches
      indices = np.arange( self.size, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in range(0, self.size, self.bsize)]

      # reward Annealing
      frac = 1.0 - (epi - 1.0) / self.epoch
      lrnow = frac * self.lr
      self.AC.optimizer.param_groups[0]["lr"] = lrnow

      c, h, w = self.states.shape
      _states = self.states.reshape(c*h, w)
      _actions = self.actions.reshape(-1)
      _probs = self.probs.reshape(-1)

      for batch in batches:
        states    = torch.tensor(_states[batch], dtype=torch.float).to('cpu')
        actions   = torch.tensor(_actions[batch], dtype=torch.float).to('cpu')
        old_probs = torch.tensor(_probs[batch], dtype=torch.float).to('cpu')

        # Evaluating old actions and values
        #print(states.shape)
        action, new_probs, crit = self.get_action(states)
        new_probs = torch.tensor( new_probs ).to('cpu')
        #print( _states.shape, states.shape, new_probs.shape, old_probs.shape)

        # Finding the ratio (pi_theta / pi_theta__old)
        #new_probs = dist.log_prob(actions)
        ratio = new_probs.exp() / old_probs.exp()
        #print(ratio)

        # Finding Surrogate Loss
        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        returns = advantage[batch] + values[batch]

        # final loss of clipped objective PPO: loss = actor_loss + 0.5*critic_loss 
        loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-crit)**2).mean()
        #print( loss) 

        # take gradient step
        self.AC.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(self.AC.parameters(), 0.5)
        self.AC.optimizer.step()

    self.idx=0
    pass

time_step = 0
update_frequency = 128
num_envs  = 4       # number of parallel environments
env = gym.vector.SyncVectorEnv(
    [lambda: gym.make('HalfCheetahBulletEnv-v0') for _ in range(num_envs) ]
    #[lambda: gym.make('CartPole-v1') for _ in range(num_envs) ]
)
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = PPO(env.single_observation_space, env.single_action_space, batch_size=4, num_steps=update_frequency, num_envs=num_envs)

#env.render() 
scores = [0]
for epi in range(3000):
  DONE = False
  obs = env.reset()
  while not DONE:


    action, probs, vals = agent.get_action( torch.tensor([obs]).float().to('cpu') )
    #action, probs, vals = env.action_space.sample(), 0 ,0
    _obs, reward, done, info = env.step(action)
    reward /= 10
    agent.store(obs, action, reward, probs, vals, done)

    time_step+=1
    if time_step % update_frequency ==0:
      agent.train()

    #if time_step % 1000 ==0:
    #  avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
    #  print(f"global_step={time_step}, episode_reward={scores[-1]}, Avg return: {avg_score}")

    obs = _obs
    for item in info:
      if "episode" in item.keys():
        scores.append(int(item['episode']['r']))
        avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
        print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
        #if epi % 10 ==0:
        #  print(f"global_step={time_step}, episode_reward={scores[-1]}, Avg return: {avg_score}")
        DONE = True
