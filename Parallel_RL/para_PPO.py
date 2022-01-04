import gym
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

'''
Multi environment adapted from 
   https://github.com/vwxyzjn/PPO-Implementation-Deep-Dive/blob/master/ppo.py#L132
'''

def make_env(gym_id, idx, seed, record=False, run_name=''):
  def thunk():
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record and idx==0:
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # seeds for reproductivity
    env.seed(seed)                   
    env.action_space.seed(seed)      
    env.observation_space.seed(seed) 
    return env
  return thunk

class ActorCritic(nn.Module):
  def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

  def __init__(self, in_dims, out_dims, lr=2.5e-4):
    super(ActorCritic, self).__init__()

    self.fc_actor = nn.Sequential(
      self.layer_init(nn.Linear(np.array(in_dims).prod(), 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, out_dims), std=0.01),
    )

    self.fc_critic = nn.Sequential(
      self.layer_init(nn.Linear(np.array(in_dims).prod(), 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 64)),
      nn.Tanh(),
      self.layer_init(nn.Linear(64, 1), std=1.0),
    )

    self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)

  def critic(self, obs):
    return self.fc_critic(obs)

  def actor(self, obs):
    dist = self.fc_actor(obs)
    probs = Categorical(logits=dist)
    action = probs.sample()
    return action.numpy(), probs.log_prob(action)

class PPO:  
  def __init__(self, in_dims, out_dims, batch_size=5):
    self.lr    = 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2
    self.bsize = batch_size
    
    self.AC = ActorCritic(in_dims, out_dims).to('cpu')
    self.memory = []

  def selectAction(self, obs):
    action, log_probs = self.AC.actor(obs)
    value = self.AC.critic(obs)
    return action, log_probs, value

  def train(self):
    for _ in range( self.epoch):
      states  = np.array([x[0] for x in self.memory])
      actions = np.array([x[1] for x in self.memory])
      rewards = np.array([x[2] for x in self.memory])
      probs   = np.array([x[3].detach().numpy() for x in self.memory])
      values  = np.array([x[4].detach().numpy() for x in self.memory])
      dones   = np.array([[1-(x[5])] for x in self.memory])

      #print( states.shape)
      ####### Advantage using gamma returns
      values = np.squeeze(values)
      nvalues = np.concatenate([values[1:] ,[values[-1]]])
      #print(rewards.shape, values.shape, nvalues.shape)

      delta = rewards + self.gamma * nvalues * dones - values
      #print(delta.shape, rewards.shape, values.shape, nvalues.shape)

      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      #print( values.shape, delta.shape, reward.shape, np.array(advantage).shape)

      advantage = torch.tensor(advantage).to('cpu')
      values    = torch.tensor(values).to('cpu')
      
      # create mini batches
      num = len( states ) 
      batch_start = np.arange(0, num, self.bsize)

      indices = np.arange( num, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in batch_start]

      for batch in batches:
        _states   = torch.tensor(states[batch], dtype=torch.float).to('cpu')
        old_probs = torch.tensor(probs[batch], dtype=torch.float).to('cpu')
        _actions  = torch.tensor(actions[batch], dtype=torch.float).to('cpu')

        # Evaluating old actions and values
        _, new_probs = self.AC.actor(_states)
        crit = self.AC.critic(_states)
        crit = torch.squeeze(crit)

        # Finding the ratio (pi_theta / pi_theta__old)
        ratio = new_probs.exp() / old_probs.exp()

        # Finding Surrogate Loss
        #print( ratio.shape, new_probs.shape, old_probs.shape,  advantage[batch].shape)
        #print( ratio.shape, advantage.shape, advantage[batch].shape, batch.shape)
        surr1 = ratio * advantage[batch]

        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        returns = advantage[batch] + values[batch]

        # final loss of clipped objective PPO: loss = actor_loss + 0.5*critic_loss 
        loss = -torch.min(surr1, surr2).mean() + 0.5*((returns-crit)**2).mean()

        # take gradient step
        self.AC.optimizer.zero_grad()
        loss.mean().backward()
        self.AC.optimizer.step()

    self.memory = []
    pass


# Global variables
gym_id = "CartPole-v1"
seed        = 1
num_envs    = 4       # number of parallel environments

max_ep_len  = 400   # max time steps per episodes
total_steps = 25000 # number of time steps to observe  ( n / num of enivoronments )
update_freq = 128   # update policy every n times teps ( n / num of enivoronments )
batch_size  = num_envs * update_freq

#update_freq = max_ep_len * 4    # update policy every n timesteps
#num_updates = total_steps // batch_size # number of updates we need to do

envs = gym.vector.SyncVectorEnv([
      make_env(gym_id, i, seed+i, record=False, run_name=f"{gym_id}__{int(time.time())}" ) 
          for i in range(num_envs)
      ])

agent = PPO(envs.single_observation_space.shape, envs.single_action_space.n, batch_size)

timestep  = 0 
i_episode = 0
scores, avg_scores = [], []

obs = envs.reset()
while timestep <= total_steps:
  score = 0
  for t in range(1, max_ep_len+1):

    #obs = torch.tensor(obs).to('cpu') 
    action, log_probs, value = agent.selectAction(torch.tensor(obs).to('cpu'))
    #action, log_probs, value = agent.selectAction(obs)
    _obs, reward, done, info = envs.step(action)
    agent.memory.append( (obs, action, reward, log_probs, value, done) )
    obs = _obs

    timestep += 1
    score += reward

    # train PPO agent
    if timestep % update_freq == 0: 
      agent.train()

      for item in info:
        if "episode" in item.keys():
          score = item['episode']['r']
          scores.append( score  )
          avg = np.round(np.mean(scores[-100:]), 2)
          i_episode+=1
          print(f"Episode: {i_episode}  Episodic return: {score} Avg returns:{avg} Time step: {timestep} ")
          #print(f"Episode: {i_episode}  Episode stats: {item['episode']}  Time step: {timestep} ")

