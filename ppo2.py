import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class ActorNet(nn.Module):
  def __init__(self, in_dims, out_dims, lr):
    super(ActorNet, self).__init__()
    self.actor = nn.Sequential(
      nn.Linear(in_dims, 256), nn.ReLU(),
      nn.Linear(256, 256),     nn.ReLU(),
      nn.Linear(256, out_dims),
      nn.Softmax(dim=-1)
    )    
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, state):
    dist = self.actor(state)
    dist = Categorical(dist)
    return dist

class CriticNet(nn.Module):
  def __init__(self, in_dims, lr):
    super(CriticNet, self).__init__()
    self.critic = nn.Sequential(
      nn.Linear(in_dims, 256), nn.ReLU(),
      nn.Linear(256, 256),     nn.ReLU(),
      nn.Linear(256, 1),
    )    
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, state):
    value = self.critic(state)
    return value

class PPO: 
  def __init__(self,in_dims, out_dims):

    self.lr    = 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 10 
    self.eps_clip = 0.2  

    self.actor  = ActorNet( in_dims, out_dims, self.lr)
    self.critic = CriticNet(in_dims, self.lr)
    self.memory = []

  def selectAction(self, obs):
    state  = T.tensor(obs, dtype=T.float).to(self.actor.device)
    dist   = self.actor(state)
    value  = self.critic(state)
    action = dist.sample()

    probs  = T.squeeze(dist.log_prob(action)).item()
    action = T.squeeze(action).item()
    value  = T.squeeze(value).item()
    return action, probs, value
   
  def store(self,state,action,reward,probs,val,done):
    self.memory.append( (state,action,reward,probs,val,done) )

  def train(self, batch_size=20):
    states  = T.tensor([ x[0] for x in self.memory ], dtype=T.float).to(self.actor.device)
    actions = T.tensor([ x[1] for x in self.memory ], dtype=T.float).to(self.actor.device)
    rewards = np.array([ x[2] for x in self.memory ])
    probs   = T.tensor([ x[3] for x in self.memory ]).to(self.actor.device)
    vals    = np.array([ x[4] for x in self.memory ])
    dones   = np.array([1-int(x[5]) for x in self.memory]) ### potential error

    for epi in range( self.epoch ):
      ####### Advantage using gamma returns
      nvals = np.concatenate([vals[1:] ,[0]])
      delta = rewards + self.gamma*nvals*dones - vals
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()
      advantage = T.tensor(advantage, dtype=T.float)

      #print( advantage.shape, advantage.mean(), advantage, '\n')
      #print( advantage.shape)

      #advantage = np.zeros(len(rewards), dtype=np.float32)
      #for t in range(len(rewards)-1):
      #  adv, discount = 1, 0
      #  for k in range(t, len(rewards)-1):
      #    adv += discount*( rewards[k] + self.gamma*vals[k+1]*dones[k] - vals[k]  )
      #    discount *= self.gamma*self.lamda
      #  advantage[t] = adv
      #advantage = T.tensor(advantage).to(self.actor.device)

      # create minti batches
      num = len( states ) 
      indices = np.arange( num, dtype=np.int64 )
      np.random.shuffle( indices )
      batch_start = np.arange(0, num, batch_size)
      batches = [indices[i:i+batch_size] for i in batch_start]

      # iterate over batches and create loss function (actor loss + critic loss)
      for batch in batches:
        dist   = self.actor(states)
        value  = self.critic(states)

        new_probs = dist.log_prob(actions)
        ratio = new_probs.exp() / probs.exp()
        #print( ratio.mean() )
        #print( advantage[batch].mean() )
        #print( advantage[batch] )

        surr1 = ratio * advantage[batch]
        surr2 = T.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        actor_loss = -T.min(surr1, surr2).mean() 

        returns = T.tensor(vals[batch]) + advantage[batch]  
        critic_loss = (returns-value)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        #print( total_loss.item(), actor_loss, critic_loss )

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
    self.memory = []
    pass

import gym

# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

model = PPO(input_size, output_size)

#Hyperparameters
num_rollout = 20
games = 1000

N = 20
num_steps = 0
avg_score = 0

scores = []
for epi in range( games ):
  state = env.reset()
  done = False
  score = 0
  while not done:
    action, probs, val = model.selectAction(state)
    nstate, reward, done, info = env.step(action)
    model.store(state, action, reward, probs, val, done)

    num_steps +=1
    if num_steps % N == 0:
      model.train()
      #learn_iters += 1

    score += reward
    state = nstate

  scores.append(score)
  if(epi % 20 == 0):
    avg_score = np.mean(scores[-100:])
    print('episode', epi, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', num_steps)
  #print('episode', epi, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', num_steps, 'learning_steps', learn_iters)

env.close()
  
from bokeh.plotting import figure, show
p = figure(title="Simple line example", x_axis_label="iterations", y_axis_label="Scores")
x = np.arange(len(scores))
p.line(x, scores,  legend_label="scores", line_color="blue", line_width=2)
show(p) 
