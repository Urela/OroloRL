import gym
import torch 
import numpy as np
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
    self.to('cpu')

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
    self.to('cpu')

  def forward(self, state):
    value = self.critic(state)
    return value

class PPO: 
  def __init__(self, in_dims, out_dims, bsize=20):
    self.lr    = 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2
    self.bsize = 5 # batch_size


    self.actor  = ActorNet( in_dims, out_dims, self.lr)
    self.critic = CriticNet(in_dims, self.lr)

    self.memory = []

  def store(self, state, action, reward, probs, vals, done):
    self.memory.append( (state, action, reward, probs, vals, done) )

  def selectAction(self, obs):
    obs  = torch.tensor([obs], dtype=torch.float).to('cpu')
    value  = self.critic(obs)

    dist   = self.actor(obs)
    action = dist.sample()

    probs  = torch.squeeze(dist.log_prob(action)).item()
    action = torch.squeeze(action).item()
    value  = torch.squeeze(value).item()
    return action, probs, value

  def train(self):
    for _ in range( self.epoch):
      states  = np.array([x[0] for x in self.memory])
      actions = np.array([x[1] for x in self.memory])
      rewards = np.array([x[2] for x in self.memory])
      probs   = np.array([x[3] for x in self.memory])
      values  = np.array([x[4] for x in self.memory])
      dones   = np.array([1-int(x[5]) for x in self.memory])

      ####### Advantage using gamma returns
      nvalues = np.concatenate([values[1:] ,[values[-1]]])
      delta = rewards + self.gamma*nvalues*dones - values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu')
      values    = torch.tensor(values).to('cpu')
      
      # create mini batches
      num = len( states ) 
      batch_start = np.arange(0, num, self.bsize)

      indices = np.arange( num, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in batch_start]

      for batch in batches:
        _states  = torch.tensor(states[batch], dtype=torch.float).to('cpu')
        _probs   = torch.tensor(probs[batch], dtype=torch.float).to('cpu')
        _actions = torch.tensor(actions[batch], dtype=torch.float).to('cpu')

        dist   = self.actor(_states)
        nvalue = self.critic(_states)
        nvalue = torch.squeeze(nvalue)

        new_probs = dist.log_prob(_actions)
        ratio = new_probs.exp() / _probs.exp()

        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        actor_loss = -torch.min(surr1, surr2).mean() 

        returns = advantage[batch] + values[batch]
        critic_loss = (returns-nvalue)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    self.memory = []
    pass





# Global variables
env=gym.make('CartPole-v0')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

n_games = 300
N = 20


agent = PPO(input_size, output_size)

learn_iters = 0
avg_score   = 0
n_steps     = 0


scores = []
avg_scores = []

for i in range(n_games):
  state = env.reset()
  done = False
  score = 0
  while not done:

    #env.render()
    #action,prob,val = env.action_space.sample() , 0 ,0
    action, prob, val = agent.selectAction(state)

    nstate, reward, done, info = env.step(action)
    n_steps += 1
    score += reward
    agent.store(state, action, reward, prob, val, done)
    if n_steps % N == 0:
      agent.train()
      learn_iters += 1
    state = nstate
  scores.append(score)
  avg_score = np.mean(scores[-10:])
  avg_scores.append( avg_score)

  print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
  #if(i % 20 == 0):
  #  print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

env.close()
  
from bokeh.plotting import figure, show
p = figure(title="running average of past 10 games", x_axis_label="iterations", y_axis_label="Scores")
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="scores", line_color="blue", line_width=2)
show(p) 
