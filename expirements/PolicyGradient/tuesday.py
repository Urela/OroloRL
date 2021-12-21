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
      nn.Linear(in_dims, 64), nn.ReLU(),
      nn.Linear(64, 64),     nn.ReLU(),
      nn.Linear(64, out_dims),
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
      nn.Linear(in_dims, 64), nn.ReLU(),
      nn.Linear(64, 64),     nn.ReLU(),
      nn.Linear(64, 1),
    )    
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.to('cpu')

  def forward(self, state):
    value = self.critic(state)
    return value

class PPO: 
  def __init__(self, in_dims, out_dims, size, bsize):
    self.lr    = 0.0003
    self.gamma = 0.99 
    self.lamda = 0.95
    self.epoch = 4
    self.eps_clip = 0.2

    self.actor  = ActorNet( in_dims, out_dims, self.lr)
    self.critic = CriticNet(in_dims, self.lr)

    self.states  = np.zeros((size, in_dims), dtype=np.float32)
    self.actions = np.zeros(size, dtype=np.int32)
    self.rewards = np.zeros(size, dtype=np.float32)
    self.probs   = np.zeros(size, dtype=np.float32)
    self.values  = np.zeros(size, dtype=np.float32)
    self.dones   = np.zeros(size, dtype=np.int32)
    self.size, self.bsize, self.idx  = size, bsize, 0

  def store(self, state, action, reward, probs, vals, done):
    idx = self.idx % self.size
    self.idx += 1
    self.states  [idx] = state
    self.actions [idx] = action
    self.rewards [idx] = reward
    self.probs   [idx] = probs
    self.values  [idx] = vals
    self.dones   [idx] = 1-int(done)

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
      ####### Advantage using gamma returns
      nvalues = np.concatenate([self.values[1:] ,[self.values[-1]]])
      delta = self.rewards + self.gamma*nvalues* self.dones - self.values
      advantage, adv = [], 0
      for d in delta[::-1]:
        adv = self.gamma * self.lamda * adv + d
        advantage.append(adv)
      advantage.reverse()

      advantage = torch.tensor(advantage).to('cpu')
      values    = torch.tensor(self.values).to('cpu')
      
      # create mini batches
      batch_start = np.arange(0, self.size, self.bsize)

      indices = np.arange( self.size, dtype=np.int64 )
      np.random.shuffle( indices )
      batches = [indices[i:i+self.bsize] for i in batch_start]

      for batch in batches:
        states  = torch.tensor(self.states[batch], dtype=torch.float).to('cpu')
        actions = torch.tensor(self.actions[batch], dtype=torch.float).to('cpu')
        old_probs = torch.tensor(self.probs[batch], dtype=torch.float).to('cpu')

        dist = self.actor(states)
        crit = self.critic(states)
        crit = torch.squeeze(crit)

        new_probs = dist.log_prob(actions)
        ratio = new_probs.exp() / old_probs.exp()

        surr1 = ratio * advantage[batch]
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage[batch]
        actor_loss = -torch.min(surr1, surr2).mean() 

        returns = advantage[batch] + values[batch]
        critic_loss = (returns-crit)**2
        critic_loss = critic_loss.mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    self.idx=0
    pass

# Global variables
env=gym.make('CartPole-v0')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

n_games = 300
N = 20

agent = PPO(input_size, output_size, size=N, bsize=5)

learn_iters = 0
avg_score   = 0
n_steps     = 0

scores, avg_scores = [], []

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

  if(i % 10 == 0):
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

env.close()
  
from bokeh.plotting import figure, show
p = figure(title="running average of past 10 games", x_axis_label="iterations", y_axis_label="Scores")
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="scores", line_color="blue", line_width=2)
show(p) 
