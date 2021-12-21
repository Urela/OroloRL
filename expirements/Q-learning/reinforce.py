import gym
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, lr=3e-4):
    super(PolicyNetwork, self).__init__()
    self.fc1 = nn.Linear(state_size, 128)
    self.fc2 = nn.Linear(128, 128)
    self.fc3 = nn.Linear(128, action_size)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self,state): 
    x = F.relu( self.fc1(state))
    x = F.relu( self.fc2(x))
    x = self.fc3(x)
    return x 

class PG_Agent():
  def __init__(self, state_size, action_size=4):
    self.lr = 1e-3 
    self.gamma = 0.99
    self.device = T.device('cpu')

    self.reward_memory = []
    self.action_memory = []

    self.policy = PolicyNetwork(state_size, action_size, self.lr)

  def select_action(self, state):
    state = T.Tensor([state]).to(self.device)
    probs = F.softmax(self.policy.forward(state))

    action_probs = T.distributions.Categorical(probs)
    action = action_probs.sample()
    log_probs = action_probs.log_prob(action)
    self.action_memory.append(log_probs)

    return action.item()

  def store_rewards(self, reward):
    self.reward_memory.append(reward)

  def learn(self):
    self.policy.optimizer.zero_grad()

    # G_t = R_t+1 + gamma * R_t+2 + gamma**2 * R_t+3
    # G_t = sum from k=0 to k=T {gamma**k * R_t+k+1}
    G = np.zeros_like(self.reward_memory, dtype=np.float64)
    for t in range(len(self.reward_memory)):
      G_sum = 0
      discount = 1
      for k in range(t, len(self.reward_memory)):
        G_sum += self.reward_memory[k] * discount
        discount *= self.gamma
      G[t] = G_sum
    G = T.tensor(G, dtype=T.float).to(self.device)
    
    loss = 0
    for g, logprob in zip(G, self.action_memory):
      loss += -g * logprob
    
    loss.backward()
    self.policy.optimizer.step()

    self.action_memory = []
    self.reward_memory = []

# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

agent = PG_Agent(input_size, output_size)
episodes = 3000

scores = []
for epi in range( episodes ):
  state = env.reset()
  done = False
  score = 0
  while not done:
    action = agent.select_action( state )
    next_state, reward, done, info = env.step(action)
    agent.store_rewards( reward )

    score += reward
    state = next_state
  agent.learn()
  scores.append(score)
  avg_score = np.mean(scores[-100:])

  print('Episode: {}, Score: {}, Avg score: {}, '.format(epi, score,avg_score))

running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
  running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
from bokeh.plotting import figure, show

# create a new plot with a title and axis labels
p = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")

# add a line renderer with legend and line thickness
x = np.arange(len(scores))
p.line(x, running_avg,  legend_label="Running average of previous 100 scores", line_color="blue", line_width=2)
show(p) # show the results
