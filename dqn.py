import gym
import random
import numpy as np
import collections
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0001):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, out_space.n)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    x = self.fc3(x)
    return x

class DQN:
  def __init__(self, in_space, out_space):
    self.lr = 1e-3
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4
    self.action_size = out_space.n
    self.memory  = collections.deque(maxlen=100000)
    self.policy = Network( in_space, out_space, self.lr).to('cpu')
    self.target = Network( in_space, out_space, self.lr).to('cpu') 

  def update_target(self):
    self.target.load_state_dict( self.policy.state_dict() )

  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)

  def get_action(self, obs):
    if np.random.random() > self.epsilon:
      obs = torch.tensor(obs, dtype=torch.float).to('cpu')
      action = self.policy(obs)
      action = action.argmax().item()
    else:
      action = np.random.randint( self.action_size, size=1)[0]
    return action

  def train(self, batch_size=64):
    self.loss = nn.MSELoss()
    if len(self.memory) >= batch_size:
      for i in range(10):
        batch = random.sample(self.memory, batch_size)
        states  = torch.tensor([x[0] for x in batch], dtype=torch.float)
        actions = torch.tensor([[x[1]] for x in batch])
        rewards = torch.tensor([[x[2]] for x in batch]).float()
        nstates = torch.tensor([x[3] for x in batch], dtype=torch.float)
        dones   = torch.tensor([x[4] for x in batch])

        q_pred = self.policy(states).gather(1, actions)
        q_targ = self.target(nstates).max(1)[0].unsqueeze(1)
        q_targ[dones] = 0.0  # set all terminal states' value to zero
        q_targ = rewards + self.gamma * q_targ 

        #dones   = torch.tensor([[1-int(x[4])] for x in batch]  )
        #q_targ = rewards + self.gamma * q_targ  *dones

        loss = F.smooth_l1_loss(q_pred, q_targ).to('cpu')
        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()
    pass

env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = DQN( env.observation_space, env.action_space )

update_frequency = 500
time_step = 0

scores = []
for epi in range(1000):
  obs = env.reset()
  while True:
    action = agent.get_action(obs)
    _obs, reward, done, info = env.step(action)
    agent.memory.append((obs, action, reward, _obs, done))
    time_step+=1
    agent.train()

    obs = _obs
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
  agent.update_epsilon()    # update epsilon value after each episode
  agent.update_target() # update target network after each episode
