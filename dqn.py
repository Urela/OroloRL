import gym
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayMemory():
  def __init__(self, MaxSize, state_space,  action_space):
    self.idx  = 0         # postion in memory stack
    self.size = MaxSize   # max size of memory stack
    self.states  = np.zeros((MaxSize, state_space), dtype=np.float32)
    self.nstates = np.zeros((MaxSize, state_space), dtype=np.float32)
    self.actions = np.zeros( MaxSize, dtype=np.int32)
    self.rewards = np.zeros( MaxSize, dtype=np.float32)
    self.dones = np.zeros(MaxSize, dtype=bool)

  def store(self, state, action, reward, nstate, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = state
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.nstates[idx] = nstate
    self.dones[idx] = done

  def sample(self, batch_size, device='cpu'):
    csize = min(self.idx, self.size)    # current size of memory stack
    batch = np.random.choice(csize, batch_size) # sample batch
    
    states  = torch.from_numpy( self.states[batch]  ).float().to(device)
    rewards = torch.from_numpy( self.rewards[batch] ).float().to(device)  
    nstates = torch.from_numpy( self.nstates[batch] ).float().to(device)  
    actions = self.actions[batch] 
    dones   = self.dones[batch]   
    return states, actions, rewards, nstates, dones

  def __len__(self): return self.idx

class Network(nn.Module):
  def __init__(self, inputs, outputs, lr=1e-3):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(inputs, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, outputs)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)
    return actions

  def save(self, filename): torch.save(self.state_dict(), filename)
  def load(self, filename, device='cpu'): self.load_state_dict(torch.load(filename, map_location=device))

class DQN(object):
  def __init__(self, inputs, outputs):
    self.lr = 1e-3 
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4
    self.device  = torch.device('cpu')
    self.outputs = outputs
    
    self.memory = ReplayMemory(100000, inputs, outputs) # memory
    self.policy = Network(inputs, outputs, self.lr).to(self.device) # policy network
    self.target = Network(inputs, outputs, self.lr).to(self.device) # target network
    self.target.eval() # since no learning is performed on the target net

  def store_memory(self, s, a, r, ns, done): self.memory.store(s, a, r, ns, done)
  def update_epsilon(self): self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)
  def update_target(self): self.target.load_state_dict( self.policy.state_dict() )

  def select_action(self, state): 
    if np.random.random() > self.epsilon:
      state = torch.tensor([state], dtype=torch.float32).to(self.device).float() # make sure state is pytorch tensor
      return self.policy.forward(state).argmax().item()  # return action with highest Q value
    return np.random.randint( self.outputs, size=1)[0]  # random action

  def learn(self, batch_size=1):
    if len(self.memory) < batch_size: return # if buffer not full don't learn 
    else:
      s, a, r, ns, dones = self.memory.sample(batch_size) # sample memory batchess
      b_idx = np.arange(batch_size, dtype=np.int32)

      q_pred = self.policy.forward(s)[b_idx, a]
      q_next = self.target.forward(ns)
      q_next[dones] = 0.0                   # set all terminal state values to zero 

      ## calculate the loss as the mean-squared error of q_next and q_pred
      q_next = r + self.gamma*torch.max(q_next, dim=1)[0]
      self.policy.optimizer.zero_grad()     # clearing tempory gradients 
      loss = self.policy.loss(q_next, q_pred).to(self.device) # detemine loss
      loss.backward()                       # determine gradients
      self.policy.optimizer.step()          # update weights

# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

episodes       = 10000
max_time_steps = 1000
agent = DQN(input_size, output_size)

scores, avg_scores, eps_history = [], [], []
for epi in range( episodes ):
  state = env.reset()
  score = 0
  done = False
  while not done:
    #if epi > 1000: env.render()
    action = agent.select_action( state )
    nstate, reward, done, info = env.step(action)

    agent.store_memory(state, action, reward, nstate, done)
    agent.learn(batch_size=64) # do some learing if our buffer is full

    score += reward
    state = nstate

  scores.append(score)
  avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
  avg_scores.append(avg_score)
  eps_history.append(agent.epsilon)
  print('Episode: {}, Score: {}, Avg score: {}, '.format(epi, score, avg_score))

  agent.update_epsilon() # update epsilon value after each episode 
  agent.update_target()  # update target network after each episode
  if len( scores ) > 100: scores=[]


from bokeh.plotting import figure, show

# create a new plot with a title and axis labels
p = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")

# add a line renderer with legend and line thickness
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="Rewards", line_color="blue", line_width=2)
p.line(x, eps_history, legend_label="Epsilon", line_color="red",  line_width=2)

show(p) # show the results
