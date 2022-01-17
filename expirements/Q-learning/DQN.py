import gym
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############# stores memory for experience replay ############# 
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
    
    states  = T.from_numpy( self.states[batch]  ).float().to(device)
    rewards = T.from_numpy( self.rewards[batch] ).float().to(device)  
    nstates = T.from_numpy( self.nstates[batch] ).float().to(device)  
    actions = self.actions[batch] 
    dones   = self.dones[batch]   
    return states, actions, rewards, nstates, dones

  def __len__(self): return self.idx

############### Neural netowrk model ############### 
class Network(nn.Module):
  def __init__(self, input_size, output_size, lr=1e-3):
    super(Network, self).__init__()
    self.fc1 = nn.Linear(input_size, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, output_size)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)
    return actions

  def save_model(self, filename):
    T.save(self.state_dict(), filename)

  def load_model(self, filename, device):
    # map_location is required to ensure that a model that is trained on GPU can be run even on CPU
    self.load_state_dict(T.load(filename, map_location=device))

class DQNAgent():
  def __init__(self, state_size, action_size):
    self.lr = 1e-3 
    self.gamma   = 0.99
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4
    self.device  = T.device('cpu')

    # size of the state vectors and number of possible actions
    self.state_size  = state_size
    self.action_size = action_size

    # memory
    self.memory = ReplayMemory(100000, state_size, action_size)

    # neural networks
    self.policy_net = Network(state_size, action_size, self.lr).to(self.device) # policy network
    self.target_net = Network(state_size, action_size, self.lr).to(self.device) # target network
    self.target_net.eval() # since no learning is performed on the target net
    # if not train_mode: self.policy_net.eval()

  def update_target_net(self):
    self.target_net.load_state_dict( self.policy_net.state_dict() )

  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)

  def remember(self, state, action, reward, nstate, done):
    self.memory.store(state, action, reward, nstate, done)

  # actions are selected using epsilon greedy method
  def select_action(self, state):
    if np.random.random() > self.epsilon:
      state = T.tensor([state], dtype=T.float32).to(self.device).float()  # make sure state is pytorch tensor
      actions = self.policy_net.forward(state)  # collect actions values
      action  = T.argmax( actions ).item()      # return action with highest Q
    else:
      action = np.random.randint( self.action_size, size=1)[0] 
    return action

  def learn(self, batch_size=1):
    if len(self.memory) < batch_size: return # if buffer not full don't learn 
    else:
      
      batch_index = np.arange(batch_size, dtype=np.int32)
      states, actions, rewards, nstates, terminals = self.memory.sample(batch_size)

      # want the action in a batch ??? not sure
      q_pred   = self.policy_net.forward( states )[batch_index, actions] 
      q_target = self.target_net.forward( nstates )
      q_target[terminals] = 0.0  # set all terminal states' value to zero 

      ## calculate the loss as the mean-squared error of yj and qpred
      q_target = rewards + self.gamma*T.max(q_target, dim=1)[0]
      self.policy_net.optimizer.zero_grad()   # clearing tempory gradients 
      loss = self.policy_net.loss(q_target, q_pred).to(self.device) # detemine loss
      loss.backward()                          # determine gradients
      self.policy_net.optimizer.step()         # update weights



# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

episodes       = 300
max_time_steps = 1000
update_frequency = 500
agent = DQNAgent(input_size, output_size)

Reward_history, eps_history = [], []
avg_scores = []
for epi in range( episodes ):
  state = env.reset()
  score = 0
  done = False
  step_cnt = 0
  while not done:
    #if epi > 100: env.render()
    # create and store agent experince from interacting with environment
    action = agent.select_action( state )
    next_state, reward, done, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)

    agent.learn(64) # do some learing if our buffer is full

    score += reward
    state = next_state
    step_cnt += 1
  eps_history.append(agent.epsilon)
  Reward_history.append(score)
  current_avg_score = np.mean(Reward_history[-100:]) # moving average of last 100 episodes
  avg_scores.append( current_avg_score )

  print('Episode: {}, Score: {}, Avg score: {}, '.format(epi, score,current_avg_score))

  agent.update_epsilon()    # update epsilon value after each episode 
  agent.update_target_net() # update target network after each episode

from bokeh.plotting import figure, show
# create a new plot with a title and axis labels
p = figure(title="Running average of last 100 rewards on CartPole-v1", x_axis_label="Episode", y_axis_label="Rewards")
# add a line renderer with legend and line thickness
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="Rewards", line_color="blue", line_width=2)
p.line(x, eps_history, legend_label="Epsilon", line_color="red",  line_width=2)

show(p) # show the results
