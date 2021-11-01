import gym
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

############# stores memory for experience replay ############# 

class ReplayBuffer():
  def __init__(self, MaxSize, state_space,  action_space):
    self.idx  = 0         # postion in memory stack
    self.size = MaxSize   # max size of memory stack
    self.states  = np.zeros((MaxSize, *state_space), dtype=np.float32)
    self.nstates = np.zeros((MaxSize, *state_space), dtype=np.float32)
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

  def sample(self, batch_size):
    csize = min(self.idx, self.size)    # current size of memory stack
    batch = np.random.choice(csize, batch_size) # sample batch
    
    states  = self.states[batch]  
    actions = self.actions[batch]   
    rewards = self.rewards[batch]   
    nstates = self.nstates[batch]   
    dones   = self.dones[batch]     
    return states, actions, rewards, nstates, dones

  def __len__(self): return self.idx

#buffer = ReplayBuffer(10, [8], 2)
#for i in range( 100 ):
#  buffer.store(i,i,i,i,False)
#
#states, actions, rewards, nstates, dones = buffer.sample( 3 ) 
#print( states, actions, rewards, nstates, dones )

############### Neural netowrk model ############### 
class ANN(nn.Module):
  def __init__(self, input_size, mid1_size, mid2_size, output_size, lr=1e-3):
    super(ANN, self).__init__()

    self.fc1 = nn.Linear(*input_size, mid1_size)
    self.fc2 = nn.Linear(mid1_size, mid2_size)
    self.fc3 = nn.Linear(mid2_size, output_size)

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.loss = nn.MSELoss()
    self.device = T.device('cpu')
    self.to(self.device)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    actions = self.fc3(x)
    return actions

# testing
#net = ANN([100], 50,24,10)

############### RL agent (wrapper class for neural network) ############### 
class DQN_Agent:
  def __init__(self, state_size, action_size, lr=1e-3):
    self.gamma   = 0.95
    self.epsilon = 1.0
    self.eps_min = 0.05
    self.eps_dec = 5e-4
    self.lr = lr
    self.action_space = [i for i in range(action_size)]

    # memory
    self.memory = ReplayBuffer(100000, state_size, action_size)

    # neural networks
    self.policy_net = ANN(state_size, 24, 24, action_size, lr) # policy network
    self.target_net = ANN(state_size, 24, 24, action_size, lr) # target network
    self.target_net.eval() # since no learning is performed on the target net

  def update_target_net(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())
  def update_epsilon(self):
    self.epsilon = max(self.eps_min, self.epsilon*self.eps_dec)

  def remember(self, state, action, reward, nstate, done):
    self.memory.store(state, action, reward, nstate, done)

  def select_action(self, state):
    if np.random.random() > self.epsilon:
      state   = T.tensor([state]).to(self.policy_net.device).float()
      actions = self.policy_net.forward(state)
      action  = T.argmax( actions ).item()
    else:
      action = np.random.choice( self.action_space)
    return action

  def learn(self, batch_size=1):
    if len(self.memory) < batch_size: return # if buffer not full don't learn 
    else:
      
      batch_index = np.arange(batch_size, dtype=np.int32)
      states, actions, rewards, nstates, terminals = self.memory.sample(batch_size)

      states  = T.from_numpy( states  ).to(self.policy_net.device).float()
      nstates = T.from_numpy( nstates ).to(self.policy_net.device).float()
      rewards = T.from_numpy( rewards ).to(self.policy_net.device).float()

      # want the action in a batch ??? not sure
      q_pred   = self.policy_net.forward( states )[batch_index, actions] 
      q_target = self.target_net.forward( nstates ).float()

      q_target[terminals] = 0.0  # set all terminal states' value to zero 

      # we want to get max action from the action column in our batch matrix
      q_target = rewards + self.gamma*T.max(q_target, dim=1)[0]

      self.policy_net.optimizer.zero_grad()   # clearing tempory gradients 
      loss = self.policy_net.loss(q_target, q_pred).to(self.policy_net.device)
      loss.backward()
      self.policy_net.optimizer.step()


# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n


episodes       = 10000
max_time_steps = 1000
update_frequency = 500
agent = DQN_Agent([input_size], output_size)

Reward_history, avg_scores, eps_history = [], [], []
for epi in range( episodes ):
  state = env.reset()
  score = 0
  done = False
  while not done:
    #if score > 1000: env.render()

    # create and store agent experince from interacting with environment
    action = agent.select_action( state )
    next_state, reward, done, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)

    agent.learn(64) # now do some learing if our buffer is full

    score += reward
    state = next_state

  eps_history.append(agent.epsilon)
  Reward_history.append(score)
  current_avg_score = np.mean(Reward_history[-100:]) # moving average of last 100 episodes
  avg_scores.append( current_avg_score )

  # If the pole has tipped over, end this episode
  #print('Episode {} ended after {} timesteps'.format(epi, t+1))
  #print('episode ', epi, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
  print('Episode {} ended after {} timesteps and Avg score {}'.format(epi, score, round( current_avg_score, 2)))

  agent.update_epsilon()    # update epsilon value after each episode 
  agent.update_target_net() # update target network after each episode


from bokeh.plotting import figure, show

# create a new plot with a title and axis labels
p = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")

# add a line renderer with legend and line thickness
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="Rewards", line_color="blue", line_width=2)
p.line(x, eps_history, legend_label="Epsilon", line_color="red",  line_width=2)

show(p) # show the results
