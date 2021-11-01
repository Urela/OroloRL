### FINISH learn function
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
  def __init__(self, inputs, outputs, lr=1e-3):
    super(ActorCritic, self).__init__()
    self.gamma = 0.98

    # using one network to share produce policy and value estimates
    self.fc1   = nn.Linear(inputs, 256)
    self.fc_pi = nn.Linear(256, outputs)   # policy output
    self.fc_v  = nn.Linear(256, 1)   # value output

    self.optimizer = optim.Adam(self.parameters(), lr=lr)
    self.device = torch.device('cpu')
    #self.to(self.device)

    self.memory = [] # memory

  def forward(self, state, softmax_dim = 0):
    x = F.relu( self.fc1(state) )
    pi = self.fc_pi(x)
    prob = F.softmax(pi, dim=softmax_dim)
    v  = self.fc_v(x)
    return prob, v

  def pi(self, x, softmax_dim = 0):
    x = F.relu(self.fc1(x))
    x = self.fc_pi(x)
    prob = F.softmax(x, dim=softmax_dim)
    return prob

  def v(self, x):
    x = F.relu(self.fc1(x))
    v = self.fc_v(x)
    return v


  
  def store_data(self, s,a,r,s_, done):
    self.memory.append( (s,a,r,s_,done) )

  def genBatch(self):
    states, actions, rewards, nstates, dones = [], [], [], [], []
    for (s,a,r,s_,done) in self.memory:
      states.append(s)
      actions.append([a])
      rewards.append([r/100.0])
      nstates.append(s_)
      done_mask = 0.0 if done else 1.0
      dones.append([done_mask])

    states  = torch.tensor(states,  dtype=torch.float)
    actions = torch.tensor(actions, dtype=torch.int64)
    nstates = torch.tensor(nstates, dtype=torch.float)
    rewards = torch.tensor(rewards, dtype=torch.float)
    dones = torch.tensor(dones, dtype=torch.float)

    self.memory= []
    return states, actions, rewards, nstates, dones 
  
  def learn(self):
    states, actions, rewards, nstates, dones = self.genBatch()

    # critic value as a TD error estimate
    TD_target = rewards + self.gamma*self.v(nstates)*dones 
    delta = TD_target - self.v(states)

    # actor value from log probility
    policy = self.pi(states ,softmax_dim=1) #get actions as a probabilty distribution
    #print( policy.shape, actions.shape)
    pi_a = policy.gather(1, actions) # get actions from policy

    loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(states), TD_target.detach())

    self.optimizer.zero_grad() # clean gradients
    loss.mean().backward()     # generate gradients to iterate backwards
    self.optimizer.step()      # iterated gradients backwards


# Global variables
env=gym.make('CartPole-v1')
input_size  = env.observation_space.shape[0]
output_size = env.action_space.n

model = ActorCritic(input_size, output_size)    

#Hyperparameters
num_rollout = 10
episodes = 10000

print_interval = 20
score = 0.0

scores = []
for epi in range(episodes):
  done = False
  state = env.reset()
  while not done:
    for t in range(num_rollout):
        #env.render()
        prob = model.pi(torch.from_numpy(state).float())
        #prob, _= model.forward(torch.from_numpy(state).float())
        m = Categorical(prob)
        action = m.sample().item()
        nstate, reward, done, info = env.step(action)

        model.store_data(state,action,reward,nstate,done)
        
        state = nstate
        score += reward
        if done: break
    model.learn()
      
  if epi%print_interval==0 and epi!=0:
      print("# of episode :{}, avg score : {:.1f}".format(epi, score/print_interval))
      scores.append(score/print_interval)
      score = 0.0
env.close()

#from bokeh.plotting import figure, show
#
## create a new plot with a title and axis labels
#p = figure(title="Simple line example", x_axis_label="x", y_axis_label="y")
#
## add a line renderer with legend and line thickness
#x = np.arange(len(scores))
#p.line(x, scores,  legend_label="scores", line_color="blue", line_width=2)
#
#show(p) # show the results
