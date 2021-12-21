# TODO: fix log_probs
# https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/actor_critic/actor_critic_continuous.py

# continuous Actor Critic
import gym
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
  def __init__(self, in_dims, out_dims):
    super(ActorCritic, self).__init__()
    self.lr    = 0.0003
    self.gamma = 0.99
    self.memory = [] # memory

    self.fc1 = nn.Linear(in_dims, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fcp = nn.Linear(64, out_dims)
    self.fcv = nn.Linear(64, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    self.to('cpu')

  def forward(self, obs):
    z = F.relu( self.fc1(obs) )
    z = F.relu( self.fc2(z) )
    value  = self.fcv(z) 
    policy = self.fcp(z)
    return policy, value

  def choose_action(self, obs):
    pi, v = self.forward(obs)
    mean, std = pi, torch.exp(v)                   # create distribution varibles
    dist = torch.distributions.Normal( mean, std ) # we create a Normal distribution pi, v
    probs  = dist.sample()                         # sample from distribution
    action = torch.tanh(probs)                     # normalize actions
    log_probs = dist.log_prob(probs).to('cpu')
    
    return action, probs

  # state, reward, nstate, log_prbs, done
  def store(self, s, r, ns, p, d):
    self.memory.append((s, r, ns, p, d))

  def train(self):
    states  = torch.tensor([x[0] for x in self.memory], dtype=torch.float)
    rewards = torch.tensor([x[1] for x in self.memory], dtype=torch.float)
    nstates = torch.tensor([x[2] for x in self.memory], dtype=torch.float)
    log_probs = torch.tensor([x[3] for x in self.memory], dtype=torch.float)
    #log_probs = torch.tensor([x[3].detach().item() for x in self.memory])
    dones   = torch.tensor([1-int(x[4]) for x in self.memory], dtype=torch.float)
    self.memory= []

    _, value  = self.forward(states)
    _, nvalue = self.forward(nstates)

    delta = rewards + self.gamma*nvalue*done - value
    loss = -log_probs * delta  + delta**2
    #loss = 0

    self.optimizer.zero_grad() # clean gradients
    loss.backward()            # generate gradients to iterate backwards
    self.optimizer.step()      # iterated gradients backwards

# Global variables
#env=gym.make('BipedalWalkerHardcore-v3')
env=gym.make('BipedalWalker-v3')
#env = gym.make('CarRacing-v0')

input_size  = env.observation_space.shape[0]
output_size = env.action_space.shape[0]


# Environment Hyperparameters
max_ep_len = 400                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps
update_timestep = max_ep_len * 4    # update policy every n timesteps
print_freq = max_ep_len * 4         # print avg reward in the interval (in num timesteps)

print_running_reward   = 0
print_running_episodes = 0

time_step = 0
i_episode = 0

agent = ActorCritic(input_size, output_size)

scores, avg_scores = [], []
while time_step <= max_training_timesteps:
  state = env.reset()
  score = 0
  for t in range(1, max_ep_len+1):

    action, log_probs= agent.choose_action(torch.from_numpy(state))
    #print(action)
    #action = env.action_space.sample() 
    #print(action)

    nstate, reward, done, _ = env.step(action)
    agent.store(state, reward, nstate, log_probs, done)

    time_step +=1
    score += reward

    # train PPO agent
    if time_step % update_timestep == 0: agent.train()

    # printing average reward
    if time_step % print_freq == 0:
      # print average reward till last episode
      print_avg_reward = print_running_reward / print_running_episodes
      print_avg_reward = round(print_avg_reward, 2)

      print("Episode : {}  Timestep : {}  Average Reward : {}".format(i_episode, time_step, print_avg_reward))

      print_running_reward = 0
      print_running_episodes = 0
            
    # break; if the episode is over
    if done: break

  scores.append(score)
  avg_scores.append(  np.mean(scores[-100:]) )

  print_running_reward += score
  print_running_episodes += 1
  i_episode += 1

env.close()


