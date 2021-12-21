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
    #obs  = torch.tensor([obs], dtype=torch.float).to('cpu')
    obs    = torch.from_numpy(obs)
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
#env=gym.make('CartPole-v0')
#env=gym.make('MountainCar-v0')
#env=gym.make('LunarLander-v2')

#env=gym.make('BipedalWalkerHardcore-v3')
env=gym.make('BipedalWalker-v3')
#env = gym.make('CarRacing-v0')

input_size  = env.observation_space.shape[0]
output_size = env.action_space.n
#output_size = env.action_space.shape[0]


# Environment Hyperparameters
max_ep_len = 400                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps
update_timestep = max_ep_len * 4    # update policy every n timesteps
print_freq = max_ep_len * 4         # print avg reward in the interval (in num timesteps)

print_running_reward   = 0
print_running_episodes = 0

time_step = 0
i_episode = 0

agent = PPO(input_size, output_size, size=update_timestep, bsize=1)

scores, avg_scores = [], []
while time_step <= max_training_timesteps:
  state = env.reset()
  score = 0
  for t in range(1, max_ep_len+1):

    action, probs, val = agent.selectAction(state)
    state, reward, done, _ = env.step(action)
    agent.store(state, action, reward, probs, val, done)

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


  
from bokeh.plotting import figure, show
p = figure(title="(BipedalWalkerHardcore-v3) Running average of past 100 games", x_axis_label="iterations", y_axis_label="Scores")
x = np.arange(len(avg_scores))
p.line(x, avg_scores,  legend_label="scores", line_color="blue", line_width=2)
show(p) 
