import gym
import pybullet_envs
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
  def __init__(self, size=50000):
    self.memory = collections.deque(maxlen=size)

  def __len__(self): return len(self.memory)

  def store(self, experiance):
    self.memory.append( experiance )

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states  = torch.tensor([x[0] for x in batch], dtype=torch.float).to('cpu')
    actions = torch.tensor([x[1] for x in batch], dtype=torch.float).to('cpu')
    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float).to('cpu')
    nstates = torch.tensor([x[3] for x in batch], dtype=torch.float).to('cpu')
    dones   = torch.tensor([1-int(x[4]) for x in batch]).to('cpu')
    return states, actions, rewards, nstates, dones

class Actor(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0], 256) # Better result with slightly wider networks.
    self.fc2 = nn.Linear(256, 128)
    self.fc_mean = nn.Linear(128, out_space.shape[0])
    #self.apply(layer_init)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    mean = self.fc_mean(x)
    return torch.tanh( mean ) 

class Critic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0005):
    super(Critic, self).__init__()
    self.fc1 = nn.Linear(in_space.shape[0]+out_space.shape[0], 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s, a], dim=1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class TD3:
  def __init__(self, in_space, out_space):

    self.tau    = 0.005 # for target network soft update
    self.gamma  = 0.90  # discount
    self.memory = ReplayBuffer(size=1000000)

    self.noise = 0.1 # exploration noise
    self.max_act = (in_space.high)[0]
    self.min_act = (in_space.low)[0]

    self.delay_step    = 1   # Denis Yarats' implementation delays this by 2.
    self.target_update = 1

    self.actor = Actor(in_space, out_space, lr=3e-4).to('cpu')
    self.critic1 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.critic2 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.targ_critic1 = Critic(in_space, out_space, lr=1e-3).to('cpu')
    self.targ_critic2 = Critic(in_space, out_space, lr=1e-3).to('cpu')

    self.targ_critic1.load_state_dict(self.critic1.state_dict())
    self.targ_critic2.load_state_dict(self.critic2.state_dict())

    self.loss_fn = nn.MSELoss()

  def store(self, exp): 
    self.memory.store((exp))

  def get_action(self, obs):
    obs = torch.from_numpy(obs).float()
    mu = self.actor(obs) 
    mu_prime = mu + torch.tensor(np.random.normal(scale=self.noise), dtype=torch.float).to('cpu')
    mu_prime = torch.clamp(mu_prime, self.min_act, self.max_act)
    return mu_prime.cpu().detach().numpy()

  def train(self, batch_size=256):
    if len(self.memory) >= learning_start:
      states, actions, rewards, nstates, dones = self.memory.sample(batch_size)
    pass

time_step = 0          # global time step logger
learning_start = 5e3   # timestep to start learning

#env = gym.make('Pendulum-v0')
#env = gym.make('InvertedPendulumBulletEnv-v0')
env = gym.make('HopperBulletEnv-v0')
env = gym.wrappers.RecordEpisodeStatistics(env)

agent = TD3( env.observation_space, env.action_space )

#env.render()
scores = []
for epi in range(2000):
  obs = env.reset()
  while True:
    #if time_step < learning_start:
    #  action = env.action_space.sample()
    #else: 
    #  action = agent.get_action(obs)
    action = agent.get_action(obs)
    #action = env.action_space.sample()

    _obs, reward, done, info = env.step(action)
    #agent.store((obs, action, reward, _obs, done))

    #agent.train()
    obs = _obs
    time_step+=1
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      #print(f"Episode: {epi}, Time step: {time_step}, Return: {scores[-1]}, Avg return: {avg_score}")
      if epi % 10 ==0:
        print(f"global_step={time_step}, episode_reward={int(info['episode']['r'])}")
      break
env.close()
