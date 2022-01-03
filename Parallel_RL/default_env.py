'''
Multi environment adapted from 
   https://github.com/vwxyzjn/PPO-Implementation-Deep-Dive/blob/master/ppo.py#L132
'''
import gym
import time
###########################################################################################
# simple single environment 

env = gym.make("CartPole-v1")
env = gym.wrappers.RecordEpisodeStatistics(env)
#env = gym.wrappers.RecordVideo(env, "videos", record_video_trigger=lambda t: t %100==0)
#env = gym.wrappers.RecordVideo(env, f"videos/fish")
obs = env.reset()
for x in range(200):
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  if done:  
    obs = env.reset()
    print(f"Episodic return: {info['episode']['r']}")
    score = 0
env.close()

###########################################################################################
# multi environment 

""" 
 - make vector environment beacuse PPO deals with best with vector environments?
 - multiple environments
"""

def make_env(gym_id, idx, seed, record=False, run_name=''):
  def thunk():
    env = gym.make(gym_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if record and idx==0:
      env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    # seeds for reproductivity
    env.seed(seed)                   
    env.action_space.seed(seed)      
    env.observation_space.seed(seed) 
    return env
  return thunk
  
    
gym_id = "CartPole-v1"
num_envs = 3
seed     = 1

envs = gym.vector.SyncVectorEnv([
      make_env(gym_id, i, seed+i, record=False, run_name=f"{gym_id}__{int(time.time())}" ) 
          for i in range(num_envs)
      ])

obs = envs.reset()
for x in range(200):
  action = envs.action_space.sample()
  obs, reward, done, info = envs.step(action)
  for item in info:
    if "episode" in item.keys():
      print(f"Episodic return: {item['episode']['r']}")
      # no need to obs = env.rest as gym vector does it automatically

