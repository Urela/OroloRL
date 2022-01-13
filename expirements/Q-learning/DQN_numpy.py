# FINISH ADD LOSS calulateion, update using sword
import gym
import numpy as np
from collections import deque
import warnings

#### Miscellaneous functions
class ANN:
  def __init__(self, size):
    self.params, self.cache, self.grads = {}, {}, {}
    self.size, self.num = size, len(size)

    self.params["b"] = [ np.random.randn(y, 1).astype(np.float32)*0.01 for y in size[1:] ]
    self.params["w"] = [ np.random.randn(y, x).astype(np.float32)*0.01 for x,y in zip(size[:-1], size[1:]) ]

    # gradients for learning
    self.grads["b"]  = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"]  = [ np.zeros(w.shape) for w in self.params['w'] ]

  def actFunc(self, z, deriv=False):
    ## relu
    if deriv: return 1. * (z > 0)
    return np.maximum(z, 0)

    ## sigmoid
    #sigmoid = 1.0/(1.0+np.exp(-z))
    #if deriv: return sigmoid * (1-sigmoid)
    #return sigmoid

  def forward(self, image):
    for w, b in zip(self.params["w"], self.params["b"]):
      image = self.actFunc( np.matmul(w, image) + b )
    return image

  def backward(self, image, label):
    layer_vec = []     # each layer's z vector
    layer_act = []     # each layer's z vector after activation function
      
    # determinig the output of each layer
    layer_act.append( image )     # adding the input to update first layer
    for w, b in zip(self.params["w"], self.params["b"]):
      image = np.matmul(w, image) + b
      layer_vec.append(image)

      image = self.actFunc(image) 
      layer_act.append(image)

    loss = np.argmax(layer_act[-1]) - np.argmax(label) 
    dout = np.argmax(layer_act[-1]) - np.argmax(label) 
    delta = dout * self.actFunc(layer_vec[-1], deriv=True)  
    self.grads["b"][-1] += delta
    self.grads["w"][-1] += np.dot(delta, layer_act[-2].T)
    # determing HIDDEN neurons weights and bias
    for l in range(2, self.num):
      delta = np.matmul(self.params["w"][-l+1].T, delta) * self.actFunc(layer_vec[-l], deriv=True)  #  delta * dz/da * input 
      self.grads["b"][-l] += delta
      self.grads["w"][-l] += np.matmul(delta, layer_act[-l-1].T)
    return loss

  def optimize(self, batch_size, lr=0.8):
    #### SGD update rule
    self.params["b"] = [b-(lr/batch_size) * nb for b, nb in zip(self.params["b"], self.grads["b"])]
    self.params["w"] = [w-(lr/batch_size) * nw for w, nw in zip(self.params["w"], self.grads["w"])]

    ### clean grads
    self.grads["b"] = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"] = [ np.zeros(w.shape) for w in self.params['w'] ]



class RLAgent:      # class representing a reinforcement learning agent
  def __init__(self, env=None):
    self.env = env
    self.topology = [env.observation_space.shape[0], 24, env.action_space.n]
    self.net = ANN( self.topology )
    self.gamma   = 0.95
    self.epsilon = 1.0
    self.memory = deque([],1000000)

  def forward(self, observation, remember_for_backprop=True):
    return self.net.forward( observation.T )

  def remember(self, done, action, observation, prev_obs):
    self.memory.append([done, action, observation, prev_obs])

  def select_action(self, observation):
    values = self.forward(np.asmatrix(observation))
    if (np.random.random() > self.epsilon):
      return np.argmax(values)
    else:
      return np.random.randint(self.env.action_space.n)

  def experience_replay(self, update_size=20):
    if (len(self.memory) < update_size): return
    else: 
      batch_indices = np.random.choice(len(self.memory), update_size)
      for idx in batch_indices:

        done, action, nstate, cstate = self.memory[idx]

        q_value     = self.net.forward( cstate )
        new_q_value = self.net.forward( nstate )

        tmp_q = np.copy(q_value)
        if done: tmp_q[action] = -1
        else:    tmp_q[action] = 1 + self.gamma*np.max(new_q_value)

        self.net.backward(cstate, tmp_q)    # generate delta weights
        self.net.optimize(1)                # update weights with SGD

        #self.net.forward( q_value )
        #self.net.forward( tmp_q )

    self.epsilon = self.epsilon if self.epsilon < 0.01 else self.epsilon*0.997
 
env=gym.make('CartPole-v1')

# Global variables
NUM_EPISODES = 10000
MAX_TIMESTEPS = 1000
model = RLAgent(env)

# The main program loop
for i_episode in range(NUM_EPISODES):
    observation = env.reset()
    # Iterating through time steps within an episode
    for t in range(MAX_TIMESTEPS):
        env.render()
        action = model.select_action(observation)
        prev_obs = observation
        observation, reward, done, info = env.step(action)
        # Keep a store of the agent's experiences
        model.remember(done, action, observation, prev_obs)
        model.experience_replay(20)
        # epsilon decay
        if done:
            # If the pole has tipped over, end this episode
            print('Episode {} ended after {} timesteps'.format(i_episode, t+1))
            #print(model.layers[0].lr)
            break
