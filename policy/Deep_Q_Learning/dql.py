import torch
import gymnasium as gym
import numpy as np
import random
from collections import deque
from feature_extractors import mlp
from torchinfo import summary
from tqdm import tqdm
import math
from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

def save_frames_as_gif(frames, path='./', filename='top_of_the_world.gif'):

    #A utility function to save the succesful episodes as gifs. 
    # Taken from: 

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

class Memory():
    def __init__(self, size):
        self.memory = deque([], maxlen=size)

    def store(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class DQNAgent():
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def predict(self, state):
        return self.model(state)
    
    def step(self, actual_discounted_reward, predicted_discounted_reward, loss_func, optim):
        optim.zero_grad()
        loss = loss_func(predicted_discounted_reward, actual_discounted_reward)
        loss.backward()
        optim.step()

    def run(self,memory, num_episodes, num_timesteps):
        
        terminal = 0
        loss_func = torch.nn.MSELoss()
        optim = torch.optim.SGD(self.model.parameters(), 0.001)
        for i in tqdm(range(num_episodes)):
            frames = []
            state, __ = self.env.reset()
            steps_done = 0
            c_reward = 0
            for t in range(num_timesteps):
                eps_threshold = 0.05 + 0.85*math.exp(-1*steps_done/50000)
                steps_done += 1
                if random.random()>eps_threshold:
                    action = torch.argmax(self.predict(torch.tensor(state))).item()
                else:
                    action = np.random.randint(0,3)
                next_state, reward, terminal, __, __ = self.env.step(action)
                frames.append(self.env.render())
                reward += 0.2
                if state[0]>0.3 and state[1]>0.05:
                    reward += 10
                if next_state[1]!=0:
                    reward += 0.2
                if next_state[1]>0.04 or next_state[1]<-0.04:
                    reward += 0.6
                if next_state[0]>=0.5:
                    reward += 500
                    save_frames_as_gif(frames)
                    print("Reached!!")
                    break
                

                one_hot_action = [0, 0, 0]
                one_hot_action[action] = 1
                transition = []
                transition.extend(state.tolist())
                transition.extend(next_state.tolist())
                transition.extend(one_hot_action)
                transition.extend([reward, int(terminal)])
                memory.store(transition)
                
                if len(memory.memory)>1024:
                    sample = torch.tensor(memory.sample(1024),dtype=torch.float32)
                    states = sample[:,:2]
                    next_states = sample[:,2:4]
                    actions = torch.tensor(sample[:,4:7],dtype=torch.float32)   
                    rewards = sample[:,7]
                    terminals = sample[:,8]

                    predicted_discounted_reward = torch.sum(self.predict(states)*actions, dim=1)
                    actual_discounted_reward = rewards+0.99*torch.max(self.predict(next_states),dim=1)[0]*terminals
                    self.step(actual_discounted_reward, predicted_discounted_reward, loss_func, optim)                           
                    c_reward += reward
    
        return
