import torch
import numpy as np
import random
from copy import deepcopy
import utility

class DQNAgent():
    def __init__(self, env, env_parameters, exploration_func, model, hyperparameters, loss_function, optimisation_function):
        self.env = env
        self.env_parameters = env_parameters
        self.exploration_func = exploration_func
        self.model = model  
        self.target_net = deepcopy(model)
        self.hyps = hyperparameters
        self.loss_func = loss_function
        self.optim_func = optimisation_function

    def predict(self, state):
        return self.model(state)
    
    def target_predict(self, state):
        return self.target_net(state)
    
    def step(self, actual_discounted_reward, predicted_discounted_reward):
        self.optim_func.zero_grad()
        loss = self.loss_func(predicted_discounted_reward, actual_discounted_reward)
        loss.backward()
        self.optim_func.step()
        return loss

    def get_reward(self, next_state, reward):
        reward += 0.2
        success = False
        if next_state[0]>0.3 and next_state[1]>0.05:
            reward += 10
        if abs(next_state[1])>0.04:
            reward += abs(next_state[1])
        if next_state[0]>=0.5:
            reward += 500
            success = True
        
        return reward, success

    def run(self, memory, num_episodes, num_timesteps):
        
        terminal = 0
        success = 0
        for e in range(num_episodes):
            frame_buffer = []
            state, __ = self.env.reset()
            steps_done = 0
            c_reward = 0
            for t in range(num_timesteps):
                if random.random()>self.exploration_func.explore_probability(t+e*num_timesteps):
                    action = torch.argmax(self.predict(torch.tensor(state))).item()
                else:
                    action = np.random.randint(0,3)

                next_state, reward, terminal, __, __ = self.env.step(action)
                frame_buffer.append(self.env.render())

                reward, success = self.get_reward(next_state, reward)

                if success:
                    c_reward += reward
                    success += 1
                    name = "dqn" + str(success)
                    utility.save_frames_as_gif(frame_buffer, path="policies/Deep_Q_Learning/success/"+name)
                    break               

                one_hot_action = [0, 0, 0]
                one_hot_action[action] = 1
                transition = []
                transition.extend(state.tolist())
                transition.extend(next_state.tolist())
                transition.extend(one_hot_action)
                transition.extend([reward, int(terminal)])
                memory.store(transition)       #------> Maybe make this a dict
                
                if len(memory.memory)>self.hyps["batch_size"]*100:
                    sample = torch.tensor(memory.sample(self.hyps["batch_size"]),dtype=torch.float32)

                    states = sample[:,:2]
                    next_states = sample[:,2:4]
                    actions = torch.tensor(sample[:,4:7],dtype=torch.float32)   
                    rewards = sample[:,7]
                    terminals = sample[:,8]

                    predicted_discounted_reward = torch.sum(self.predict(states)*actions, dim=1)
                    actual_discounted_reward = rewards+(self.hyps["discount_factor"]*torch.max(self.target_predict(next_states),dim=1)[0])*terminals
                    loss = self.step(actual_discounted_reward, predicted_discounted_reward)
                    if t%1000 == 0:
                        self.target_net = deepcopy(self.model)
                        
                steps_done += 1
                c_reward += reward
            try:
                print("Episode: ", e, "Episode reward:", c_reward, "Loss:", loss.item())
            except:
                pass

        return