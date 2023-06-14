import gymnasium as gym
from policies.Deep_Q_Learning import dql
from feature_extractors import mlp
import torch
from memory import experience_replay
from exploration import e_greedy

#Define your environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")
env_parameters = {"action_space":3, "obs_space":2}

#Initiliase the exprience replay buffer (required for Q-Learning) and the agent
experience_replay_memory = experience_replay.Memory(1000000)

#
exploration_parameters = {
    "e_start": 1,
    "e_end": 0.05,
    "max_timesteps": 100000,
}
exploration_func = e_greedy.E_Greedy_Exploration(exploration_parameters)

model = mlp.MLP([(env_parameters["obs_space"],64), (64,64), (64, env_parameters["action_space"])])

hyperparameters = {
    "discount_factor":0.99, 
    "learning_rate":0.0001,
    "batch_size":64
}

loss_function = torch.nn.MSELoss()
optimisation_function = torch.optim.SGD(model.parameters(), hyperparameters["learning_rate"])
agent = dql.DQNAgent(env, env_parameters, exploration_func, model, hyperparameters, loss_function, optimisation_function)

#learn
agent.run(experience_replay_memory, num_episodes=1000, num_timesteps=1000)
  

