import gymnasium as gym
from policy.Deep_Q_Learning import dql
from feature_extractors import mlp

#Define your environment
env = gym.make("MountainCar-v0", render_mode="rgb_array")

#Initiliase the exprience replay buffer (required for Q-Learning) and the agent
memory = dql.Memory(1000000)
agent = dql.DQNAgent(env, mlp.MLP([(2,64), (64,64), (64, 3)]))

#learn
agent.run(memory, num_episodes=10000, num_timesteps=5000)
  

