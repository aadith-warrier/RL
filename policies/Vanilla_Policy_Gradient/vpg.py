import torch


class VPGAgent():
    def __init__(self, env, policy, value_func, memory, explore_func, hyperparameters):
        self.env = env
        self.policy = policy
        self.value_func = value_func
        self.memory = memory
        self.explore_func = explore_func
        self.hyperparameters = hyperparameters

    def collect_rollouts(self, timesteps):
        rollout = []
        state, __ = self.env.reset()
        for t in range(timesteps):
            action = torch.argmax(self.policy(state))
            next_state, reward, terminal, __, __ = self.env.step(action)

            transition = {"state":state,
                        "action":action,
                        "reward":reward,
                        "next_state":next_state,
                        "terminal": terminal,
                        "future_discounted_reward":0}
            
            rollout.append(transition)
        self.compute_rewards_to_go(rollout)
        self.memory.store(rollout)

    def compute_rewards_to_go(self, rollout):
        discounted_reward = 0
        for i in range(len(rollout)):
            discounted_reward = self.hyperparameters["disount_factor"]*discounted_reward + rollout[-i]["reward"]
            rollout[-i]["future_discounted_reward"] = discounted_reward


    def compute_adv_est():
        pass

    def run():
        #rollouts
        #rewards
        #adv est
        #step on policy
        #step on value function
        pass