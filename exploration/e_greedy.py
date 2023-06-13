import math

class E_Greedy_Exploration:
    def __init__(self, parameters):
        self.parameters = parameters

    def explore_probability(self, timesteps):
        return self.parameters["e_end"] + (self.parameters["e_start"]-self.parameters["e_end"])*math.exp(-1*timesteps/self.parameters["max_timesteps"])

