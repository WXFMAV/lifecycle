# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
#    def __init__(self,action_dimension,mu=0, theta=0.005, sigma=0.025):
    def __init__(self,action_dimension,mu=0, theta=0.05, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.gamma = 0.0007
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.sigma += self.gamma * (0.0 - self.sigma)
        
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    n = 3600
    for i in range(n):
        states.append(ou.noise())

    with open('dat/noise.txt','w') as f:
        for j in range(n):
            line = np.array_str(states[j])
            print(line)
            f.write(line + '\n')

