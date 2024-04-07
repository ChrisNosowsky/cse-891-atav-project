import numpy as np

class OU:
    def __init__(self, x, mu, theta, sigma):
        self.x = x
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
    def call_func(self):
        return self.theta * (self.mu - self.x) + self.sigma * np.random.randn(1)