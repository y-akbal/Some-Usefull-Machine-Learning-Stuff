import numpy as np
class interval:
    def __init__(self, a = -np.inf, b = np.inf):
        
        self.a = a
        self.b = b
        assert self.a < self.b 
        self.isolated_points = []
    def __contains__(self, x):
        return x > self.a and x < self.b
 
