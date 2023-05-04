import random
import numpy as np

class list_(list):
    def __init__(self, args):
        super().__init__(args)
    def drop(self, indices: list):
        for indice in indices:
            self.remove(indice)
    def choice(self, size:int, p:np.ndarray = None)-> list:
        if len(self)>= size:
            random_choice = np.random.choice(self, size = size, p = p, replace= False)
            self.drop(random_choice)
            return random_choice
        elif 0 < len(self)< size:
            random_choice = list(self).copy()
            self.drop(random_choice)
            return random_choice
        else:
            raise IndexError("Nothing left here bro!!!")
