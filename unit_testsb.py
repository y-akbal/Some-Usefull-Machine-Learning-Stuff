import numpy as np

### Do not touch any part of the below code ####


class uncle_billy:
    def __init__(self):
        self.__q__ = np.random.uniform(0.3, 0.7)
        self.__p__ = np.random.uniform(0.3, 0.7)
        self.__w__ = np.random.uniform(0.3, 0.7)
        self.__h__ = np.random.uniform(0.3, 0.7)
        self.__l__ = np.random.uniform(0.3, 0.7)
        self.__n__ = np.random.uniform(0.3, 0.7)
        self.yakisikliserdar = np.random.uniform(0.3, 0.7)
        self.ezelbayraktar = np.random.uniform(0.3, 0.7)
        self.dayii = np.random.uniform(0.3, 0.7)
        self.F = ["H", "T"]
    
    @property
    def p(self):
        return self.__p__
    @p.getter
    def p(self):
        print("HAHAHAHHA come on bro, can't get this!!!!!")
    @p.setter
    def p(self, x):
        print("HAHAHAHHA come on bro, can't touch this")

    
    def flip_once(self):
        return np.random.choice(self.F, p = [self.__p__, 1-self.__p__])        
    


class unit_test:
    def __init__(self):
        self.f = [lambda x : x**2, lambda x: np.sin(x**2), 
                  lambda x: np.cos(x**2), lambda x: x*np.sin(x)]
        self.pts = [1, np.pi/2, np.pi/2, np.pi]
        
    def __check__(self):
        M = 0
        self.res = self.result_num - np.array([2, -2.45425, -1.961189, -3.14159])
        
        norm = np.linalg.norm(self.res)
        if not self.result_billy:
            M += 1
        if np.isclose(norm, 0, atol = 0.1):
            M += 1
        
        if M == 0:
            np.random.seed(10)
            return np.random.randint(100, 500)
        if M == 1:
            np.random.seed(25)
            return  np.random.randint(100, 500)
        if M == 2:            
            np.random.seed(15)
            return np.random.randint(100, 500)
        
        
    def __call__(self, play_with_billy, numerical_derivative):
        
        self.result_billy = play_with_billy(play_times = 10000)
        self.result_num = [numerical_derivative(f, pts) for f, pts in zip(self.f, self.pts)]
        
        L = self.__check__()
        
        
        
        print(F"Your token is {L}. DO NOT FORGET TO SUBMIT YOUR TOKEN, AS THIS MAY CAUSE DEGRADATION OF YOUR GRADE!!!!")
        


if __name__ == '__main__':
    print("OK Computer!")



