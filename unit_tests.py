import numpy as np

### Do not touch any part of the below code ####

"""
Some global variables    
"""
True_tokens = 235, 512, 23
False_tokens = 123, 233, 114
real_tokens = 23123, 23345, 114123
fake_tokens = 223, 233, 1123
reyiz_tokens = 223, 233, 1123


def uncle_sam_(tokenized):
    vocab_ = dict()
    for tweet in tokenized:
        for token in tweet:
            try:
                vocab_[token] += 1
            except Exception:
                vocab_[token] = 1
    return vocab_            
    

    
def uncle_sam__(tokenized, n_grams = 2, beginning = "<s>", end = "<e>"):
    n_grams_ = []
    beginning = [beginning]
    end = [end]
    for tweet in tokenized:
        tweet_ = (n_grams -1)*beginning + tweet + end
        for i in range(len(tweet_)-1):
            p_tweet = tweet_[i:i+n_grams]
            if len(p_tweet) >= n_grams:
                n_grams_.append(p_tweet) 
    return n_grams_


def __unit_tests__(n_grams_):
    n_grams_dict = dict()
    for gram in n_grams_:
        gram = tuple(gram)
        try:
            n_grams_dict[gram] += 1
        except Exception:
            n_grams_dict[gram] = 1
    return n_grams_dict

"""
n_grams = get_n_grams(train_tweets, n_grams = 2)            
n_1_grams = get_n_grams(train_tweets, n_grams =1)
n_1_grams_dict = count_n_grams(n_1_grams)
n_grams_dict = count_n_grams(n_grams)

from math import log

from collections import defaultdict

def auto_correction(n_grams_dict = n_grams_dict, n_1_grams_dict = n_1_grams_dict):
    n_grams_probs = dict()
    n_grams_best_probs = defaultdict(list)
    for grams in n_grams_dict.keys():
        try:
            n_grams_probs[grams] = log(n_grams_dict[grams]/n_1_grams_dict[grams[:-1]])
            n_grams_best_probs[grams[:-1]].append([log(n_grams_dict[grams]/n_1_grams_dict[grams[:-1]]), grams[-1]])
        except Exception:
            pass
    
    return n_grams_probs, n_grams_best_probs

probs, best_probs = auto_correction()    

best_probs[("sdfadsfa",)]

def return_completion(string):
    word = ""
    prob = -10000000000
    for tuple_ in best_probs[(string,)]:
        if tuple_[0] >= prob:
            prob = tuple_[0]
            word = tuple_[1]
    return word

"""


class reyiz:
    def __init__(self, uncle_same_):
        self.reyiz = uncle_same_
    def forward(self):
        print("Hoppa")

class interval:
    def __init__(self, a = -np.inf, b = np.inf):
        
        self.a = a
        self.b = b
        assert self.a < self.b 
        
    def __contains__(self, x):
        return x > self.a and x < self.b
 


def bubble_sort(list__, return_args = False):
    list_ = [el for el in list__]
    args = [i for i in range(len(list_))]
    le = len(list_)
    J = 0
    T = True
    while T:
        for i in range(le-1):
            if list_[i] > list_[i+1]:
                list_[i],list_[i+1] = list_[i+1], list_[i]
                args[i], args[i+1] = args[i+1], args[i]
                J = 0
            else:
                J += 1
            if J == le:
                T = False
    if not return_args:                
        return list_
    else:
        return list_, args



class uncle__sam:
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
    



def distancebetween_sams(s, g, del_cost = 1, insert_cost = 1, replace_cost = 2, return_distance_matrix = False):

    s = f"\t" + s
    g = f"\t" + g
    l_s = len(s)
    l_g = len(g)
    D = np.zeros((l_s,l_g), dtype = np.dtype(int))
    replace = lambda x,y: replace_cost if x != y else 0
    D[:,0] = [i for i in range(len(s))]
    D[0,:] = [i for i in range(len(g))]
    for i in range(1, len(s)):
        for j in range(1, len(g)):
            D[i,j] = min(D[i,j-1]+1, D[i-1,j]+1, D[i-1,j-1]+replace(s[i], g[j]))
    if not return_distance_matrix:
        return D[-1,-1]
    else:
        return D, D[-1,-1]



class unit_tests:
    def __init__(self, uncle_sam = uncle_sam):
        self.f = [lambda x : x**2, lambda x: np.sin(x**2), 
                  lambda x: np.cos(x**2), lambda x: x*np.sin(x)]
        self.pts = [1, np.pi/2, np.pi/2, np.pi]
        self.uncle_same = uncle_sam
        
    def __check__(self):
        M = 0
        res = self.result_num - np.array([2, -2.45425, -1.961189, -3.14159])
        norm = np.linalg.norm(res)
        if not self.result_billy:
            M += 1
        if np.isclose(norm, 0, atol = 0.1):
            M += 1
        
        if M == 0:
            np.random.seed(10)
            return np.random.randint(100, 500)
        elif M == 1:
            np.random.seed(15)
            return  np.random.randint(100, 500)
        elif M == 2:            
            np.random.seed(25)
            return np.random.randint(100, 500)
        
        
    def __call__(self, play_with_billy, numerical_derivative):
        
        self.result_billy = play_with_billy(play_times = 10000)
        self.result_num = [numerical_derivative(f, pts) for f, pts in zip(self.f, self.pts)]
        
        L = self.__check__()
        
        
        
        print(F"Your token is {L}. DO NOT FORGET TO SUBMIT YOUR TOKEN, AS THIS MAY CAUSE DEGRADATION OF YOUR GRADE!!!!")
        


def true_answers(s, g, del_cost = 1, insert_cost = 1, replace_cost = 2, return_distance_matrix = False):



    True_tokens = 235, 512, 23
    False_tokens = 123, 233, 114
    s = f"\t" + s
    g = f"\t" + g
    l_s = len(s)
    l_g = len(g)
    D = np.zeros((l_s,l_g), dtype = np.dtype(int))
    replace = lambda x,y: replace_cost if x != y else 0
    D[:,0] = [i for i in range(len(s))]
    D[0,:] = [i for i in range(len(g))]
    for i in range(1, len(s)):
        for j in range(1, len(g)):
            D[i,j] = min(D[i,j-1]+1, D[i-1,j]+1, D[i-1,j-1]+replace(s[i], g[j]))
    if not return_distance_matrix:
        return D[-1,-1]
    else:
        return D, D[-1,-1]


        
if __name__ == '__main__':
    print("OK Computer!")



