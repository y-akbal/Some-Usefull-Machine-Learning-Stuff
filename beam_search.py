# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 21:40:05 2022

@author: hamsi
"""

### Beam search function

import numpy as np



def f(L_probs, L_tokens, next_probs, B):
    """
    Parameters
    ----------
    L_tokens : This is the batch of tokens of size (B, length). (numpy array)
    L_probs: Probs of tokens in L (of size (B, 1)). (numpy array)
    next_probs: Probs of next words given L_tokens of size (B,vocab_size)
    B : Beam size
    
    Outputs
    New_tokens (en extra column added of size (B, length+1)), probs of the sentencens (B of'm)
    
    """
    Probs = L_probs*next_probs
    f_Probs = Probs.reshape(-1,)
    vocab_size = len(next_probs[0,:])
    
    sort = np.argsort(f_Probs)[-B:]
    
    max_prob_rows = []
    max_prob_tokens = []
    for args in sort:
        max_prob_rows.append(args //vocab_size)
        max_prob_tokens.append(args % vocab_size)
    
    reshaped_max_prob_tokens = np.array(max_prob_tokens).reshape(-1,1)
    
    New_tokens = np.hstack([L_tokens[max_prob_rows], reshaped_max_prob_tokens])
    
    return New_tokens, f_Probs[sort].reshape(-1,1)


### Some test....
B = 150 ###beam length
L_tokens = np.random.randint(0,100, size =(B, 1))
vocab_size = 250

L_probs = np.random.uniform(size= B).reshape(-1,1)
next_probs = np.random.uniform(size= (B,vocab_size))

L_tokens , L_probs = f(L_probs, L_tokens, next_probs, B)





    
    
                

    
        
    
