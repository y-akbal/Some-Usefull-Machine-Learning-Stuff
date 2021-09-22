# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 09:11:32 2021

@author: Au
"""

import numpy as np
"""
returns distance between two words
"""
def distance(s, g, del_cost = 1, insert_cost = 1, replace_cost = 2, return_distance_matrix = False):
    """
returns distance between two words by means of linear programming
"""
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

