# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:32:01 2021

@author: hamsi
"""



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

        
    
    
    
    
    
    