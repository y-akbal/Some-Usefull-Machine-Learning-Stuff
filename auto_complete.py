# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 22:51:38 2021

@author: hamsi
"""

import os
os.chdir("C:/Users/hamsi/Dropbox/Datascience/Natural_Language_Processing/Practice/auto_correction")

import nltk
from nltk import tokenize
tokenizer = tokenize.TweetTokenizer(preserve_case= False)

def return_tweets(file):
    tweets = []
    with open(file, "r", encoding = "utf-8") as f:
        data = f.readlines()
        for tweet in data:
            tweets.append(tokenizer.tokenize(tweet))    
    return tweets

os.listdir()
tokenized_tweets = return_tweets("en_US.twitter.txt")         

import numpy
numpy.random.shuffle(tokenized_tweets)

test_tweets = tokenized_tweets[int(0.8*len(tokenized_tweets)):]
train_tweets = tokenized_tweets[:int(0.8*len(tokenized_tweets))]

def get_vocab(tokenized_tweets):
    vocab_ = dict()
    for tweet in tokenized_tweets:
        for token in tweet:
            try:
                vocab_[token] += 1
            except Exception:
                vocab_[token] = 1
    return vocab_            
    
vocab_dict = get_vocab(train_tweets)                
    
def get_n_grams(tokenized_tweets, n_grams = 2, beginning = "<s>", end = "<e>"):
    n_grams_ = []
    beginning = [beginning]
    end = [end]
    for tweet in tokenized_tweets:
        tweet_ = (n_grams -1)*beginning + tweet + end
        for i in range(len(tweet_)-1):
            p_tweet = tweet_[i:i+n_grams]
            if len(p_tweet) >= n_grams:
                n_grams_.append(p_tweet) 
    return n_grams_


def count_n_grams(n_grams_):
    n_grams_dict = dict()
    for gram in n_grams_:
        gram = tuple(gram)
        try:
            n_grams_dict[gram] += 1
        except Exception:
            n_grams_dict[gram] = 1
    return n_grams_dict


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




        
    
    

