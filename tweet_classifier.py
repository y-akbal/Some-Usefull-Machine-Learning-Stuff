# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:25:35 2021

@author: Au
"""

import os
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import numpy as np





def return_embeddings(file):
    embeddings_ = dict()
    values = []
    with open(file, encoding ="utf8") as f:
        data = f.readlines()
        for i, line in enumerate(data):
            embedding = line.split()
            embeddings_[embedding[0]] = i
            values.append([float(j) for j in embedding[1:]])
    return embeddings_, values
        
embeddings, values = return_embeddings("glove.6B.50d.txt") ### your current directory should containg glove embeddings



import nltk
from nltk.corpus import twitter_samples  ## below we download tweet

n_tweets = twitter_samples.strings("negative_tweets.json")
p_tweets = twitter_samples.strings("positive_tweets.json")


def preprocess_tweet(tweet):
  
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=False,
                               reduce_len=False)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    return tweet_tokens

def return_vectorized_tweet(tweet, embeddings = embeddings, values = values, dims = 50):  
    """
    Parameters
    ----------
    tweet : preprocessed tweet.
    embeddings : dictionary returning indices of embeddings
    values : embedded vectors, here indices are to be taken from embeddings
    dims : embedding dimension (come from values)

    Returns
    -------
    vectorized tweet

    """
    vectorized_tweet = []
    preprocessed_tweet = preprocess_tweet(tweet)
    if len(preprocessed_tweet) == 0:
        vectorized_tweet.append(dims*[1])
        return vectorized_tweet
    else:
        for token in preprocessed_tweet:
            try:
                indice = embeddings[token]
                vectorized_tweet.append(values[indice])
            except Exception:
                vectorized_tweet.append(dims*[1])
    
        return vectorized_tweet  ### this dude vectorizes given preprocessed tweet, if the preprocessed tweet it empty it returns a line of ones



def return_dataset(negative_tweets = n_tweets, positive_tweets = p_tweets, padding_len = 10, dim = 50):
    """
    Parameters
    ----------
    negative_tweets : list of negative (embedded and padded) tweets 
    positive_tweets :list of positive (embedded and padded) tweets 
    padding_len : Padding length (20 works fine most cases)
    dim : embedding dimension 
    Returns
    -------
    X : padded tweets (this dude is an ordinary list)
    y : labels of tweets.

    """
    y = [1 for i in range(len(positive_tweets))]+[0 for i in range(len(negative_tweets))]
    X = []
    for tweet in positive_tweets:
        vec_tweet = return_vectorized_tweet(tweet)
        if len(vec_tweet) <= padding_len:
            X.append(vec_tweet + (padding_len - len(vec_tweet))*[[0 for i in range(dim)]])
        else:
            X.append(vec_tweet[:padding_len])
    for tweet in negative_tweets:
        vec_tweet = return_vectorized_tweet(tweet)
        if len(vec_tweet) <= padding_len:
            X.append(vec_tweet + (padding_len - len(vec_tweet))*[[0 for i in range(dim)]])
        else:
            X.append(np.array(vec_tweet[:padding_len]))
    return X,y


""" some testing below with  cross validation"""

X,y = return_dataset()
X = np.array(X)
X = X.mean(axis = 1) ## in the case that your classifier accepts 2d data you can skip this step

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
grid = {"n_neighbors":range(2,25), "weights":["uniform", "distance"]}
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
y_pred = knn.predict(X_test)

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(n_jobs = -1, estimator = knn, param_grid = grid, )
cv.fit(X_train, y_train)
cv.score(X_test, y_test)



from sklearn.svm import SVC
svc = SVC()
grid = {"C":np.arange(1,2,0.1), "kernel": ['linear', 'poly', 'rbf', 'sigmoid'], "degree":range(1,5)}
cv = GridSearchCV(n_jobs = -1, estimator = svc, param_grid = grid)
cv.fit(X_train, y_train)
A = pd.DataFrame(cv.cv_results_)
cv.score(X_test, y_test)

""" based on the best option test a tweet"""
def tweet_test(tweet, classifier, return_boolean = True):
    vec_tweet = np.array(return_vectorized_tweet(tweet)).mean(axis =0)
    vec_tweet = vec_tweet.reshape(1,-1)
    result = classifier.predict(vec_tweet)
    return result



