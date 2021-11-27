# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:37:29 2021

@author: hamsi
"""
import tensorflow as tf
from tensorflow.keras.activations import relu
import numpy as np


X = np.random.randn(1000, 1,1)
y = np.random.randn(1000, 1,1)




def loss(y_true, y_pred):  ## this dude is penalized loss function
    sq = tf.square(y_true-y_pred)
    return tf.math.reduce_mean(sq) + 0.001*float(tf.norm(y_pred) )

###poor mans regression stuff without subclassing

class ann:
    def __init__(self):
        self.weights_set = False  ### checking if weights initialized depending on the data
    def build(self, shape):
        W_i = tf.random_normal_initializer()
        b_i = tf.zeros_initializer()
        self.W = tf.Variable(W_i(shape), trainable= True)  ## initialize weights
        self.b = tf.Variable(b_i(1), trainable = True)
        self.weights_set = True
        
            
    
    
    def call(self, X):  ## for forward pass,
        if self.weights_set: ## if weights are already initialized, then do forwards pass
            
            return tf.matmul(self.W, X)+self.b
        else: ### else, first do the initialization then do forward pass
            x,y = X.shape[1:]
            self.build((y,x))
            
            return self.call(X)
            
        
        
    def fit(self, X,y, epochs = 10):
        
            for i in range(epochs):  ## take some gradients for updating the weights
                with tf.GradientTape(persistent = True) as tape:
                    y_pred = self.call(X)
                    loss = self.loss(y_pred, y)
                               
                grad_W = tape.gradient(loss, self.W)
                grad_b = tape.gradient(loss, self.b)
                self.W.assign_sub(self.lr*grad_W)          
                self.b.assign_sub(self.lr*grad_b)
                loss = self.loss(y_pred, y)
                print(f"{i}th epoch passed, the loss on training data is {loss}")
                del(tape)
                
    
    def compile(self, lr = 0.001, loss = relu):
        self.loss = loss
        self.lr = lr
        self.compiled = True
        
    
model = ann()

model.compile(loss = loss, lr = 0.01)

model.fit(X,y, epochs = 100)



    






