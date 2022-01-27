#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:13:47 2022

@author: sahmaran
"""
### The model below achieves %95 percent accuracy on the mnist + fashion_mnist dataset!

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
(X_train_f, y_train_f), (X_test_f, y_test_f) = fashion_mnist.load_data()
(X_train_m, y_train_m), (X_test_m, y_test_m) = mnist.load_data()

X_train = np.concatenate([X_train_f,X_train_m], axis = 0)
X_test = np.concatenate([X_test_f,X_test_m], axis = 0)

y_train = np.concatenate([y_train_f, y_train_m], axis = 0)
y_test = np.concatenate([y_test_f, y_test_m], axis = 0)

y_test_m = y_test_m + 10
y_train_m = y_train_m + 10


X_train = X_train/255.0
X_test = X_test/255.0

from tensorflow.keras.layers import Reshape, Layer, Dense, MultiHeadAttention, BatchNormalization, Dropout, Add, Input, Flatten, Embedding, LayerNormalization, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import Accuracy, CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
from tensorflow.keras.activations import relu, softmax



class Lay_(Layer): ### this dude takes linear combination (in a trainable way) rows of a given hidden states and therefore squaes the information
    def __init__(self, trainable = True):
        super().__init__()
        self.trainable_ = trainable
    def build(self, input_shape):
        w_kernel = tf.random_normal_initializer()
        self.w_kernel = tf.Variable(initial_value = w_kernel((1,input_shape[-2])), dtype = "float32", trainable = self.trainable_)
    def call(self, inputs):
        raw = tf.matmul(self.w_kernel, inputs) 
        return raw




I = Input((28,28))
A, B, C = LSTM(350, return_sequences= True, return_state=True)(I)
A = Dropout(0.1)(A)
A = Lay_()(A)
A = Dropout(0.1)(A)
A = Reshape((350,))(A)
L = Dense(20, "softmax")(A)
mo = Model(I, L)

mo.compile(optimizer= "adam", loss= sparse_categorical_crossentropy, metrics = "accuracy")

mo.fit(X_train, y_train, batch_size = 32, epochs = 35, validation_data=(X_test, y_test))





