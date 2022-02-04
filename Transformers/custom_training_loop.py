#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 09:40:06 2022

@author: sahmaran
"""
from time import time
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, BatchNormalization, Dropout, Add, Input, Flatten, Embedding, LayerNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import Accuracy, CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
from tensorflow.keras.activations import relu, softmax


from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
from tensorflow.keras.datasets import cifar100

cifar100.load_data()

X_train = X_train/255.0
X_test = X_test/255.0



class ffn(Model): #### FF part
    def __init__(self, n_units, n_layers, dropout_rate, activation_of_dense_layers = "relu"):
        super().__init__();
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.activations = tf.keras.activations.get(activation_of_dense_layers)
        self.flatten = Flatten()
        self.layers_ = [Dense(self.n_units, activation = self.activations) for _ in range(self.n_layers)]
        self.dropout = [Dropout(self.dropout_rate) for i in range(self.n_layers)]
        self.final_dense = Dense(10, activation = "softmax",  name = "Final_dens")
        self.metric_v = tf.keras.metrics.SparseCategoricalAccuracy()
        self.metric_t = tf.keras.metrics.SparseCategoricalAccuracy()
        
    @tf.function(jit_compile = True)
    def call(self, x, training = None):
        x = self.flatten(x)
        if training:
            for i, dense_layer in enumerate(self.layers_):
                x = self.dropout[i](x, training)
                x = dense_layer(x)
                
            return self.final_dense(x)
        else:
            
            for i, _ in enumerate(self.layers_):
                x = self.layers_[i](x)
            return self.final_dense(x)
        
    @tf.function(jit_compile = True) ### we do here just-in-time compilation for boosting
    ### since this dude is compiled, trainable weights can not be changed later on.
    def apply_grads(self, train_x, train_y):  
        
        with tf.GradientTape() as tape:
            logits = self.call(train_x, training = True)
            loss_ = self.loss(train_y, logits)
        grads = tape.gradient(loss_, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return loss_, logits
    
    
    @tf.function
    def train_for_one_epoch(self, X, y, batch_size = 64, validation_data = None):
        train_data = tf.data.Dataset.from_tensor_slices((X,y))
        train_data_batched = train_data.shuffle(buffer_size = len(X)).batch(batch_size)
        ##apply.grad
        if validation_data:
            test_data = tf.data.Dataset.from_tensor_slices(validation_data)
            test_data_batched = test_data.shuffle(buffer_size = len(validation_data[0])).batch(batch_size)
            for  (train_x, train_y), (test_x, test_y) in zip(train_data_batched, test_data_batched):
                loss_, logits = self.apply_grads(train_x, train_y)
                logits_v = model(test_x)
                self.metric_v.update_state(test_y, logits_v)
                self.metric_t.update_state(train_y, logits)
            
        else:
            for (train_x, train_y) in train_data_batched:
                loss_, logits = self.apply_grads(train_x, train_y)
                self.metric_t.update_state(train_y, logits)
            
    @tf.function
    def fit_custom(self, X, y, batch_size, epoch, validation_data = None):
        for i in range(epoch):
            self.train_for_one_epoch(X, y, batch_size, validation_data)
            if validation_data:
                tf.print(self.metric_v.result(), self.metric_t.result())
            else:
                tf.print(self.metric_t.result())
            self.metric_v.reset_state()
            self.metric_t.reset_state()
            
        

        
        ###return , loss_avg, vs vs vs
            
model = ffn(250,5, 0.1)
model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy())

a = time()
with tf.device("gpu:1"):
    model.fit_custom(X_train, y_train, 64, 10, (X_test, y_test))
print(time()-a)

