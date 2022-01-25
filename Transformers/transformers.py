# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:31:26 2022

@author: hamsi
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import tensorflow as tf
tf.config.experimental.list_physical_devices()





from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0




from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, BatchNormalization, Dropout, Add, Input, Flatten, Embedding
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.metrics import Accuracy, CategoricalCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
from tensorflow.keras.activations import relu, softmax



class ffn(Layer): #### FF part
    def __init__(self, n_units, n_layers, dropout_rate, activation_of_dense_layers = "relu"):
        super().__init__();
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.activations = tf.keras.activations.get(activation_of_dense_layers)

        self.layers_ = [Dense(self.n_units, activation = self.activations) for _ in range(self.n_layers)]
        self.dropout = [Dropout(self.dropout_rate) for i in range(self.n_layers)]
        
    def call(self, x, training = None):
        if training:
            
            for i, _ in enumerate(self.layers_):
                x = self.dropout[i](x)
                x = self.layers_[i](x)
                
            return x
        else:
            
            for i, _ in enumerate(self.layers_):
                x = self.layers_[i](x)
            return x


    
class multi_att(Layer):
    def __init__(self, n_heads, key_dim, dropout_rate):
        super().__init__();
        self.num_heads = n_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.heads = MultiHeadAttention(self.num_heads,self.key_dim, dropout = self.dropout_rate)
        
    def call(self,T, training = None, return_attention_scores = False):
        return self.heads(query = T, key = T, value = T, training = training, return_attention_scores = return_attention_scores)

class positional_encoding(Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = Embedding(input_dim+1, input_dim, input_length = input_dim)
    def call(self, x):
        return self.embedding(x)


 
### there is comes the dope!
class model(Model):
    def __init__(self, n_heads, key_dim, dropout_rate, n_units, n_layers):
        super().__init__();
        self.multi_att_ = multi_att(n_heads, key_dim, dropout_rate) 
        self.positional_encoding = positional_encoding(n_units)
        self.ffnn = ffn(n_units, n_layers, dropout_rate, "relu")
        self.flatten = Flatten()
        self.final_dense = Dense(10, activation = "softmax")
        self.add = Add()
    @tf.function
    def call(self, x, training, return_attention_scores = False):
        x = self.add([x, self.positional_encoding(x)])
        k = self.multi_att_(x, return_attention_scores = return_attention_scores, training = training)
        x = self.add([k,x])
        k = self.multi_att_(x, return_attention_scores = return_attention_scores, training = training)
        x = self.add([k,x])
        
        x = self.ffnn(x, training)
        
        
        x = self.flatten(x)
        return self.final_dense(x)
    
mo = model(40,350,0.1,28,3)
mo.compile(optimizer= "adam", loss= sparse_categorical_crossentropy, metrics = "accuracy")
    

mo.fit(X_train, y_train, batch_size = 32, epochs = 150, validation_data=[X_test, y_test])


