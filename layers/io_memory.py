import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import *

class IO_Layer(Layer):
    def __init__(self, memory_size, entry_size, output_size, **kwargs):
        print("Initialisation...")
        self.mem_size=memory_size
        self.entry_size=entry_size
        self.output_size=output_size
        super(IO_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building...")

        input_size=list(input_shape)[-1]

        self.memory = self.add_weight(shape=(self.mem_size, self.entry_size),
                initializer="zeros",
                trainable=False,
                name="MEMORY")
        print(self.memory)
        # Units that will built the weight vector
        self.w_key_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer='uniform',
                trainable=True,
                name="W_KG")
        
        self.w_vect_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer='uniform',
                trainable=True,
                name="W_VG")

        self.w_erase_vect_gen = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer='uniform',
                trainable=True,
                name="W_EVG")

        self.r_key_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer='uniform',
                trainable=True,
                name="R_KG")
        

        self.post_network = self.add_weight(
                shape=(self.entry_size + input_size, self.output_size),
                initializer='uniform',
                trainable=True,
                name="POST")
        
        super(IO_Layer, self).build(input_shape)  
    def call(self, x):
        def vect_dist(x, y):
            p = K.dot(x, K.transpose(y))

            nx = tf.norm(x) 
            ny = tf.norm(y) 
            return p/(nx*ny)

        print("Calling...")
        def focus_by_content(x):
            dists = [vect_dist(x, tf.reshape(m, [1, int(m.shape[0])]))
                for m in tf.unstack(self.memory)]
            s = sum(dists)
            v = [d/s for d in dists]
            r = tf.reshape(v, (self.mem_size,1))
            return r 

        def read_mem(weight):
            r = tf.multiply(self.memory,weight)
            res = tf.add_n(tf.unstack(r))
            return res

        def write_mem(weight, erase, vector):
            erase = tf.reshape(erase, (1, self.entry_size))
            one = tf.constant(np.ones((1, self.entry_size), dtype="float32"))
            stacked_ones = tf.stack([one for i in range(self.mem_size)])
            stacked_erase = tf.stack([erase for i in range(self.mem_size)])
            weight = tf.reshape(weight, [self.mem_size, 1, 1])
            
            
            #print("memory: ", self.memory)
            #print("one: ", one)
            #print("erase: ", erase)
            #print("stacked_one: ", stacked_ones)
            #print("stacked_erase: ", stacked_erase)
            #print("weight: ", weight)
            

            weighted_erase = weight*stacked_erase
            # print("weighted_erase shape: ", weighted_erase.shape)
            # print(self.memory)
            minus = stacked_ones - weighted_erase
            minus = tf.reshape(minus, (self.mem_size, self.entry_size))
            # print("minus: ", minus)
            # print("mem: ", self.memory)
            self.memory = tf.multiply(self.memory,minus)
            # print("new memory: ", self.memory)

        x = x[0]
        print("Input vector shape: "+str(x.shape))
        print("Generators shape: " + str(self.w_key_generator.shape))
        # Generating read and write keys
        w_key = K.dot(x, self.w_key_generator)
        r_key = K.dot(x, self.r_key_generator)
        print("Key shape: " + str(w_key.shape))

        # Focus by content
        print("Generating weights...")
        w_weight = focus_by_content(w_key)
        r_weight = focus_by_content(r_key)

        # Erase vector for writing
        print("Erasing...")
        erase = K.dot(x, self.w_erase_vect_gen)

        # Vector to be written
        print("Generating write vector...")
        vect = K.dot(x, self.w_vect_generator)

        # Writing
        print("Writing...")
        write_mem(w_weight, erase, vect)
    
        # Reading
        print("Reading...")
        r_vect = read_mem(r_weight)
        r_vect = tf.reshape(r_vect, (1, self.entry_size))
        print(r_vect)
        x = tf.reshape(x, (1, self.entry_size))
        print("x: ", x)
        print("Computing output...")
        print(tf.concat([x, r_vect], 1))
        print(self.post_network)
        r =  K.dot(tf.concat([x, r_vect], 1), self.post_network)
        return r

    def compute_output_shape(self, input_shape):
        print("Computing output shape...")
        print("Input: " + str(input_shape))
        output_shape = list(input_shape)
        output_shape[-1] = self.output_size
        print("Output: " + str(tuple(output_shape)))
        return (tuple(output_shape))
