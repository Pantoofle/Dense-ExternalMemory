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
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=False,
                name="MEMORY")

        self.memory.eval(K.get_session())
        # Units that will built the weight vector
        self.w_key_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=True,
                name="W_KG")
         
        self.w_vector_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=True,
                name="W_VG")

        self.w_erase_vect_gen = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=True,
                name="W_EVG")

        self.r_key_generator = self.add_weight(
                shape=(input_size, self.entry_size),
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=True,
                name="R_KG")
        

        self.post_network = self.add_weight(
                shape=(self.entry_size + input_size, self.output_size),
                initializer=initializers.random_uniform(minval=0.0, maxval=1.0),
                trainable=True,
                name="POST")
        
        self.build = True
    
    def call(self, x):
        total_input=x.shape[0]
        
        def vect_dist(x, y):
            p = K.dot(x, K.transpose(y))

            nx = tf.norm(x) 
            ny = tf.norm(y)
            return p/(nx*ny)

        print("Calling...")
        def focus_by_content(x):
            dists = [vect_dist(x, tf.reshape(m, (1, int(m.shape[0]))))
                for m in tf.unstack(self.memory)]
            
            s = sum(dists)
            print("\nSum: ", s)

            v = [d/s for d in dists]
            return tf.concat(v, axis=1)

        def read_mem(weight):
            r = tf.multiply(self.memory,weight)
            res = tf.reduce_sum(r, axis=1, keep_dims=True)
            return res

        def write_mem(weight, erase, vector):
            eraser = tf.multiply(weight, erase)
            self.memory = tf.subtract(self.memory,
                    tf.multiply(self.memory, eraser))
            self.memory = tf.add(self.memory, tf.multiply(weight, vector))

        print("Input vector shape: ", x.shape)

#        print("Generators shape: ", self.w_key_generator.shape)
        # Generating read and write keys
        w_key = K.dot(x, self.w_key_generator)
        r_key = K.dot(x, self.r_key_generator)
#        print("Key shape: ", w_key.shape)

        # Focus by content
        print("Generating weights...")
        w_weight = focus_by_content(w_key)
        r_weight = focus_by_content(r_key)
#        print("Weights shape: ", w_weight)
        # Erase vector for writing
        print("Generating erase vector...")
        erase = K.dot(x, self.w_erase_vect_gen)

        # Vector to be written
        print("Generating write vector...")
        vect = K.dot(x, self.w_vector_generator)
        # Writing
        print("Writing...")

        write_mem(w_weight, erase, vect)
    
        # Reading
        print("Reading...")
        r_vect = read_mem(r_weight)
        print("Computing output...")
        conc = tf.concat([x, r_vect], 2)
        r =  K.dot(conc, self.post_network)
        return r

    def compute_output_shape(self, input_shape):
        print("Computing output shape...")
        return (input_shape[0], 1, self.output_size) 

    def get_config(self):
        config = {
                'memory_size': self.mem_size, 
                'entry_size': self.entry_size,
                'output_size': self.output_size
        }
        base_config = super(IO_Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
