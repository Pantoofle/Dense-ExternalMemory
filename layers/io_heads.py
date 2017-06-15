import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import *


class IO_Heads(Layer):
    def __init__(self, memory_size, vector_size, output_size, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.output_size = output_size
        self.vect_size = vector_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.memory = self.add_weight(
                shape=(self.memory_size, self.vect_size),
                initializer=initializers.constant(0.1),
                trainable=False,
                name="MEMORY")

        self.memory.eval(K.get_session())
    
        self.build = True

    def call(self, x):
        def dist(x, y):
            p = tf.matmul(x, tf.reshape(y, shape=(self.vect_size,1)))
            nx = tf.norm(x)
            ny = tf.norm(y)
            d = tf.divide(p, tf.multiply(nx, ny))
            d = tf.reshape(d, ())
            print("d: ", d)
            l = tf.constant([1000.], shape=())
            return tf.minimum(d, l)

        def focus_by_content(x):
            x = tf.reshape(x, (1, self.vect_size))
            dists = tf.map_fn(lambda y: dist(x, y), self.memory)
            tot = tf.reduce_sum(dists, axis=0)
            return tf.divide(dists, tot)

        def read_mem(weight):
            weight = tf.reshape(weight, (1, self.memory_size))
            r = tf.matmul(weight, self.memory)
            return r

        def write_mem(weight, erase, vector):
            weight = tf.reshape(weight, (self.memory_size, 1))
            erase  = tf.reshape(erase,  (1, self.vect_size))
            eraser = tf.matmul(weight, erase)
            self.memory = tf.subtract(self.memory,
                    tf.multiply(self.memory, eraser))
            adder = tf.multiply(weight, vector)
            self.memory = tf.add(self.memory, adder)

        print("Calling IO_Head...")
        vs = self.vect_size
        ms = self.memory_size
        r_vect, w_weight, w_vect, erase = tf.split(x, 
                [vs, ms, vs, vs], axis=1)
        r_weight = focus_by_content(r_vect)

        print("Writing...")
        write_mem(w_weight, erase, w_vect)

        print("Reading...")
        read = read_mem(r_weight)

        #  res = tf.concat([x, read], axis=1)
        print("Returning: ", read)
        return read

    def compute_output_shape(self, input_shape):
        print("Computing output shape...")
        print("Shape: ", (1, self.vect_size))
        return ((1, self.vect_size))


















