import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    def __init__(self, memory_size, vector_size, output_size, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.units = output_size + memory_size*vector_size
        self.vector_size = vector_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.memory = self.add_weight(
                shape=(self.memory_size, self.vector_size),
                initializer="zeros",
                trainable=False,
                name="MEMORY")
        print("Input_shape: ", input_shape) 
        self.memory.eval(K.get_session())
        self.states=[None]
        self.seq_size=input_shape[1]
        self.built = True

    def step(self, inputs, states):
        def dist(x, y):
            diff = tf.norm(tf.subtract(x, y))
            return diff

        def focus_by_content(x):
            dists = tf.map_fn(
                    lambda y: dist(x, tf.reshape(y, (self.vector_size, ))),
                    self.memory)
            dists = tf.nn.softmax(dists)
            return tf.nn.softmax(dists)

        def read_mem(weight):
            weight = tf.nn.softmax(tf.reshape(weight, (1, self.memory_size)))
            print("weight: ", weight)
            r = tf.matmul(weight, self.memory)
            r = tf.nn.softmax(r)
            print("Read: ", r)
            #  r = tf.reshape(r, (self.vector_size,))
            return r

        def write_mem(weight, erase, vector):
            weight = tf.nn.softmax(weight)
            weight = tf.reshape(weight, (self.memory_size, 1))
            vector = tf.reshape(vector, (1, self.vector_size))
            adder = K.dot(weight, vector)
            self.memory = tf.add(self.memory, adder)

        print("Calling IO_Head...")
        inputs = inputs[0]
        print("Inputs: ", inputs.shape)
        x, w_weight, erase = tf.split(inputs, 
            [self.vector_size, self.memory_size, 1], axis=0)

        r_weight = focus_by_content(x)

        print("Reading...")
        read = read_mem(r_weight)

        print("Writing...")
            #  w_weight = tf.nn.sigmoid(w_weight)
        write_mem(w_weight, erase, x)   
       
        flat = tf.reshape(self.memory, (1, self.memory_size*self.vector_size))
        ret = tf.concat([read, flat], axis = -1)
        print("Ret: ", ret)
        return ret, [ret]

    def get_config(self):
        config = {
                'memory_size': self.memory_size, 
                'vector_size': self.vector_size,
                'output_size': self.units
        }
        base_config = super(IO_Heads, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    #  def compute_output_shape(self, input_shape):
    #      output_shape = (input_shape[0], input_shape[1], self.units)
    #      return output_shape
#  class ResetMemory(Callback):
#      def on_batch_begin(self, batch, logs={}):
#          self.layer[0] = tf.random_uniform((self.memory_size, self.vect_size), 0, 1,  dtype="float32")
#
    
