import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    def __init__(self, memory_size, entry_size, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.units = entry_size
        self.entry_size = entry_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.memory = self.add_weight(
                shape=(self.memory_size, self.entry_size),
                initializer=RandomUniform(minval=0., maxval=0.25),
                trainable=False,
                name="MEMORY")
       
        self.memory.eval(K.get_session())
        self.state_shape = (self.memory_size, self.entry_size)
        self.states=[tf.random_uniform(shape=self.state_shape, minval=0., maxval=0.25, dtype="float32")]
        self.seq_size=input_shape[1]
        self.built = True

    def get_initial_state(self, inputs):
        return [tf.random_uniform(shape=self.state_shape, minval=0., maxval=0.25, dtype="float32")]

    def step(self, inputs, states):
        def dist(x, y):
            m = tf.matmul(tf.reshape(x, (1, self.entry_size)) , y)
            return m

        def focus_by_content(x):
            dists = tf.map_fn(
                    lambda y: dist(x, tf.reshape(y, (self.entry_size, 1))),
                    self.memory)
            return dists

        def read_mem(weight):
            weight = tf.reshape(weight, (1, self.memory_size))
            r = tf.matmul(weight, self.memory)
            return r

        def write_mem(weight, vector, eraser):
            weight = tf.reshape(weight, (self.memory_size, 1))
            vector = tf.reshape(vector, (1, self.entry_size))
            eraser = tf.reshape(eraser, (1, self.entry_size))
            subb = K.dot(weight, eraser)
            adder = K.dot(weight, vector)
            self.memory = tf.subtract(self.memory, subb)
            self.memory = tf.add(self.memory, adder)

        # _, m = tf.split(states[0], [self.vector_size, self.memory_size*(self.vector_size+2)], axis=1)
        print("Getting previous memory state...")
        m = tf.reshape(states[0], (self.memory_size, self.entry_size))
        self.memory = m
        
        inputs = inputs[0]
        r_vect, w_vect, w_weight, erase = tf.split(inputs, 
            [self.entry_size, self.entry_size, self.memory_size, self.entry_size], axis=0)

        print("Reading...")
        r_weight = focus_by_content(r_vect)
        read = read_mem(r_weight)
        
        print("Writing...")
        write_mem(w_weight, w_vect, erase)   
      
        print("Read: ", read)
        return read, [self.memory]

    def get_config(self):
        config = {
                'memory_size': self.memory_size, 
                'entry_size': self.entry_size,
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
    
