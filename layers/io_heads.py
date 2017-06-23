import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    def __init__(self, units, memory_size, entry_size, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.units = units
        self.entry_size = entry_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.pre_treat = self.add_weight(
                shape=(self.units, self.memory_size+3*self.entry_size),
                initializer="uniform",
                trainable=True,
                name="PRE")
        
        self.memory = self.add_weight(
                shape=(self.memory_size, self.entry_size),
                initializer=RandomUniform(minval=0., maxval=0.25),
                trainable=False,
                name="MEMORY")
       
        self.memory.eval(K.get_session())

        self.post = self.add_weight(
                shape=(self.units*2, self.units),
                initializer="uniform",
                trainable=True,
                name="POST")

        self.mem_shape = (self.memory_size, self.entry_size)
        self.states=[tf.constant(0., shape=(1, self.units)), tf.random_uniform(shape=self.mem_shape, minval=0., maxval=0.25, dtype="float32")]
        self.built = True

    def get_initial_state(self, inputs):
        print("inputs, ", inputs)
        return [tf.reshape(inputs[0][0],(1, self.units)), tf.random_uniform(shape=self.mem_shape, minval=0., maxval=0.25, dtype="float32")]

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
        self.memory = states[1]
        print("in: ", states[0])
        
        inputs = states[0]
        
        pre = K.dot(tf.reshape(inputs, (1, self.units)), self.pre_treat)

        print("pre: ", pre)

        r_vect, w_vect, w_weight, erase = tf.split(pre, 
            [self.entry_size, self.entry_size, self.memory_size, self.entry_size], axis=1)

        print("Reading... ", r_vect)
        r_weight = focus_by_content(r_vect)
        read = read_mem(r_weight)
        
        print("Writing...")
        write_mem(w_weight, w_vect, erase)   
        
        r = K.concatenate([read, tf.reshape(inputs, (1, self.units))], axis=1)
        res = K.dot(r, self.post)
        print("Ret: ", res)
        return res, [res, self.memory]

    def get_config(self):
        config = {
                'memory_size': self.memory_size, 
                'entry_size': self.entry_size,
                'units': self.units
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
    
