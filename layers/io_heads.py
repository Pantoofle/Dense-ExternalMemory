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
        self.units = output_size + memory_size*(vector_size+2)
        self.vector_size = vector_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.memory = self.add_weight(
                shape=(self.memory_size, self.vector_size),
                initializer=RandomUniform(minval=0., maxval=0.25),
                trainable=False,
                name="MEMORY")
        self.memory.eval(K.get_session())
        self.state_shape = (1, self.memory_size*(self.vector_size+2) +3)
        self.states=[tf.random_uniform(shape=self.state_shape, minval=0., maxval=0.25, dtype="float32")]
        self.seq_size=input_shape[1]
        self.built = True

    def get_initial_state(self, inputs):
        return [tf.random_uniform(shape=self.state_shape, minval=0., maxval=0.25, dtype="float32")]

    def step(self, inputs, states):
        def dist(x, y):
            nx = tf.norm(x)
            #  ny = tf.norm(y)
            m = tf.matmul(tf.reshape(x, (1, self.vector_size)) , y)
            return m

        def focus_by_content(x):
            dists = tf.map_fn(
                    lambda y: dist(x, tf.reshape(y, (self.vector_size, 1))),
                    self.memory)
            # dists = 1.-tf.nn.softmax(dists)
            #  dists = tf.reshape(dists, (self.memory_size, ))
            #  dists = tf.nn.softmax(dists)
            #  dists = tf.reshape(dists, (self.memory_size, 1, 1))
            #  print("Dist: ", dists)
            return dists

        def read_mem(weight):
            weight = tf.reshape(weight, (1, self.memory_size))
            #  weight = tf.nn.softmax(weight)
            r = tf.matmul(weight, self.memory)
            return r

        def write_mem(weight, vector):
            weight = tf.reshape(weight, (self.memory_size, 1))
            vector = tf.reshape(vector, (1, self.vector_size))
            adder = K.dot(weight, vector)
            self.memory = tf.add(self.memory, adder)

        _, m = tf.split(states[0], [self.vector_size, self.memory_size*(self.vector_size+2)], axis=1)
        m = tf.reshape(m, (self.memory_size, self.vector_size+2))
        m, _ = tf.split(m, [self.vector_size, 2], axis=1)
        self.memory = m
        
        print("Calling IO_Head...")
        inputs = inputs[0]
        x, w_weight, erase = tf.split(inputs, 
            [self.vector_size, self.memory_size, 1], axis=0)

        print("Reading...")
        r_weight = focus_by_content(x)
        read = read_mem(r_weight)

        print("Writing...")
        write_mem(w_weight, x)   
      
        w_w = tf.reshape(w_weight, (self.memory_size, 1))
        r_w = tf.reshape(r_weight, (self.memory_size, 1))
        m = tf.concat([self.memory, r_w, w_w], axis=-1)
        
        flat = tf.reshape(m, (1, self.memory_size*(self.vector_size+2)))

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
    
