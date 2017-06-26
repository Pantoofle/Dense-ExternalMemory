import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    def __init__(self, units, vector_size, memory_size, entry_size, depth=0, read_heads=1, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.vector_size = vector_size
        self.units = units
        self.entry_size = entry_size
        self.depth = depth
        self.read_heads = read_heads
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.pre_treat = self.add_weight(
                shape=(self.vector_size, self.memory_size+(2+self.read_heads)*self.entry_size),
                initializer="uniform",
                trainable=True,
                name="PRE")
        
        self.deep_pre = []
        for i in range(self.depth):
            self.deep_pre.append( self.add_weight(
                    shape = (self.memory_size+(2+self.read_heads)*self.entry_size, self.memory_size+(2+self.read_heads)*self.entry_size),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_PRE_"+str(i)))

        self.memory = self.add_weight(
                shape=(self.memory_size, self.entry_size),
                initializer=RandomUniform(minval=0., maxval=0.25),
                trainable=False,
                name="MEMORY")
       
        self.memory.eval(K.get_session())

        self.deep_post = []
        for i in range(self.depth):
            self.deep_post.append( self.add_weight(
                    shape = (self.units*(1+self.read_heads), self.units*(1+self.read_heads)),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_POST_"+str(i)))


        self.post = self.add_weight(
                shape=(self.vector_size + self.read_heads*self.entry_size, self.units),
                initializer="uniform",
                trainable=True,
                name="POST")

        self.mem_shape = (self.memory_size, self.entry_size)
        self.states=[tf.random_uniform(shape=self.mem_shape, minval=0., maxval=0.25, dtype="float32")]
        self.built = True

    def get_initial_state(self, inputs):
        print("inputs, ", inputs)
        return [tf.random_uniform(shape=self.mem_shape, minval=0., maxval=0.25, dtype="float32")]

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
        self.memory = states[0]
        
        inputs = inputs[0]       
        print("in: ", inputs)

        pre = K.dot(tf.reshape(inputs, (1, self.vector_size)), self.pre_treat)
        
        for i in range(self.depth):
            pre = K.dot(pre, self.deep_pre[i])

        print("pre: ", pre)
        
        
        w_vect, w_weight, erase, r_vect = tf.split(pre, 
            [self.entry_size, self.memory_size, self.entry_size, self.entry_size*self.read_heads], axis=1)

        r_vectors = []
        for i in range(self.read_heads):
            r, r_vect = tf.split(r_vect, [self.entry_size, self.entry_size*(self.read_heads-i-1)], axis=1)
            r_vectors.append(r)

        print("Reading... ")

        reads = []
        for i in range(self.read_heads):
            r_weight = focus_by_content(r_vectors[i])
            reads.append(read_mem(r_weight))
            print("Read: ", reads[-1])
        
        print("Writing...")
        write_mem(w_weight, w_vect, erase)   
        
        r = K.concatenate(reads+[tf.reshape(inputs, (1, self.vector_size))], axis=1)

        for i in range(self.depth):
            r = K.dot(r, self.deep_post[i])

        res = K.dot(r, self.post)
        print("Ret: ", res)
        return res, [self.memory]

    def get_config(self):
        config = {
                'memory_size': self.memory_size,
                'depth': self.depth,
                'vector_size': self.vector_size,
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
    
