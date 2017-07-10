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
        heads = self.read_heads + 1
        pre_size = self.entry_size*heads+1*heads+self.memory_size*heads+self.entry_size+self.entry_size
        self.pre_treat = self.add_weight(
                shape=(self.vector_size, pre_size),
                initializer="uniform",
                trainable=True,
                name="PRE")
        
        self.deep_pre = []
        for i in range(self.depth):
            self.deep_pre.append( self.add_weight(
                    shape = (pre_size, pre_size),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_PRE_"+str(i)))

        self.memory = tf.constant(0., shape=(self.memory_size, self.entry_size))
       
        self.deep_post = []
        for i in range(self.depth):
            self.deep_post.append( self.add_weight(
                    shape = (self.vector_size + self.read_heads*self.entry_size,self.vector_size + self.read_heads*self.entry_size),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_POST_"+str(i)))


        self.post = self.add_weight(
                shape=(self.vector_size + self.read_heads*self.entry_size, self.units),
                initializer="uniform",
                trainable=True,
                name="POST")

        self.mem_shape = (self.memory_size, self.entry_size)
        w = [tf.constant(0., shape=(self.memory_size,)) for i in range(heads)]
        self.states=[tf.constant(0., shape=self.mem_shape)] + w
        print("States at init: ", self.states)
        self.built = True

    def get_initial_state(self, inputs):
        w = [tf.constant(0., shape=(self.memory_size,)) for i in range(self.read_heads+1)]
        return [tf.constant(0., shape=self.mem_shape)] + w

    def step(self, inputs, states):
        def dist(x, y):
            m = tf.matmul(tf.reshape(x, (1, self.entry_size)) , y)
            if tf.norm(x) == 0 or tf.norm(y) == 0:
                return tf.constant(1., shape=(1, 1))
            else:
                return m/(tf.norm(y) * tf.norm(x))

        def focus_by_content(x):
            dists = tf.map_fn(
                    lambda y: dist(x, tf.reshape(y, (self.entry_size, 1))),
                    self.memory)
            dists = tf.nn.softmax(dists)
            return dists

        def interpolation(w_t, w_t2, g):
            return w_t*g + w_t2*(1-g)

        def convolution(w, s):
            si = [K.concatenate([s[i:], s[:i]], axis=0) for i in range(s.shape[0])]
            r = tf.stack(si)
            return r

        def read_mem(weight):
            weight = tf.reshape(weight, (1, self.memory_size))
            r = tf.matmul(weight, self.memory)
            r = tf.nn.softmax(r)
            return r

        def write_mem(weight, vector, eraser):
            weight = tf.reshape(weight, (self.memory_size, 1))
            
            vector = tf.reshape(vector, (1, self.entry_size))
            vector = tf.nn.softmax(vector)
            
            eraser = tf.reshape(eraser, (1, self.entry_size))
            eraser = tf.nn.softmax(eraser)

            dell = K.dot(weight, eraser)
            adder = K.dot(weight, vector)
            
            self.memory = tf.multiply(self.memory, dell)
            self.memory = tf.add(self.memory, adder)

        def address(w_t, r_key, interpol_gate, shift):
            w = focus_by_content(r_key)
            w = interpolation(w, w_t, interpol_gate)
            w = convolution(w, shift)
            return w


        # _, m = tf.split(states[0], [self.vector_size, self.memory_size*(self.vector_size+2)], axis=1)
        print("Getting previous memory state and weights...")
        print("States: ", states)
        
        
        self.memory = states[0]
        w_t = states[1:]

        inputs = inputs[0]       

        pre = K.dot(tf.reshape(inputs, (1, self.vector_size)), self.pre_treat)
        
        for i in range(self.depth):
            pre = K.dot(pre, self.deep_pre[i])

        print("pre: ", pre)
        
        heads = self.read_heads + 1

        r_keys, interpol_gates, shifts, erase, w_vect = tf.split(pre, 
            [self.entry_size*heads, 
                1*heads, 
                self.memory_size*heads, 
                self.entry_size, 
                self.entry_size],
            axis=1)

        r_key = []
        for i in range(heads):
            r, r_keys = tf.split(r_keys, [self.entry_size, self.entry_size*(heads-i-1)], axis=1)
            r_key.append(tf.nn.softmax(r))

        interpol_gate = []
        for i in range(heads):
            ig, interpol_gates = tf.split(interpol_gates, [1, 1*(heads-i-1)], axis=1)
            interpol_gate.append(ig)
        
        shift = []
        for i in range(heads):
            s, shifts = tf.split(shifts, [self.memory_size, self.memory_size*(heads-i-1)], axis=1)
            shift.append(s)

        print("Reading... ")
        reads = []
        r_weight = [] 
        for i in range(heads-1):
            r= address(w_t[i], r_key[i], interpol_gate[i], shift[i])
            r_weight.append(tf.reshape(r, (self.memory_size,)))
            reads.append(read_mem(r))
        
        print("Writing...")
        w_weight = address(w_t[-1], r_key[-1], interpol_gate[-1], shift[-1])
        write_mem(w_weight, w_vect, erase)   
        
        r = K.concatenate(reads+[tf.reshape(inputs, (1, self.vector_size))], axis=1)

        for i in range(self.depth):
            r = K.dot(r, self.deep_post[i])

        res = K.dot(r, self.post)
        print("Ret: ", res)
        print("States: ",[self.memory] + r_weight + [tf.reshape(w_weight, (self.memory_size,))])
        return res, [self.memory] + r_weight + [tf.reshape(w_weight, (self.memory_size,))] 

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
    
