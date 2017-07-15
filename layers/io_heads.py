import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    """
    The DNC kernel
    Generates and uses the memory, the read heads
    Read:  Graves, Alex, et al. "Hybrid computing using a neural network with dynamic external memory." Nature 538.7626 (2016): 471-476.
    to understand the notations and the different parts
    """

    def __init__(self, units, 
            vector_size, 
            memory_size, 
            entry_size, 
            depth=0, 
            read_heads=1, 
            write_heads=1, 
            **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.vector_size = vector_size
        self.units = units
        self.entry_size = entry_size
        self.depth = depth
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.heads = read_heads + write_heads
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        
        # The key vectors for "focus by content"
        pre_size = self.entry_size*self.heads
        # The interpolation coefficient
        pre_size += 1*self.heads
        # The shift vectors
        pre_size += self.memory_size*self.heads
        # The erase vectors
        pre_size += self.entry_size*self.write_heads
        # The vectors to write
        pre_size += self.entry_size*self.write_heads
        
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
        
        # The weights will be passed as states for next step
        w = [tf.constant(0., shape=(self.memory_size,)) for i in range(self.heads)]
        self.states=[tf.constant(0., shape=self.mem_shape)] + w

        print("States at init: ", self.states)
        self.built = True

    def get_initial_state(self, inputs):
        w = [tf.constant(0., shape=(self.memory_size,)) for i in range(self.heads)]
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

        print("Getting previous memory state and weights...")
        
        self.memory = states[0]
        w_t = states[1:]

        # Only use one vect at a time. Forces batch_size=1
        inputs = inputs[0]       

        # Apply pre treatment
        pre = K.dot(tf.reshape(inputs, (1, self.vector_size)), self.pre_treat)
        
        for i in range(self.depth):
            pre = K.dot(pre, self.deep_pre[i])

        print("pre: ", pre)
        # Extract the vectors
        keys, interpol_gates, shifts, erasers, w_vects = tf.split(pre, 
            [self.entry_size*self.heads, 
                1*self.heads, 
                self.memory_size*self.heads, 
                self.entry_size*self.write_heads, 
                self.entry_size*self.write_heads],
            axis=1)

        key = []
        for i in range(self.heads):
            k, keys = tf.split(keys, [self.entry_size, self.entry_size*(self.heads-i-1)], axis=1)
            key.append(tf.nn.softmax(k))

        interpol_gate = []
        for i in range(self.heads):
            ig, interpol_gates = tf.split(interpol_gates, [1, 1*(self.heads-i-1)], axis=1)
            interpol_gate.append(ig)
        
        shift = []
        for i in range(self.heads):
            s, shifts = tf.split(shifts, [self.memory_size, self.memory_size*(self.heads-i-1)], axis=1)
            shift.append(s)

        erase = []
        for i in range(self.write_heads):
            e, erasers = tf.split(erasers, 
                    [self.entry_size, self.entry_size*(self.write_heads-i-1)], axis=1)
            erase.append(e)

        write = []
        for i in range(self.write_heads):
            a, w_vects = tf.split(w_vects, 
                    [self.entry_size, self.entry_size*(self.write_heads-i-1)], axis=1)
            write.append(a)

        print("Reading... ")
        reads = []
        weight = [] 
        for i in range(self.read_heads):
            r= address(w_t[i], key[i], interpol_gate[i], shift[i])
            weight.append(tf.reshape(r, (self.memory_size,)))
            reads.append(read_mem(r))
       
        print("WH: ", len(erase), "   ", len(write))

        print("Writing...")
        for i in range(self.write_heads):
            j = self.read_heads +i
            w_weight = address(w_t[j], key[j], interpol_gate[j], shift[j])
            write_mem(w_weight, write[i], erase[i])   
            weight.append(tf.reshape(w_weight, (self.memory_size,)))
        
        r = K.concatenate(reads+[tf.reshape(inputs, (1, self.vector_size))], axis=1)

        for i in range(self.depth):
            r = K.dot(r, self.deep_post[i])

        res = K.dot(r, self.post)
        return res, [self.memory] + weight 

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

