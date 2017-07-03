import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Recurrent
from keras.initializers import *


class IO_Heads(Recurrent):
    def __init__(self, units, vector_size, memory_size, entry_size, depth=0, read_heads=1, **kwargs):
        """ 
        Initialise the network
            - units       : output size 
            - vector_size : size of one element from the input sequence
            - memory_size : number of memory slots
            - entry_size  : size of a memory slot 
            - depth       : nb of Dense layer applied to generate the memory parameters 
                            and after read to return the output
            - read_heads  : number of read heads 
        """
        
        self.memory_size = memory_size
        self.vector_size = vector_size
        self.units = units
        self.entry_size = entry_size
        self.depth = depth
        self.read_heads = read_heads
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Builds the network, sets the initial weights
        """

        # Generates the memory access params 
        self.pre_treat = self.add_weight(
                shape=(self.vector_size + self.entry_size, 
                    self.memory_size+(2+self.read_heads)*self.entry_size),
                initializer="uniform",
                trainable=True,
                name="PRE")
        
        # The deap network
        self.deep_pre = []
        for i in range(self.depth):
            self.deep_pre.append( self.add_weight(
                    shape = (self.memory_size+(2+self.read_heads)*self.entry_size, self.memory_size+(2+self.read_heads)*self.entry_size),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_PRE_"+str(i)))

        # The memory, note the 'trainable=False'
        self.memory = self.add_weight(
                shape=(self.memory_size, self.entry_size),
                initializer=RandomUniform(minval=0., maxval=0.25),
                trainable=False,
                name="MEMORY")
       
        # Because it is not trainable we need to evaluate it before 
        self.memory.eval(K.get_session())

        # Deep 
        self.deep_post = []
        for i in range(self.depth):
            self.deep_post.append( self.add_weight(
                    shape = (self.vector_size + self.entry_size + self.read_heads*self.entry_size, 
                        self.vector_size + self.entry_size + self.read_heads*self.entry_size),
                    initializer = "uniform",
                    trainable = True,
                    name = "DEEP_POST_"+str(i)))

        # Generates the output from the read vectors 
        self.post = self.add_weight(
                shape=(self.vector_size + self.entry_size + self.read_heads*self.entry_size, 3),
                initializer="uniform",
                trainable=True,
                name="POST")

        # The memory and prev state will be given to the next state 
        self.mem_shape = (self.memory_size, self.entry_size)
        self.states=[tf.constant(0., shape=(self.entry_size,)),
                tf.random_uniform(shape=self.mem_shape, minval=0., maxval=0.25, dtype="float32")]
        self.built = True

    def get_initial_state(self, inputs):
        """
        Initialisation of the states = initial memory content
        """
        return [tf.constant(0., shape=(self.entry_size,)), tf.constant(0.01, shape=(self.memory_size, self.entry_size))]

    def step(self, inputs, states):
        def dist(x, y):
            """
            Distance between two vectors. Was cosine in the begining but too messy with a random initial memory
            """
            m = tf.matmul(tf.reshape(x, (1, self.entry_size)) , y)
            return m

        def focus_by_content(x):
            """
            Generates the weight vector
            """
            dists = tf.map_fn(
                    lambda y: dist(x, tf.reshape(y, (self.entry_size, 1))),
                    self.memory)
            return dists

        def read_mem(weight):
            """
            Applies the weight and produces the read vector
            """
            weight = tf.reshape(weight, (1, self.memory_size))
            r = tf.matmul(weight, self.memory)
            return r

        def write_mem(weight, vector, eraser):
            """
            Writes vector to memory with weight
            """
            weight = tf.reshape(weight, (self.memory_size, 1))
            vector = tf.reshape(vector, (1, self.entry_size))
            eraser = tf.reshape(eraser, (1, self.entry_size))
            subb = K.dot(weight, eraser)
            adder = K.dot(weight, vector)
            self.memory = tf.subtract(self.memory, subb)
            self.memory = tf.add(self.memory, adder)

        self.memory = states[1]
        
        print("Before concat: ", inputs[0].shape, states[0].shape)
        inputs = K.concatenate([inputs[0], states[0]], axis=-1)
        print("After concat: ", inputs)
        # Applies the first layers
        pre = K.dot(tf.reshape(inputs, (1, self.vector_size + self.entry_size)), self.pre_treat)
        
        # Deep enough 
        for i in range(self.depth):
            pre = K.dot(pre, self.deep_pre[i])

        
        # Extracts data from the generated vector
        w_vect, w_weight, erase, r_vect = tf.split(pre, 
            [self.entry_size, self.memory_size, self.entry_size, self.entry_size*self.read_heads], axis=1)

        r_vectors = []
        for i in range(self.read_heads):
            r, r_vect = tf.split(r_vect, [self.entry_size, self.entry_size*(self.read_heads-i-1)], axis=1)
            r_vectors.append(r)

        reads = []
        for i in range(self.read_heads):
            r_weight = focus_by_content(r_vectors[i])
            reads.append(read_mem(r_weight))
        write_mem(w_weight, w_vect, erase)   
        
        # The post operations
        r = K.concatenate(reads+[tf.reshape(inputs, (1, self.vector_size + self.entry_size))], axis=1)

        for i in range(self.depth):
            r = K.dot(r, self.deep_post[i])
        
        res = K.dot(r, self.post)
        # We concatenate the memory to the output to print the memory state from outside the layer
        return res, [tf.reshape(reads[0], (self.entry_size, )), self.memory]

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

