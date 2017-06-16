import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import *


class IO_Heads(Recurrent):
    def __init__(self, memory_size, vector_size, **kwargs):
        print("Initialisating IO_Heads...")
        self.memory_size = memory_size
        self.units = vector_size
        self.vect_size = vector_size
        super(IO_Heads, self).__init__(**kwargs)

    def build(self, input_shape):
        print("Building IO_Heads...")
        self.memory = self.add_weight(
                shape=(self.memory_size, self.vect_size),
                initializer=initializers.constant(0.1),
                trainable=False,
                name="MEMORY")

        self.generator = self.add_weight(
                shape=(self.vect_size*2, 
                    self.vect_size*3+self.memory_size),
                initializer='uniform',
                trainable=True,
                name="Generator")

        self.post_treatment = self.add_weight(
               shape=(self.vect_size*2, self.units),
               initializer='uniform',
               trainable=True,
               name="PostTreatment")

        self.memory.eval(K.get_session())
        self.states = [None]
        self.built = True

    def step(self, inputs, states):
        def dist(x, y):
            p = tf.matmul(x, tf.reshape(y, shape=(self.vect_size,1)))
            nx = tf.norm(x)
            ny = tf.norm(y)
            d = tf.divide(p, tf.multiply(nx, ny))
            d = tf.reshape(d, ())
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

        if states[0] is None:
            states[0] = tf.constant(0.1, shape=(1, self.vect_size))
            self.memory = tf.constant(0.1, shape=(self.memory_size, self.vect_size))

        vs = self.vect_size
        ms = self.memory_size
        inputs=tf.reshape(inputs[0], (1, self.vect_size))
        stat = tf.reshape(states[0][0], (1, self.vect_size))
        entry = tf.concat([inputs, stat], axis=1)
        gen = tf.matmul(entry, self.generator)
        r_vect, w_weight, w_vect, erase = tf.split(gen, 
                [vs, ms, vs, vs], axis=1)
        r_weight = focus_by_content(r_vect)

        print("Writing...")
        write_mem(w_weight, erase, w_vect)

        print("Reading...")
        read = read_mem(r_weight)
        post = tf.concat([inputs, read], axis=1)
        ret = tf.matmul(post, self.post_treatment)
        return ret, [ret]

    def get_config(self):
        config = {
                'memory_size': self.memory_size, 
                'vector_size': self.vect_size
        }
        base_config = super(IO_Heads, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

