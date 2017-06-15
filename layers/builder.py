from io_heads import *
from keras.layers import *
from keras.models import *

def build_RNN(input_shape, memory_size, vect_size, output_size):
    model = Sequential()
    model.add(Dense(memory_size + 3*vect_size, input_shape=input_shape))
    model.summary()
    model.add(IO_Heads(memory_size=memory_size, 
        vector_size=vect_size, output_size=2*vect_size))
    model.add(Dense(output_size))
    return model
