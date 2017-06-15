from io_heads import *
from keras.layers import *
from keras.models import *

def build_RNN(input_shape, memory_size, vect_size, output_size):
    model = Sequential()
    model.add(IO_Heads(memory_size=memory_size, 
        vector_size=vect_size, 
        units=output_size,
        input_shape=input_shape))
    # model.add(Dense(1, activation='sigmoid'))
    return model
