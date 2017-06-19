from io_heads import *
from keras.layers import *
from keras.models import *

def build_RNN(input_shape, memory_size, vect_size, output_size, depth):
    model = Sequential()
    model.add(IO_Heads(memory_size=memory_size, 
        vector_size=vect_size, 
        input_shape=input_shape,
        return_sequences=True,
        depth=depth))
    model.add(Dense(output_size))
    return model

def build_LSTM(input_shape, vect_size, output_size):
    model = Sequential()
    model.add(LSTM(output_size, input_shape=input_shape, return_sequences=True))
    model.add(Dense(output_size, activation='softmax'))
    return model
