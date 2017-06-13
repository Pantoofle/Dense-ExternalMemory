import numpy as np
import sys

sys.path.append("layers/")

from keras.models import *
from keras.layers import *

from io_memory import *
from data import *

UPPER_BOUND=100
NB_TIMESTEP=1

NB_TRAIN=1000
NB_TESTS=10

NB_EPOCH=50
BATCH_SIZE=32
MEMORY_SIZE=100
ENTRY_SIZE=UPPER_BOUND

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        print("Loading model...")
        model = load_model(sys.argv[2], {'IO_Layer': IO_Layer})
        save_path = sys.argv[2]
    else:
        print("Building new model...")
        model = Sequential()
        model.add(IO_Layer(input_shape=(NB_TIMESTEP, UPPER_BOUND),
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE,
            output_size=ENTRY_SIZE))
        model.add(Dense(1, activation='sigmoid'))
        save_path = input("Enter directory where to save the model: ")

    print("Compiling...")
    model.compile(optimizer='rmsprop',
                  loss='logcosh',
                  metrics=['accuracy'])

    model.summary() 
    print("Input shape: ", model.input_shape)
    print("Getting data...")
    x_in, y_in = parity_batch(NB_TRAIN, UPPER_BOUND) 
    x_train = np.reshape(x_in, (NB_TRAIN, 1, UPPER_BOUND))
    y_train = np.reshape(y_in, (NB_TRAIN, 1, 1))
 
    print("Saving model...")
    model.save(save_path+"model.h5")

    print("Training...")
    model.fit(x_train, y_train, epochs=NB_EPOCH, batch_size=BATCH_SIZE)
 
    print("Saving model...")
    model.save(save_path+"model.h5")

    print("Testing model...")
    x_in, y_in = parity_batch(NB_TESTS, UPPER_BOUND) 
    x_tst = np.reshape(x_in, (NB_TESTS, 1, UPPER_BOUND))
    y_tst = np.reshape(y_in, (NB_TESTS, 1, 1))
 
    pre = model.predict(x_tst)
    print("Prediction: ", pre)
    print("Solution: ", y_tst)


