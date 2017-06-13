import numpy as np
import sys

sys.path.append("layers/")

from keras.models import *
from keras.layers import *

from io_memory import *
from data import *

UPPER_BOUND=1000
NB_TIMESTEP=1
NB_TESTS=1000

NB_EPOCH=50
BATCH_SIZE=50
MEMORY_SIZE=100
ENTRY_SIZE=UPPER_BOUND

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        model = load_model(sys.argv[2])
        save_path = sys.argv[2]
    else:
        model = Sequential()
        model.add(IO_Layer(input_shape=(NB_TIMESTEP, UPPER_BOUND),
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE,
            output_size=1))
        save_path = input("Enter directory where to save the model: ")

    print("Compiling...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.summary() 
    print("Input shape: ", model.input_shape)
    print("Getting data...")
    x_in, y_in = parity_batch(NB_TESTS, UPPER_BOUND) 
    x_train = np.reshape(x_in, (NB_TESTS, 1, UPPER_BOUND))
    y_train = np.reshape(y_in, (NB_TESTS, 1, 1))
    print("Saving...")
    model.save(save_path+"model.h5")

    print("Training...")
    model.fit(x_train, y_train, epochs=NB_EPOCH, batch_size=BATCH_SIZE)
 
    model.save(save_path+"model.h5")
