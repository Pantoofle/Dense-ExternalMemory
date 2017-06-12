import numpy as np
import sys

sys.path.append("layers/")

from keras.models import *
from keras.layers import *

from io_memory import *

UPPER_BOUND=1000
NB_TIMESTEP=1
NB_TESTS=42

NB_EPOCH=5
BATCH_SIZE=66
MEMORY_SIZE=3
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
            output_size=ENTRY_SIZE))
        save_path = input("Enter directory where to save the model: ")

    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.summary() 
    print("Getting data...")
    x_train, y_train = read_input(INPUT_SIZE, NB_TIMESTEP, NB_TESTS)

    print("Saving...")
    model.save(save_path+"model.h5")

    print("Training...")
    model.fit(x_train, y_train, epochs=NB_EPOCH, 
            batch_size=BATCH_SIZE)
 
    model.save(save_path+"model.h5")

    print("Testing...")
    x_tst, y_tst = read_input(1, INPUT_DIM)
    res = model.evaluate(x_tst, y_tst, batch_size=BATCH_SIZE)
    print(res)
