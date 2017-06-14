import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *

from io_memory import *
from data import *

UPPER_BOUND=100
MEMORY_SIZE=10
NB_TIMESTEP=1

NB_TRAIN=5000
NB_TESTS=10

NB_EPOCH=50
BATCH_SIZE=50
ENTRY_SIZE=UPPER_BOUND

DELTA_STOP=0.001

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        print("Loading model...")
        model = load_model(SAVE_DIR+sys.argv[2], {'IO_Layer': IO_Layer})
        save_name = sys.argv[2]
    else:
        print("Building new model...")
        model = Sequential()
        model.add(IO_Layer(input_shape=(NB_TIMESTEP, UPPER_BOUND),
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE,
            output_size=1))
        # model.add(Dense(1, activation='sigmoid'))
        save_name = input("Enter model name: ")

    print("Compiling...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.summary() 
    print("Input shape: ", model.input_shape)
    print("Getting data...")
    x_in, y_in = parity_batch(NB_TRAIN, UPPER_BOUND) 
    x_train = np.reshape(x_in, (NB_TRAIN, 1, UPPER_BOUND))
    y_train = np.reshape(y_in, (NB_TRAIN, 1, 1))
 
    print("Saving model...")
    model.save(SAVE_DIR+save_name)


    print("Training...")
    print("Variables: ")
    print("UPPER_BOUND: ", UPPER_BOUND)
    print("MEMORY_SIZE: ", MEMORY_SIZE)
    print("NB_TRAIN:    ", NB_TRAIN)
    print("NB_EPOCH:    ", NB_EPOCH)
    print("BATCH_SIZE:  ", BATCH_SIZE)

    check_point = ModelCheckpoint(SAVE_DIR+save_name)
    log_export = TensorBoard(log_dir='./logs', 
            histogram_freq=0, 
            batch_size=32, 
            write_graph=True, 
            write_grads=False, 
            write_images=False, 
            embeddings_freq=0, 
            embeddings_layer_names=None, 
            embeddings_metadata=None)
    early_stop = EarlyStopping(monitor='loss', min_delta=DELTA_STOP, patience=0, verbose=0, mode='auto')


    model.fit(x_train, y_train, 
            epochs=NB_EPOCH, 
            batch_size=BATCH_SIZE,
             callbacks=[check_point, log_export, early_stop])
 
    print("Saving model...")
    model.save(SAVE_DIR+save_name)

    print("Testing model...")
    x_in, y_in = parity_batch(NB_TESTS, UPPER_BOUND) 
    x_tst = np.reshape(x_in, (NB_TESTS, 1, UPPER_BOUND))
    y_tst = np.reshape(y_in, (NB_TESTS, 1, 1))
 
    pre = model.predict(x_tst)
    pre = [p > 0.5 for p in pre]
    res = [p > 0.5 for p in y_tst]
    ok = sum([1 for x, y in zip(pre, res) if x == y  ])
    print("Nb tests:   ", NB_TESTS)
    print("Nb success: ", ok)

