import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
import keras.layers
from builder import *
from data import *

VECTOR_SIZE=3
MEMORY_SIZE=10
SEQ_LENGTH=5

NB_TRAIN=1000
NB_TESTS=100

BATCH_SIZE=1
NB_EPOCH=10

SAVE_DIR="models/"

if __name__ == "__main__":
    path = input("Save to: ")
    x_layer1, y_layer1= memory_batch(NB_TRAIN,
            SEQ_LENGTH,
            VECTOR_SIZE,
            MEMORY_SIZE)

    print("Building the first layer...")
    inputs = Input(shape=(SEQ_LENGTH, VECTOR_SIZE))
    densed = Dense(MEMORY_SIZE+1, name="Generator")(inputs)
    concat = keras.layers.concatenate([inputs, densed])
    inter = Model(inputs=inputs, outputs=concat)
    
    print("Compiling the first layer...")
    inter.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
 
    print("Training first layer...")
    inter.fit(x_layer1, y_layer1,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=0.0001)])
    save_model(inter, SAVE_DIR+path+".h5")
 
    print("Testing first layer...")
    x_layer1, y_layer1= memory_batch(NB_TESTS,
            SEQ_LENGTH,
            VECTOR_SIZE,
            MEMORY_SIZE)

    inter_eval = inter.evaluate(x_layer1, y_layer1,
            batch_size=BATCH_SIZE)

    print("\nEvaluation: ", inter_eval)














