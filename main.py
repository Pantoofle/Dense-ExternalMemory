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
DEPTH=0

NB_TRAIN=1000
NB_TESTS=100

BATCH_SIZE=1
NB_EPOCH=10

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE)
    
    print("Building the first layer...")
    inputs = Input(shape=(SEQ_LENGTH, VECTOR_SIZE))
    densed = Dense(MEMORY_SIZE+1, name="Generator")(inputs)
    concat = keras.layers.concatenate([inputs, densed])
    inter = Model(inputs=inputs, outputs=concat)

    path = input("Path where to load the first layer: ")
    print("Loading model...")
    inter = load_model(SAVE_DIR+path+".h5",
            {'IO_Heads': IO_Heads})

    print("Compiling the first layer...")
    inter.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
 

    path = input("Path where to load/save the model: ")
   
    print("Building the rest of the layer...")
    memory = IO_Heads(memory_size=MEMORY_SIZE, 
        vector_size=VECTOR_SIZE, 
        output_size=VECTOR_SIZE,
        return_sequences=True,
        name="MAIN")(concat)
    
    print("Memory: ", memory)
    read = Lambda(lambda x: x[:, :, :3])(memory)
    mem  = Lambda(lambda x: x[:, :, 3:])(memory)

    concat2 = keras.layers.concatenate([inputs, read], axis=-1)

    post = Dense(1, activation="sigmoid")(concat2)
    
    model = Model(inputs=inputs, outputs=post)

    if len(sys.argv) > 2 and sys.argv[2]=="load":
        print("Loading the full layer...")
        model = load_model(SAVE_DIR+path+".h5",
                {'IO_Heads': IO_Heads})

    save_model(model, SAVE_DIR+path+".h5")

    model.get_layer("Generator").trainable=False
    
    print("Compiling second layer...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()
    print("Training second layer...")
    model.fit(x_in, y_in,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH)

    print("Saving the full model...")
    save_model(model, SAVE_DIR+path+".h5")
   

    print("Testing first layer...")
    x_layer1, y_layer1= memory_batch(NB_TESTS,
            SEQ_LENGTH,
            VECTOR_SIZE,
            MEMORY_SIZE)

    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH,
            VECTOR_SIZE)

    inter_pred = inter.predict(x_layer1,
            batch_size=BATCH_SIZE)

    print("Generating memory...")
    mem_watcher = Model(inputs=inputs, outputs=mem)
    mem_watcher.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    mem_img = mem_watcher.predict(x_in)
    print("mem_img: ", mem_img.shape)

    print("Testing Full model...")
    model_pred = model.predict(x_in,
            batch_size=BATCH_SIZE)

    sol = y_in.flatten()
    pred = model_pred.flatten()
    tot = 0
    for i, x in enumerate(pred):
        if x > 0.5:
            if sol[i] == 1.:
                tot += 1
        else:
            if sol[i] == 0.:
                tot += 1
    print("Score: ", tot, "/", len(sol))

    
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(inter_pred[-1])
    plt.subplot(212)
    plt.imshow(y_layer1[-1])
    plt.savefig("inter.png")

    plt.clf()
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(model_pred[-1])
    plt.subplot(212)
    plt.imshow(y_in[-1])
    plt.savefig("out.png")

    plt.clf()
    plt.figure(1)
    plt.subplot(151)
    plt.imshow(mem_img[-1][-5].reshape((MEMORY_SIZE, VECTOR_SIZE)))
    plt.subplot(152)
    plt.imshow(mem_img[-1][-4].reshape((MEMORY_SIZE, VECTOR_SIZE)))
    plt.subplot(153)
    plt.imshow(mem_img[-1][-3].reshape((MEMORY_SIZE, VECTOR_SIZE)))
    plt.subplot(154)
    plt.imshow(mem_img[-1][-3].reshape((MEMORY_SIZE, VECTOR_SIZE)))
    plt.subplot(155)
    plt.imshow(mem_img[-1][-1].reshape((MEMORY_SIZE, VECTOR_SIZE)))

    plt.savefig("memory.png") 



