import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 
from io_heads import *
from data import *

VECTOR_SIZE=50
MEMORY_SIZE=2
ENTRY_SIZE =25
SEQ_LENGTH=50
DEPTH=0

NB_TRAIN=5000
NB_TESTS=100

BATCH_SIZE=1
NB_EPOCH=500

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE)
    
    print("Building the first layer...")
    inputs = Input(shape=(SEQ_LENGTH, VECTOR_SIZE))
    densed = Dense(MEMORY_SIZE+ENTRY_SIZE*3, name="Generator")(inputs)

    path = input("Path where to load/save the model: ")
    if path == "":
        path = "full"

    memory = IO_Heads(memory_size=MEMORY_SIZE, 
        entry_size=ENTRY_SIZE, 
        return_sequences=True,
        name="MAIN")(densed)
    
    # read = Lambda(lambda x: x[:, :, :VECTOR_SIZE])(memory)
    # mem  = Lambda(lambda x: x[:, :, VECTOR_SIZE:])(memory)
    print("mem: ", memory)
    concat2 = keras.layers.concatenate([inputs, memory], axis=-1)

    print("inputs: ", inputs)
    print("conc: ", concat2)
    post = Dense(1, activation="sigmoid")(concat2)
    
    model = Model(inputs=inputs, outputs=post)

    if len(sys.argv) > 1 and sys.argv[1]=="load":
        print("Loading the full layer...")
        model = load_model(SAVE_DIR+path+".h5",
                {'IO_Heads': IO_Heads})

    model.save_weights(SAVE_DIR+path+".h5")

    print("Compiling the model...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()

    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        print("Training second layer...")
        model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                callbacks=[
                    EarlyStopping(monitor='acc', min_delta=0.005, patience = 50),
                    TensorBoard(log_dir='./logs', histogram_freq=1, 
                        write_graph=True, 
                        write_images=True)])

        print("Saving the full model...")
        save_model(model, SAVE_DIR+path+".h5")

    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH,
            VECTOR_SIZE)

    print("Testing Full model...")
    model_pred = model.predict(x_in,
            batch_size=BATCH_SIZE)

    print("Generating memory...")
    mem_watcher = Model(inputs=inputs, outputs=memory)
    mem_watcher.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    mem_img = mem_watcher.predict(x_in,
            batch_size=BATCH_SIZE)

    print("Generating last output...")
    comp = Model(inputs=inputs, outputs=concat2)
    comp.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    concaten = comp.predict(x_in,
            batch_size=BATCH_SIZE)

    print("Counting results")
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
    theoretical = (VECTOR_SIZE/SEQ_LENGTH)*(1.- (1.-1./VECTOR_SIZE)**SEQ_LENGTH)
    print("Score: ", tot, "/", len(sol), "(Pure random: ", theoretical*len(sol), ", ", theoretical," %)" )

    #  import matplotlib.pyplot as plt
    #  plt.figure(1)
    #  plt.subplot(245)
    #  plt.imshow(concaten[-1][-4].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(246)
    #  plt.imshow(concaten[-1][-3].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(247)
    #  plt.imshow(concaten[-1][-2].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(248)
    #  plt.imshow(concaten[-1][-1].reshape(1, 2*VECTOR_SIZE))
    #
    #  plt.subplot(241)
    #  plt.imshow(mem_img[-1][-4].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(242)
    #  plt.imshow(mem_img[-1][-3].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(243)
    #  plt.imshow(mem_img[-1][-2].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(244)
    #  plt.imshow(mem_img[-1][-1].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #
    #  plt.savefig("img/memory.png")

    #  plt.clf()
    #  plt.figure(1)
    #  plt.imshow(inter_pred[-1])
    #  plt.savefig("img/inter.png")
