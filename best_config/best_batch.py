#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#
#   ###     DESCRIPTION     ###
#   This script runs a new network that optimises the batch size.
#   The networks that are found here are of no use and are not saved.
#
#

# A rescaling factor is needed for the data, otherwise the neural network doens't work
RESCALE = 10000000
# I set the other variables to reasonable ones.
KERNEL = 1
FEATURES = 8
EPOCH = 40

#function that loads files for the network
def load_files(filenames, category, num_categories):
    #the 1D convnet needs to specify the third axis of the tensor
    glitches = np.zeros((len(filenames), 100, 1))
    category_matrix = np.zeros((len(filenames), num_categories))
    for i in range(len(filenames)):
        current_data = np.loadtxt(filenames[i])
        glitches[i,:,0] = current_data[1]
        #I save 1s on the axis for that respective category
        category_matrix[i,category] = 1
    return glitches, category_matrix

# I load the files
glitches, cat_glitches = load_files(glob("../Train Data/glitch/*.txt"), 0, 2)
no_glitches, cat_no_glitches = load_files(glob("../Train Data/no_glitch/*.txt"), 1, 2)

x = np.concatenate((glitches, no_glitches))
y = np.concatenate((cat_glitches, cat_no_glitches))

# traslate the data on on each median
for i in range(len(x)):
    x[i,:,0] = x[i,:,0] - median(x[i,:,0])

# rescale the data: the network gives problems with too small data.
# The rescaling is arbitrary, but it works.
for j in range(len(x)):
    for i in range(100):
        x[j,i,0] = x[j,i,0] * RESCALE

# I run a permutation, in order to mix up the data
perm = np.random.permutation(x.shape[0])
# I want to have the same permutation on both x and y
np.take(x, perm, axis=0, out=x)
np.take(y, perm, axis=0, out=y)

time_length = len(x[0,:,0])


#
#   Code that find the best batch. It runs for all possible
#   batches between 1 and 120.
#
best_acc = 0
good_batch = 0

history_batches = np.zeros(119)
history_acc = np.zeros(119)
i=0
for BATCHES in range(1,120):
    
    network = models.Sequential()
    network.add(layers.Conv1D(FEATURES,
                              kernel_size=KERNEL,
                              activation='relu',
                              input_shape=(time_length,1)))
        
    network.add(layers.GlobalMaxPooling1D())

    network.add(layers.Dense(64, activation='softmax'))
    network.add(layers.Dropout(0.6))
    network.add(layers.Dense(64, activation='softmax'))
    network.add(layers.Dense(1, activation='sigmoid'))

    network.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = network.fit(x, y[:,0], epochs=EPOCH, batch_size=BATCHES, shuffle=False, validation_split=0.2, verbose=False)

    val_acc = history.history['val_accuracy'][-1]

    history_acc[i] = val_acc
    history_batches[i] = BATCHES
    i = i+1

    if(val_acc > best_acc):
        best_acc = val_acc
        good_batch = BATCHES
        print("\n \n ******** GOT ONE ********** \n \n Current batch: ", good_batch, " Current Accuracy: ", best_acc,"\n \n ")

print("\n \n RISULTATI FINALI \n \n")
print("Best accuracy reached: ", best_acc)
print("Batch with said best accuracy:", good_batch, "\n \n")


# GRAFICO
plt.plot(history_batches, history_acc)
plt.xlabel('Batch size')
plt.ylabel('Accuracy')
plt.title('Model accuracy to batch size.')
plt.savefig('../plots/batches.png', dpi=600)
plt.show()
