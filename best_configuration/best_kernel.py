#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Without a rescaling factor, Keras gives problems.
RESCALE = 10000000
BATCH = 10
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

#Load train data
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
#   Code to find the best kernel size.
#
best_acc = 0
good_kernel = 0
i=0
history_kernel = np.zeros(30)
history_acc = np.zeros(30)
for KERNELS in range(1,30):
    network = models.Sequential()
    
    network.add(layers.Conv1D(1, kernel_size=KERNELS,
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

    # I dati sono già stati mischiati.
    history = network.fit(x, y[:,0], epochs=EPOCH, batch_size=BATCH, shuffle=False, validation_split=0.2, verbose=False)
    val_acc = history.history['val_accuracy'][-1]

    history_kernel[i] = KERNELS
    history_acc[i] = val_acc
    i = i+1

    if(val_acc > best_acc):
        best_acc = val_acc
        good_kernel = KERNELS
        print("\n \n ******** GOT ONE ********** \n \n Current batch: ", good_kernel, " Current Accuracy: ", best_acc,"\n \n ")


print("\n \n RISULTATI FINALI \n \n")
print("Best accuracy reached: ", best_acc)
print("Kernel with said best accuracy:", good_batch, "\n \n")

# GRAFICO
plt.plot(history_kernel[:29], history_acc[:29])
plt.xlabel('Kernel Size')
plt.ylabel('Accuracy')
plt.title('Model accuracy to kernel size.')
plt.savefig('../plots/kernels.png', dpi=600)
plt.show()
