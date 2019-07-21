#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

#
#   ##  DESCRIPTION  ##
#   This a very simple testing script. It tests how well the
#   network performs on new 100 new test data.
#
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from statistics import median

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix

RESCALE = 10000000
KERNEL = 1
FEATURES = 8
BATCH = 10
EPOCH = 40
DATA_SHAPE = 100

#function that loads files for the network
def load_files(filenames, category, num_categories):
    #the 1D convnet needs to specify the third axis of the tensor
    glitches = np.zeros((len(filenames), DATA_SHAPE, 1))
    category_matrix = np.zeros((len(filenames), num_categories))
    for i in range(len(filenames)):
        current_data = np.loadtxt(filenames[i])
        glitches[i,:,0] = current_data[1]
        #I save 1s on the axis for that respective category
        category_matrix[i,category] = 1
    return glitches, category_matrix

glitches, cat_glitches = load_files(glob("../Test Data/glitch/*.txt"), 0, 2)
no_glitches, cat_no_glitches = load_files(glob("../Test Data/no_glitch/*.txt"), 1, 2)

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

my_model = models.load_model("../networks/glitch_detector.h5")
print("\n Testing on new data.\n")
my_model.evaluate(x,y[:,0])
print("\n")

y_pred = my_model.predict(x)

# I have to feed to the function "confusion_matrix" array of 1s and 0s
PREDICTION_THRESHOLD = 0.5  #I set a threshold for the prediction
y_pred = y_pred[:,0]+PREDICTION_THRESHOLD

my_matrix = confusion_matrix(y[:,0].astype(int), y_pred.astype(int))
print("Confusion matrix \n", my_matrix, end="\n")
print("\n Description: \n [[ good 1s, false 0s] \n [false 1s, good 0s]] \n")
np.savetxt("Confusion_matrix.txt", my_matrix, fmt="%i")

plt.figure(figsize=(2,2))
tb = plt.table(cellText=my_matrix, loc=(0,0), cellLoc='center')
tc = tb.properties()['child_artists']
tb.set_fontsize(20)
for cell in tc:
    cell.set_height(1/2)
    cell.set_width(1/2)

ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_ticks_position('top')
plt.title("Confusion matrix for 100 long test dataset.")
plt.xlabel("Predict class")
plt.ylabel("True class")
plt.savefig("Confusion_matrix.png", dpi=600)
