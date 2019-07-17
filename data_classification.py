#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

time_clean = np.loadtxt("data_to_classify/time_clean.txt")
print("-- Time loaded")
clean_data = np.loadtxt("data_to_classify/clean_data.txt")
print("-- Data loaded")

# I use this interval as the max two points can be from each other - it's a little to much
limit_interval = 1.1*(time_clean[50] - time_clean[49])
# Ho preso numeri così alti perché, con questo threshold, per puro caso tra i primi c'erano dei salti

#I set number of points displayed per random graph
INTERVAL_LENGTH = 100
begin = 0
end = len(time_clean)-1

n = 0
N_GLITCH = 0
N_LONG = 0
N_NO_GLITCH = 0

while(n<2000):
    
    begin = 0
    end = len(time_clean)-1
    #choose a random starting point. If the difference in time is too big, it means I'm taking the data over a "hole".
    while((time_clean[end] - time_clean[begin]) > INTERVAL_LENGTH * limit_interval):
        begin = int((len(time_clean)-301)*np.random.random_sample())
        end = begin + INTERVAL_LENGTH
    
    #plot the data
    plt.figure(1)
    plt.plot(time_clean[begin:end], clean_data[begin:end], marker='.', linestyle='dashed', color='#6a97be', alpha=1)
    plt.show()
    print('--Step number: ', n)
    print('Do you see a glitch? [ yes (y) / no (n) / long (q) / not sure (s) / beautiful glitch (b) / beautiful no-glitch (a) / beautiful long (z)]')
    answer = input()

    if(answer == 'y'):
        print('Glitch detected.')
        N_GLITCH = N_GLITCH + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/glitch/train_" + str(N_GLITCH) +".txt")
        #healpy.write_map(OUTPUT_FILE, clean_data[begin:end], coord='E')
        # There's something wrong with the line of code above. I will try to figure out why later.
        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))
    elif(answer == 'b'):
        print('Glitch detected. So beautiful I save the plot.')
        N_GLITCH = N_GLITCH + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/glitch/train_" + str(N_GLITCH) +".txt")
        
        plt.figure(2)
        plt.plot(time_clean[begin:end], clean_data[begin:end], marker='.', linestyle='dashed', color='blue', alpha=1)
        plt.savefig("plots/glitch" + str(N_GLITCH) + ".png", dpi=600)
        
        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))

    elif(answer == 'n'):
        print('No glitch detected.')
        N_NO_GLITCH = N_NO_GLITCH + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/no_glitch/train_" + str(N_NO_GLITCH) +".txt")
        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))

    elif(answer == 'a'):
        print('No glitch detected. So beautiful I save the plot.')
        N_NO_GLITCH = N_NO_GLITCH + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/no_glitch/train_" + str(N_NO_GLITCH) +".txt")
        
        plt.figure(2)
        plt.plot(time_clean[begin:end], clean_data[begin:end], marker='.', linestyle='dashed', color='blue', alpha=1)
        plt.savefig("plots/no_glitch" + str(N_NO_GLITCH) + ".png", dpi=600)
        
        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))

    elif(answer == 'q'):
        print('Long glitch detected.')
        N_LONG = N_LONG + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/long_glitch/train_" + str(N_LONG) +".txt")

        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))


    elif(answer == 'z'):
        print('Long glitch detected. So beautiful I save the plot.')
        N_LONG = N_LONG + 1
        n = n+1
        OUTPUT_FILE = ("Train Data/long_glitch/train_" + str(N_LONG) +".txt")
        
        plt.figure(2)
        plt.plot(time_clean[begin:end], clean_data[begin:end], marker='.', linestyle='dashed', color='blue', alpha=1)
        plt.savefig("plots/long_glitch" + str(N_LONG) + ".png", dpi=600)

        np.savetxt(OUTPUT_FILE, (time_clean[begin:end], clean_data[begin:end]))
    else:
        print('Unable to classify.')

