#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import healpy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import statistics as sts
from scipy.linalg import lstsq

SPEED_OF_LIGHT_M_S = 2.99792458e8
PLANCK_H_MKS = 6.62606896e-34
BOLTZMANN_K_MKS = 1.3806504e-23
SOLSYSSPEED_M_S = 370082.2332
SOLSYSDIR_ECL_COLAT_RAD = 1.7656051330336222
SOLSYSDIR_ECL_LONG_RAD = 2.9958842149922833
T_CMB = 2.72548

SOLSYS_SPEED_VEC_M_S = SOLSYSSPEED_M_S * np.array(
                                                  [
                                                   np.sin(SOLSYSDIR_ECL_COLAT_RAD) * np.cos(SOLSYSDIR_ECL_LONG_RAD),
                                                   np.sin(SOLSYSDIR_ECL_COLAT_RAD) * np.sin(SOLSYSDIR_ECL_LONG_RAD),
                                                   np.cos(SOLSYSDIR_ECL_COLAT_RAD),
                                                   ]
                                                  )


def get_dipole_temperature(directions):
    """Given one or more one-length versors, return the intensity of the CMB dipole
        
        The vectors must be expressed in the Ecliptic coordinate system.
        If "freq" (frequency in Hz) is specified, the formulation will use the
        quadrupolar correction.
        """
    beta = SOLSYS_SPEED_VEC_M_S / SPEED_OF_LIGHT_M_S
    gamma = (1 - np.dot(beta, beta)) ** (-0.5)
    
    return T_CMB * (1.0 / (gamma * (1 - np.dot(directions, beta))) - 1.0)


FILENAME_POSITIONS = "satellite_positions/HFI_TOI_100-PTG_R2.01_OD0093.fits"
FILENAME_VOLTAGE = "data_voltage/HFI_TOI_100-RAW_R2.00_OD0093.fits"
DETECTOR = "100-1a"

print("-- Loading dust mask.")
#read the dust mask
MASK = healpy.read_map("HFI_dust_mask.fits.gz", verbose=False)
print("-- Dust mask loaded.")
#the nside (number of pixels) is taken as that of the mask
NSIDE = healpy.npix2nside(len(MASK))

print("-- Loading positions.")
with fits.open(FILENAME_POSITIONS) as inpf:
    theta, phi = [inpf[DETECTOR].data.field(x) for x in ("THETA", "PHI")]
    
    # Convert the sequence of positions in the sky into a sequence of pixel indexes
    #
    # NOT USED
    # pixidx = healpy.ang2pix(NSIDE, theta, phi)
    
    #get the directions (vectors) directly from the angular coordinates
    directions = healpy.ang2vec(theta, phi)
    #take of the average of two consecutive directions
    directions = (directions[:-1:2] + directions[1::2]) / 2
    #convert the directions into pixels - taking directly from the angles creates a problem with the averaging
    pixidix = healpy.vec2pix(NSIDE ,directions[:,0], directions[:,1], directions[:,2])
    
    dipole = get_dipole_temperature(directions)
    
    #create array with 1s and 0s corresponding to the positions "good" or "bad"
    flag = MASK[pixidix] == 0
    #make an array with INT type out of the bool one.
    flag_array = flag.astype(np.int)
print("-- Positions loaded.")

print("-- Cleaning data.")
#open the voltages
with fits.open(FILENAME_VOLTAGE) as f:
    obt = f["OBT"].data.field("OBT")
    data = f[DETECTOR].data.field("RAW")

#take successive differences
data = data[1::2] - data[:-1:2]

# Convert the time from OBT clock to seconds and remove the offset
time = (obt - obt[0]) / 65536
# Average the time of two adjacent samples
time = (time[:-1:2] + time[1::2]) / 2

#calculate the medians for the data and the dipole temperatures and rescale accordingly
MEDIAN_V = sts.median(data)
MEDIAN_T = sts.median(dipole)

data = data - MEDIAN_V
dipole = dipole - MEDIAN_T

#evaluate the G factor, such as V - median(V) = G • (T - median(T))
M = dipole[:, np.newaxis]*[0, 1]
p, res, rnk, s = lstsq(M, data)
G = p[1]

#take the dipole out of the data
V_CORRECT = data - G*dipole
print("-- Almost done.")
#take out from the data the directions corresponding to the galactic dust mask
clean_data = V_CORRECT[flag_array==1]
holed_raw = data[flag_array==1]

#take out also on the time - this way I can have "holes" in the graph
time_clean = time[flag_array==1]
print("-- Data cleaned.")
print("-- Saving cleaned data.")
np.savetxt("data_to_classify/time_clean.txt", time_clean)
np.savetxt("data_to_classify/clean_data.txt", clean_data)
print("-- Cleaned data saved.")

print("-- Saving sample plots.")
#
#
#
# PLOTTING -- IF DO NOT WANT PLOT, REMOVE THIS PART
#
#
#

mask = time < 30.

# plot the dipole
plt.figure(1)
plt.plot(time[mask], dipole[mask], marker='.', linestyle='none', color='#d9b83b')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Dipole temperatures")
plt.savefig("plots/dipole_example.png", dpi=600)

#plot the data - both raw and cleaned (two times, once without holes and ones with holes)
plt.figure(2)
plt.plot(time[mask], data[mask], marker='.', linestyle='none', color='#4496e7', alpha=0.9, label="with dipole")
plt.plot(time[mask], V_CORRECT[mask], marker='.', linestyle='none', color='#df413a', alpha=0.9, label="without dipole")
plt.title("Before and after dipole removal signal — with galactic dust.")
plt.xlabel("Time [s]")
plt.ylabel("Signal")
plt.legend("best")

plt.savefig("plots/gal_signal.png", dpi=600)

plt.figure(3)
mask = time_clean < 30.
plt.plot(time_clean[mask], holed_raw[mask], marker='.', linestyle='none', color='#4496e7', alpha=0.9, label="with dipole")
plt.plot(time_clean[mask], clean_data[mask], marker='.', linestyle='none', color='#df413a', alpha=0.9, label="without dipole")
plt.title("Before and after dipole removal signal — without galactic dust.")
plt.xlabel("Time [s]")
plt.ylabel("Signal")
plt.legend("best")

plt.savefig("plots/no_gal_signal.png", dpi=600)

print("-- Sample plots saved.")
