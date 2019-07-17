#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import healpy
import numpy as np
import os
import matplotlib.pyplot as plt

# Change this to find a suitable mask
THRESHOLD = 0.005

# Load the HFI 353 GHz map from the FITS file downloaded from the Planck Legacy Archive (PLA)
hfi353 = healpy.read_map("Milky_Way/HFI_SkyMap_353-psb_2048_R3.01_full.fits", verbose=False)

# Reduce the resolution of the map in order to save memory and computational time
hfi353 = healpy.ud_grade(hfi353, 256)

#plot galaxy map without threshold - normalized as hist
healpy.mollview(hfi353, coord='GE', norm="hist")
plt.savefig('plots/elliptic_galaxy_map.png', dpi=600)

healpy.mollview(hfi353, norm="hist")
plt.savefig('plots/galactic_galaxy_map.png', dpi=600)

#check if the dust mask already exists
exists = os.path.isfile("HFI_dust_mask.fits.gz")
if exists:
    #if it exists, remove it
    print("-- Dust mask file already present. Il will be rewritten.")
    os.remove("HFI_dust_mask.fits.gz")


# Rotate the map from Galactic to Ecliptic coordinates
rotator = healpy.rotator.Rotator(coord=['G','E'])
hfi353 = rotator.rotate_map_pixel(hfi353)

# Apply a smoothing filter to the map
hfi353 = healpy.smoothing(hfi353, fwhm=np.deg2rad(1.0), verbose=False)

# Normalize the pixel values
hfi353 -= np.min(hfi353)
hfi353 /= np.max(hfi353)

# Clip the values
hfi353[hfi353 <= THRESHOLD] = 0
hfi353[hfi353 > THRESHOLD] = 1

# Save the map in a new file
healpy.write_map("HFI_dust_mask.fits.gz", hfi353, coord='E')

#save the map
dust_map = healpy.read_map("HFI_dust_mask.fits.gz", verbose=False)

#save picture of the dust map
healpy.mollview(dust_map)
plt.show()
plt.savefig('plots/eliptic_galaxy_mask.png', dpi=600)
