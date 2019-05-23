from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

def equalize_volume(volume):
    volume = exposure.rescale_intensity(exposure.equalize_hist(volume), out_range=(0, 255))
    return volume.astype("int16")

def add_noise(volume, precent, value):
    s = np.random.uniform(0, 1, volume.shape)
    s[s>precent] = 1
    volume[s==1] += value
    return volume
if __name__ == '__main__':
    vol0 = np.load((r"D:/LIDC-IDRI_npz_small/0.npz"))['arr_0']
    add_noise(vol0,0.2, 5)
    pass