from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

def equalize_volume(volume):
    volume =  exposure.rescale_intensity(exposure.equalize_hist(volume), out_range=(0, 255))

    return volume.astype("int16")

if __name__ == '__main__':
    vol0 = np.load((r"/home/almogdubin/datadrive/LIDC-IDRI_npz_small/0.npz"))['arr_0']
    plt.imshow(equalize_volume(vol0)[5])
    plt.show()