import numpy as np
import matplotlib.pyplot as plt

with np.load('/home/almogdubin/datadrive/LIDC-IDRI_npz/0.npz') as data:
    plt.imshow(data["arr_0"][0])
    plt.show()