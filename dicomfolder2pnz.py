from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pydicom
import glob
import os
from skimage import measure
from plotly.offline import iplot
from plotly.tools import FigureFactory as FF
import logging
import cv2
from stl import mesh
# import vtk.

# def decimate(mesh, multiplier=0.25):
#     mesh = vtk.


def load_dicom_image_folder(folder):
    """
    Load a folder of 2D dicom images and return a 3d volumetric numpy object
    will ignore files that are not in proper format
    Args:
        folder(str):
            the folder where the dicom images are.

    Returns:
        numpy.ndarray of the volume
    """
    volume = np.array([])
    count = 0
    datas = []
    for file in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
        datas.append(pydicom.dcmread(file))
    datas = sorted(datas, key=lambda d: d.InstanceNumber)

    for data in datas:
        if hasattr(data, "pixel_array"):
            if count == 0:
                volume = np.array([data.pixel_array])
            else:
                volume = np.append(volume, np.array([data.pixel_array]), axis=0)
            count += 1
        else:
            raise ValueError("bad dicom in folder: " + folder)
    return volume

def load_png(files, subdir):
    count = 0
    for image_path in files:
        image_path = os.path.join(subdir, image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[image < 125] = 0
        if count == 0:
            volume = [image]
        else:
            volume = np.append(volume, [image], axis=0)
        count += 1

def make_mesh(image, threshold=100, step_size=1):
    p = image
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True)
    return verts, faces

def make_meshv2(volume, subdir):
    v, f = make_mesh(volume, threshold=100, step_size=2)
    # Create the mesh
    cube = mesh.Mesh(np.zeros(f.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(f):
        for j in range(3):
            cube.vectors[i][j] = v[f[j], :]

    # Write the mesh to file "cube.stl"
    cube.save(os.path.join(subdir, 'mesh.stl'))
    print("save in " + subdir)
    # plt_3d(v, f)

def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor([0,0,0])
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    # ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    print("showing")
    plt.show()

def folder2npz(input_folder, save_location, min_number_of_files=10):
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    count = 0
    for subdir, dirs, files in os.walk(input_folder):

        # if len(dirs) != 0:
        #     logging.warning("bad format of directories")
        #     continue

        if len(files) < min_number_of_files:
            logging.warning("found folder that is too small, not including folder:  {}".format(subdir))
            continue

        volume = load_dicom_image_folder(subdir)
        np.savez_compressed(os.path.join(save_location, str(count)), volume)
        print(count)
        count += 1



if __name__ == "__main__":
    # parser = ArgumentParser()
    #
    # # positional arguments
    # parser.add_argument("moving", type=str, default=None,
    #                     help="location of the folder containing all the subfolders fo dcm")
    #
    #
    # args = parser.parse_args()
    # folder2npz(**vars(args)[0])
    folder2npz("/home/almogdubin/datadrive/LIDC-IDRI",
               "/home/almogdubin/datadrive/LIDC-IDRI_npz")




