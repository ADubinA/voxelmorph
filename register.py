"""
Example script to register two volumes with VoxelMorph models

Please make sure to use trained models appropriately. 
Let's say we have a model trained to register subject (moving) to atlas (fixed)
One could run:

python register.py --gpu 0 /path/to/test_vol.nii.gz /path/to/atlas_norm.nii.gz --out_img /path/to/out.nii.gz --model_file ../models/cvpr2018_vm2_cc.h5 
"""

# py imports
import os
import sys
from argparse import ArgumentParser
import glob

# third party
import tensorflow as tf
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import nibabel as nib
# project
import networks
import neuron.layers as nrn_layers
import losses
def register(gpu_id, moving, fixed, model_file, out_img, out_warp):
    """
    register moving and fixed. 
    """  
    assert model_file, "A model file is necessary"
    assert out_img or out_warp, "output image or warp file needs to be specified"

    # GPU handling
    if gpu_id is not None:
        gpu = '/gpu:' + str(gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        gpu = '/cpu:0'

    # load data

    fix = nib.load(fixed).get_data()[np.newaxis, ...,:48, np.newaxis]
    mov = nib.load(moving).get_data()[np.newaxis, ...,:48, np.newaxis]

    with tf.device(gpu):
        # load model
        custom_layers = {'SpatialTransformer':nrn_layers.SpatialTransformer,
                         'VecInt':nrn_layers.VecInt,
                         'Sample':networks.Sample,
                         'Rescale':networks.RescaleDouble,
                         'Resize':networks.ResizeDouble,
                         'Negate':networks.Negate,
                         "loss":losses.Grad('l2').loss}

        net = keras.models.load_model(model_file, custom_objects=custom_layers)

        # register
        [moved, warp] = net.predict([mov, fix])
    # fig, ax = plt.subplots(nrows=1, ncols=4)
    # ax[0].imshow(fix[0,0,...,0])
    # ax[1].imshow(mov[0,0,...,0])
    # ax[2].imshow(moved[0,0,...,0])
    # ax[3].imshow(warp[0,0,...,0])
    #
    # plt.show()
    # output image
    if out_img is not None:
        img = moved[0,...,0]
        print("saving image to: " + "out_img")
        img = nib.Nifti1Image(img, np.eye(4))
        nib.save(img, out_img)

    # output warp
    if out_warp is not None:
        img = warp[0,...]
        np.savez(out_warp, img)


def mass_registar(gpu_id, fixed, folder_in, model_file, folder_out):

    if not os.path.exists(folder_in):
        os.mkdir(folder_in)

    if not os.path.exists(folder_out):
        os.mkdir(folder_out)


    for file_path in glob.glob(os.path.join(folder_in, "*.npz")):
        file_name = os.path.split(file_path)[-1]
        register(gpu_id=gpu_id,
                 fixed=fixed,
                 moving=file_path,
                 model_file=model_file,
                 out_img=None,
                 out_warp=os.path.join(folder_out, file_name.replace(".", "_warp.")))
if __name__ == "__main__":
    # parser = ArgumentParser()
    #
    # # positional arguments
    # parser.add_argument("moving", type=str, default="./data/test_vol.npz",
    #                     help="moving file name")
    # parser.add_argument("fixed", type=str, default="./data/atlas_norm.npz",
    #                     help="fixed file name")
    #
    # # optional arguments
    # parser.add_argument("--model_file", type=str,
    #                     dest="model_file", default=r'C:\Users\almog\dev\models\1499.h5',
    #                     help="models h5 file")
    # parser.add_argument("--gpu", type=int, default=None,
    #                     dest="gpu_id", help="gpu id number")
    # parser.add_argument("--out_img", type=str, default=True,
    #                     dest="out_img", help="output image file name")
    # parser.add_argument("--out_warp", type=str, default=None,
    #                     dest="out_warp", help="output warp file name")
    #
    # args = parser.parse_args()
    # register(**vars(args))
    register(
                gpu_id    = 0,
                fixed     = r'D:\head-neck-reg-small\ct\HN-CHUM-007.nii.gz',
                moving    = r'D:\head-neck-reg-small\ct\HN-CHUM-010.nii.gz',
                model_file= r"C:\Users\almog\dev\models\93.h5",
                out_img   = r"D:\output.nii.gz",
                out_warp  =None)
    # mass_registar(gpu_id=0,
    #               fixed     ="/home/almogdubin/datadrive/LIDC-IDRI_npz_small/0.npz",
    #               folder_in ="/home/almogdubin/datadrive/LIDC-IDRI_npz_small",
    #               model_file="/home/almogdubin/datadrive/voxelmorph-model-weights/1500.h5",
    #               folder_out="/home/almogdubin/datadrive/small_register")
