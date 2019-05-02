import numpy as np
from vispy import app, scene
from multivol import MultiVolume
from multivol import get_translucent_cmap
from vispy import io, plot as vp

def normalize(vol_data):
    vol_data-= min(0,np.min(vol_data))
    vol_data = vol_data.astype("float64")
    vol_data *= 255.0/vol_data.max()
    return vol_data

def show_3d(volume, vol_min=-float("inf"),vol_max=float("inf")):

    fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)
    vol_data = volume
    vol_data = np.flipud(np.rollaxis(vol_data, 1))

    vol_data[vol_data > vol_max] = 0
    vol_data[vol_data < vol_min] = 0

    clim = [32, 192]
    vol_pw = fig[0, 0]

    vol_data-= min(0,np.min(vol_data))
    vol_data = vol_data.astype("float64")
    vol_data *= 255.0/vol_data.max()


    vol_pw.volume(vol_data, clim=clim,  cmap='hot')
    vol_pw.camera.elevation = 30
    vol_pw.camera.azimuth = 30
    vol_pw.camera.scale_factor /= 1.5

    shape = vol_data.shape
    fig[1, 0].image(vol_data[:, :, shape[2] // 2],   cmap='hot', clim=clim)
    fig[0, 1].image(vol_data[:, shape[1] // 2, :],   cmap='hot', clim=clim)
    fig[1, 1].image(vol_data[shape[0] // 2, :, :].T, cmap='hot', clim=clim)
    fig.show(run=True)

def show_merge_3d(volume0, volume1, vol_min=-float("inf"),vol_max=float("inf")):

    fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)

    volume0[volume0 > vol_max] = 0
    volume0[volume0 < vol_min] = 0
    volume1[volume1 > vol_max] = 0
    volume1[volume1 < vol_min] = 0

    clim = [32, 192]
    vol_pw = fig[0, 0]

    volume0 = normalize(volume0)
    volume1 = normalize(volume1)

    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    emulate_texture = False

    reds = get_translucent_cmap(1, 0, 0)
    blues = get_translucent_cmap(0, 0, 1)

    # Create the volume visuals, only one is visible
    volumes = [(volume0, None, blues), (volume1, None, reds)]
    volume = MultiVolume(volumes, parent=view.scene, threshold=0.225,
                          emulate_texture=emulate_texture)
    volume.transform = scene.STTransform(translate=(64, 64, 0))

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.
    cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                         name='Turntable')
    view.camera = cam2  # Select turntable at first

    canvas.update()
    app.run()
if __name__ == '__main__':
    vol0 = np.load(io.load_data_file(r"D:\LIDC-IDRI_npz_small\0.npz"))['arr_0']
    vol1 = np.load(io.load_data_file(r"D:\LIDC-IDRI_npz_small\1.npz"))['arr_0']
    show_merge_3d(vol0,vol1[:66,:,:], 1500)