import os
import imageio
import numpy as np
import lightcone
import scipy.ndimage
from lenstronomy.Util import image_util


def import_galaxy(galaxy_name='ngc1300', size=0.5):
    """
    import galaxy interpolated in lenstronomy configurations
    """

    # find path to data
    path = os.path.dirname(lightcone.__file__)
    module_path, _ = os.path.split(path)
    #module_path, _ = os.path.split(dirpath)
    ngc_filename = os.path.join(module_path, 'Data/Galaxies/'+galaxy_name+'.npy')
    # read data
    ngc_data_resized = np.load(ngc_filename)

    source_light_model_list = ['INTERPOL']
    scale = size / len(ngc_data_resized)

    kwargs_source = [{'image': ngc_data_resized, 'amp': 1, 'center_x': 0, 'center_y': 0, 'phi_G': 0, 'scale': scale}]
    return kwargs_source, source_light_model_list
