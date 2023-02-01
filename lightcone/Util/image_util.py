import os
import numpy as np
import lightcone


def import_galaxy(galaxy_name='ngc1300', size=0.5):
    """
    import galaxy interpolated in lenstronomy configurations that is pre-stored in the lightcone repository

    :param galaxy_name: name of galaxy (consistent with filename)
    :type galaxy_name: string
    :param size: width of image in angular units of arc seconds as seen without lensing at the source position
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
