#https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data.
#The catalogue must be downloaded and unzipped.

import numpy as np
from scipy.stats import norm, truncnorm, uniform
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.cosmos import COSMOSExcludeCatalog, COSMOSIncludeCatalog
import pandas as pd
import os


kwargs_numerics = {'supersampling_factor':1}

numpix = 128

mask_radius = 0.5
mag_cut = 2.0

output_ab_zeropoint = 25.127

os.chdir(os.path.dirname(os.path.abspath(__file__)))
cosmos_folder = r'C:\Users\jon\lightconeTests\cosmos\COSMOS_23.5_training_sample'

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': truncnorm(-20,np.inf,loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-1.1/0.15,np.inf,loc=1.1,scale=0.15).rvs,
			'e1': norm(loc=0.0,scale=0.1).rvs,
			'e2': norm(loc=0.0,scale=0.1).rvs,
			'center_x': norm(loc=0.0,scale=0.16).rvs,
			'center_y': norm(loc=0.0,scale=0.16).rvs,
			'gamma1': norm(loc=0.0,scale=0.05).rvs,
			'gamma2': norm(loc=0.0,scale=0.05).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'source':{
		'class': COSMOSExcludeCatalog,
		'parameters':{
			'z_source':1.5,'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
			'smoothing_sigma':0.00,'random_rotation':True,
			'output_ab_zeropoint':output_ab_zeropoint,
			'center_x':norm(loc=0.0,scale=0.16).rvs,
			'center_y':norm(loc=0.0,scale=0.16).rvs,
			'min_flux_radius':10.0, 'source_exclusion_list':pd.read_csv(
					'bad_galaxies.csv',
					names=['catalog_i'])['catalog_i'].to_numpy()}
        #'class': COSMOSIncludeCatalog,
        #        'parameters':{
        #            'z_source':1.5,'cosmos_folder':cosmos_folder,
        #            'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
        #            'smoothing_sigma':0.00,'random_rotation':True,
        #            'output_ab_zeropoint':output_ab_zeropoint,
        #            'center_x':norm(loc=0.0,scale=0.16).rvs,
        #            'center_y':norm(loc=0.0,scale=0.16).rvs,
        #            'min_flux_radius':10.0, 'source_inclusion_list': 47748}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'GAUSSIAN',
			'fwhm': 0.03
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.040,'ccd_gain':1.58,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1380,'sky_brightness':21.83,
			'num_exposures':4,'background_noise':None
		}
	}
}
