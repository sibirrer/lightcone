
import numpy as np
from lenstronomy.Util import constants


def ray_trace_raster(kwargs_model, kwargs_lens, theta_x, theta_y, n_z_bins):
    """

    :param kwargs_model: model parameters relevant for the lens model
    :param kwargs_lens: lens model keyword list
    :param theta_x: x-coordinate of rays
    :param theta_y: y-coordinate of rays
    :param n_z_bins: integer, number of redshift bins
    :return: list of rays x 2 x redshift bins, comoving distance to redshift bins
    """
    z_source = kwargs_model.get('z_source')
    cosmo = kwargs_model.get('cosmo', None)
    if cosmo is None:
        from astropy.cosmology import default_cosmology
        cosmo = default_cosmology.get()
    z_list = np.linspace(0, z_source, n_z_bins, endpoint=True)
    comoving_z = cosmo.comoving_distance(z_list)

    from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
    lens_model = MultiPlane(lens_model_list=kwargs_model.get('lens_model_list', None),
                            z_source=z_source,
                            lens_redshift_list=kwargs_model.get('lens_redshift_list'),
                            cosmo=cosmo)

    rays_ = rays(theta_x, theta_y, z_list, lens_model, kwargs_lens)
    return rays_, comoving_z.value


def rays(theta_x, theta_y, z_list, lens_model, kwargs_lens):
    """
    compute rays for each redshift bin

    :param theta_x: angle in x-direction [arcsec]
    :param theta_y: angle in y-direction [arcsec]
    :param z_list: list of redshifts
    :param lens_model: MultiPlane lens model instance
    :param kwargs_lens: lens model keyword argument list
    :return: list of rays x 2 x redshift bins
    """
    n_z_bins = len(z_list)
    rays = np.empty((len(theta_x), n_z_bins, 2))

    x0 = np.zeros_like(theta_x, dtype=float)
    y0 = np.zeros_like(theta_y, dtype=float)
    rays[:, 0, 0] = x0 * constants.arcsec
    rays[:, 0, 1] = y0 * constants.arcsec
    alpha_x = np.array(theta_x)
    alpha_y = np.array(theta_y)
    for i in range(n_z_bins-1):
        x_i, y_i, alpha_x, alpha_y = lens_model.ray_shooting_partial(x0, y0, alpha_x, alpha_y, z_start=z_list[i],
                                                                     z_stop=z_list[i+1], kwargs_lens=kwargs_lens)
        rays[:, i+1, 0] = x_i * constants.arcsec
        rays[:, i+1, 1] = y_i * constants.arcsec
        x0 = x_i
        y0 = y_i

    return rays
