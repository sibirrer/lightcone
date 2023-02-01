import numpy as np
from lightcone import ray_trace
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from lenstronomy.LensModel.MultiPlane.multi_plane import MultiPlane
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Util import util
from lenstronomy.Util.class_creator import create_class_instances
from lenstronomy.ImSim.image2source_mapping import Image2SourceMapping
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import constants


class Plot3d(object):
    """
    class to manage plotting settings for 3d plot
    """

    def __init__(self, kwargs_model, kwargs_lens, kwargs_source, kwargs_pixel_grid, n_z_bins):
        """

        :param kwargs_model: keyword arguments of the lenstronomy models in same format as lenstronomy uses it
        :param kwargs_lens: keyword argument list of the lens model parameters in lenstronomy conventions
        :param kwargs_source: keyword argument list of the extended source surface brightness model parameters in
         lenstronomy conventions
        :param kwargs_pixel_grid: keyword arguments that go into the lenstronomy PixelGrid() class instantization
        :param n_z_bins: integer, number of redshift bins in the animation
        """
        self._n_z_bins = n_z_bins
        # TODO: make a raise statement demanding that kwargs_model needs a keyword 'z_source' to be the source redshift
        z_source = kwargs_model.get('z_source')
        z_lens = kwargs_model.get('z_lens', None)
        # if no particular lens redshift is set, then takes the first element in the lens_redshift_list_setting
        if z_lens is None:
            z_lens = kwargs_model.get('lens_redshift_list')[0]
        cosmo = kwargs_model.get('cosmo', None)
        if cosmo is None:
            from astropy.cosmology import default_cosmology
            cosmo = default_cosmology.get()
        self._z_list = np.linspace(0, z_source, n_z_bins, endpoint=True)
        self._comoving_z = cosmo.comoving_distance(self._z_list).value

        self._comoving_zs = self._comoving_z[-1]
        self._comoving_zl = self._comoving_zs / 2 # cosmo.comoving_distance(z_lens).value

        self._lens_model = MultiPlane(lens_model_list=kwargs_model.get('lens_model_list', None),
                                z_source=z_source,
                                lens_redshift_list=kwargs_model.get('lens_redshift_list'),
                                cosmo=cosmo)
        # initiate pixel grid
        self._pixel_grid = PixelGrid(**kwargs_pixel_grid)
        theta_x, theta_y = self._pixel_grid.pixel_coordinates

        self._xx_l = theta_x * self._comoving_zl * constants.arcsec * 1000
        self._yy_l = theta_y * self._comoving_zl * constants.arcsec * 1000
        theta_x = util.image2array(theta_x)
        theta_y = util.image2array(theta_y)

        # thinned-out the pixel grid
        num_rays = 20
        # pixel_width = self._pixel_grid.pixel_width
        nx, ny = self._pixel_grid.num_pixel_axes
        low_res_factor = nx / num_rays
        kwargs_ray_grid = {'nx': num_rays,
                           'ny': num_rays,
                           'ra_at_xy_0': kwargs_pixel_grid['ra_at_xy_0'],
                           'dec_at_xy_0': kwargs_pixel_grid['dec_at_xy_0'],
                           'transform_pix2angle': kwargs_pixel_grid['transform_pix2angle'] * low_res_factor}
        pixel_grid_ray = PixelGrid(**kwargs_ray_grid)
        theta_x_ray, theta_y_ray = pixel_grid_ray.pixel_coordinates
        self._theta_x_ray = util.image2array(theta_x_ray)
        self._theta_y_ray = util.image2array(theta_y_ray)

        # perform ray-tracing
        self._rays = ray_trace.rays(self._theta_x_ray, self._theta_y_ray, self._z_list, self._lens_model, kwargs_lens)

        lens_model_class, source_model_class, lens_light_model_class, point_source_class, _ = create_class_instances(
            **kwargs_model)
        source_mapping = Image2SourceMapping(lensModel=lens_model_class, sourceModel=source_model_class)

        flux_image = source_mapping.image_flux_joint(theta_x, theta_y, kwargs_lens, kwargs_source, k=None)
        flux_image[flux_image <= 10**(-10)] = 10**(-10)
        self._image = util.array2image(flux_image)

        # setting the frame
        self._min_x, self._max_x, self._min_y, self._max_y = self.frame_setting()

        self._flux_ray = source_mapping.image_flux_joint(self._theta_x_ray, self._theta_y_ray, kwargs_lens,
                                                         kwargs_source, k=None)

        # calculate source
        self._xx_s, self._yy_s = np.meshgrid(np.linspace(self._min_x, self._max_x, 200), np.linspace(self._min_y, self._max_y, 200))

        # angular coordinates in the source
        beta_x = self._xx_s / 1000 / self._comoving_zs / constants.arcsec
        beta_y = self._yy_s / 1000 / self._comoving_zs / constants.arcsec

        light_model = LightModel(light_model_list=kwargs_model.get('source_light_model_list', None))
        # surface brightness in the source
        self._source = light_model.surface_brightness(beta_x, beta_y, kwargs_source)

    def plot3d(self, fig, angle1, angle2, n_ray, plot_source=True, plot_lens=True, plot_rays=True, alpha_lens=1,
               alpha_source=1):
        """
        3d plot of a ray shooting state with (partially) complete rays

        :param fig: matplotlib figure class
        :param angle1: float, rotation angle under which the ray-shooting animation is projected
        :param angle2: float, rotation angle under which the ray-shooting animation is projected
        :param n_ray: integer, step from which to the source the light rays are displayed
        :param plot_source: boolean, if True, plots the source
        :param plot_lens: boolean, if True, plots lens
        :param plot_rays: boolean, if True, plots light rays (to n_ray step)
        :param alpha_lens: float in [0, 1], transparency of the lens
        :param alpha_source: float in [0, 1], transparency of the source

        """
        ax = plt.axes(projection='3d')
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1., 0.5, 0.5, 1.]))
        ax.set_facecolor("black")
        ax.set_axis_off()
        plt.style.use('dark_background')

        ax.view_init(angle1, angle2)

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        # ax.set_xlabel('co-moving distance [Mpc]')
        # ax.set_ylabel('[kpc]')
        # ax.set_zlabel('[kpc]')
        ax.grid(False)

        ax.set_xlim([0, np.max(self._comoving_z)])
        ax.set_ylim([self._min_x, self._max_x])
        ax.set_zlim([self._min_y, self._max_y])

        # plot source

        if plot_source is True:
            Zs = self.ray_colors(flux=self._source)
            cset = ax.plot_surface(np.zeros_like(Zs) + self._comoving_zs, self._xx_s, self._yy_s,
                                   facecolors=plt.cm.gist_heat(Zs), shade=False, zorder=0, alpha=alpha_source)

        # plot light rays
        if plot_rays is True:
            Zr = self.ray_colors(flux=self._flux_ray)
            for i in range(len(self._theta_x_ray)):
                # alpha = abs(flux[i])/np.max(flux)
                if Zr[i] > 0.2:
                    ax.plot(self._comoving_z[n_ray:], self._rays[i, n_ray:, 0] * 1000, self._rays[i, n_ray:, 1] * 1000,
                            color=plt.cm.gist_heat(Zr[i]), alpha=1,  # alpha=Zr[i],
                            linewidth=0.1, zorder=3)

        # plot image as seen by the observer
        if plot_lens is True:
            Zl = self.ray_colors(flux=self._image)
            cset = ax.plot_surface(np.zeros_like(Zl) + self._comoving_zl, self._xx_l, self._yy_l, facecolors=plt.cm.gist_heat(Zl),
                                   shade=False, zorder=10, alpha=alpha_lens)
        fig.tight_layout()
        return fig

    def frame_setting(self):
        """
        calculates minimum and maximum range of plot in units co-moving kpc in projected axis

        :return: min_x, max_x, min_y, max_y
        """
        min_x, max_x, mean_x = np.min(self._xx_l), np.max(self._xx_l), np.mean(self._xx_l)
        min_y, max_y, mean_y = np.min(self._yy_l), np.max(self._yy_l), np.mean(self._yy_l)

        delta_x = max_x - min_x
        delta_y = max_y - min_y
        delta = np.maximum(delta_x, delta_y)
        min_x, max_x = mean_x - delta / 2, mean_x + delta / 2
        min_y, max_y = mean_y - delta / 2, mean_y + delta / 2
        return min_x, max_x, min_y, max_y

    def ray_colors(self, flux):
        """
        setting for colors of the light rays and image/source

        :param flux: value or numpy array of flux values
        :return: scaled [0, 1] normalized value for color plotting
        """
        log_flux = np.log10(self._image)
        Z_max = np.max(log_flux)
        Z_min = max(np.min(log_flux), -10)
        if Z_max - Z_min > 3:
            Z_min = Z_max - 3

        # normalize Z to [0..1]
        Z = np.log10(np.maximum(flux, 10**(-10))) - Z_min
        Z = np.maximum(Z, 0)
        Z = Z / (Z_max - Z_min)
        return Z
