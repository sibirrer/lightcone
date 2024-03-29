from lightcone.plot_3d import Plot3d
from tqdm import tqdm
import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
import imageio
import warnings


class Animate(Plot3d):
    """
    makes animated figure

    """
    def __init__(self, movie_name, folder_path, **kwargs_plot3d):
        """

        :param movie_name: string, name of movie (<movie_name>.mp4)
        :param folder_path: path to folder
        :param kwargs_plot3d: keyword arguments to initiate Plot3d class
        """
        super(Animate, self).__init__(**kwargs_plot3d)
        self._movie_name = movie_name
        self._folder_path = folder_path
        self._i_frame = 0  # frame number
        self._filename_list = []

    def _filename(self, i_frame):
        """

        :param i_frame: integer, frame number
        :return: full filename the frame should be saved
        :rtype: string
        """
        filename = self._folder_path + self._movie_name + str(i_frame) + '.png'
        return filename

    def ray_shooting(self, angle1, angle2, dpi=96):
        """

        :param angle1: float, rotation angle under which the ray-shooting animation is projected
        :param angle2: float, rotation angle under which the ray-shooting animation is projected
        :param dpi: float, dots (pixels) per inch
        """
        # ray-trace
        n_list = np.linspace(0, self._n_z_bins - 1, self._n_z_bins)
        for n in tqdm(n_list[::-1], desc="Loading ray shooting…", ascii=False, ncols=75):
            plt.ioff()
            fig = plt.figure()
            fig = self.plot3d(fig=fig, angle1=angle1, angle2=angle2, n_ray=int(n), plot_source=True, plot_lens=False,
                              plot_rays=True, alpha_lens=1)
            # fig = figure_sequence(fig=fig, n_ray=int(n), angle1=angle1, angle2=angle2, **kwargs_compute)

            # Save it & close the figure
            filename = self._filename(self._i_frame)
            plt.savefig(fname=filename, dpi=dpi)
            plt.gca()
            plt.close(fig)
            # if self._i_frame % 10 == 0:
            #     print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)
        print("Ray shooting complete!")

    def rotate_to_front(self, angle1, angle2, n_rotate=100, dpi=96):
        """
        rotate a static simulation with different transparency to be in direct projection to the front

        :param angle1: float, angle of the start of the rotation
        :param angle2: float, angle of the start of the rotation
        :param dpi: float, dots (pixels) per inch
        """
        angle1_front = 0
        angle2_front = -180

        # rotate angle to angle2 = -180, angle1=0
        angle1_list = np.linspace(angle1, angle1_front, n_rotate)
        angle2_list = np.linspace(angle2, angle2_front, n_rotate)
        # alpha_lens_list = np.logspace(-4.5, 0, n_rotate)
        # alpha_lens_list = np.zeros(int(n_rotate*0.8))
        alpha_lens_list = np.append(np.zeros(int(n_rotate*0.3)), np.logspace(-2, 0, n_rotate - int(n_rotate*0.3)))
        for n in tqdm(range(n_rotate), desc="Loading rotate to front…", ascii=False, ncols=75):
            plt.ioff()
            fig = plt.figure()
            fig = self.plot3d(fig=fig, angle1=angle1_list[n], angle2=angle2_list[n], alpha_lens=alpha_lens_list[n],
                              n_ray=int(0), plot_source=True, plot_lens=True, plot_rays=True,
                              alpha_source=1-alpha_lens_list[n])
            # Save it & close the figure
            filename = self._filename(self._i_frame)
            plt.savefig(fname=filename, dpi=dpi)
            plt.gca()
            plt.close(fig)
            # if self._i_frame % 10 == 0:
            #     print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)
        print("Rotate to front complete!")

    def transition_to_noised_img(self, config_handler, sample, dpi=96):
        """
        After rotate_to_front(), this method can be called to transition the lensed image into what the observer would
        expect to see from the CCD with expected noise.

        :param config_handler: ConfigHandler object used for Paltas configuration file. Need to import ConfigHandler
         from paltas.Configs.config_handler and initialize the ConfigHandler
        :param sample: dict of config_handler.get_current_sample()
        :param dpi: float, dots (pixels) per inch
        """

        # plot3d lens
        Zl = self.ray_colors(flux=self._image)
        # plot simulated observable lens with expected noise
        img = self.sim_source_with_noise(config_handler=config_handler, sample=sample)

        # Converting the plot3d lens to a 100x100 image
        # as well as the simulated image with noise to keep both datatypes consistent (both become PIL images)
        Zl_im = self.convert_arr_to_RGB_Im(Zl)
        sim_im = self.convert_arr_to_RGB_Im(img)

        # alpha_lens_list1 = np.linspace(0, 1, 20)
        # alpha_lens_list2 = np.zeros(alpha_lens_list1.shape[0] * 2)
        # alpha_lens_list2[-20:] = alpha_lens_list1
        alpha_lens_list = np.ones(100)
        alpha_lens_list[:80] = np.linspace(0, 1, 80)
        for n in tqdm(range(alpha_lens_list.size), desc="transitioning…", ascii=False, ncols=75):
            plt.ioff()
            fig = plt.figure()
            # if n < alpha_lens_list1.size:
            #    fig = self.match_2d_plot_to_3d_plot(fig=fig, plot_2d=Zl_im, alpha=alpha_lens_list1[n])
            # else:
            #    fig = self.match_2d_plot_to_3d_plot(fig=fig, plot_2d=Zl_im, alpha=1)
            fig = self.match_2d_plot_to_3d_plot(fig=fig, plot_2d=Zl_im, alpha=1)
            fig = self.match_2d_plot_to_3d_plot(fig=fig, plot_2d=sim_im, alpha=alpha_lens_list[n])
            # Save it & close the figure
            filename = self._filename(self._i_frame)
            plt.savefig(fname=filename, dpi=dpi)
            plt.gca()
            plt.close(fig)

            self._i_frame += 1
            self._filename_list.append(filename)

        print("Transition to noised image complete!")

    def mp4(self, fps=20):
        """
        runs ffmpeg

        :param fps: integer, frames per second
        :return: saved mp4 file of the animation of all saved files in order
        """

        def save():
            os_string = "ffmpeg -r " + str(fps) + " -i " + self._folder_path + self._movie_name + \
                        "%01d.png -vcodec mpeg4 -y " + self._folder_path + self._movie_name + ".mp4"
            os.system(os_string)

        save()
        movie_file = self.movie_name()
        file_exists = exists(movie_file)
        if not file_exists:
            warnings.warn("WARNING!: MP4 File not detected. Make sure you have FFMPEG installed and in your PATH.")

    def finish(self):
        # Remove files
        for filename in set(self._filename_list):
            os.remove(filename)
        self._filename_list = []

    def movie_name(self):
        """
        full movie name with path
        """
        return self._folder_path + self._movie_name + ".mp4"

    def gif(self, fps=1):
        """
        Creates a gif

        :param fps: integer, frames per second
        """
        # build gif
        movie_name = self._folder_path + self._movie_name + ".gif"

        images = []
        for file_name in self._filename_list:
            images.append(imageio.imread(file_name))
        imageio.mimwrite(movie_name, images, format='.gif', fps=fps)
