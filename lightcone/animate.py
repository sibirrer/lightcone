from lightcone.plot_3d import Plot3d
from tqdm import tqdm
import numpy as np
import os
from os.path import exists
import matplotlib.pyplot as plt
import imageio


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

    def ray_shooting(self, angle1, angle2):
        """

        :param angle1: float, rotation angle under which the ray-shooting animation is projected
        :param angle2: float, rotation angle under which the ray-shooting animation is projected
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
            plt.savefig(fname=filename, dpi=96)
            plt.gca()
            plt.close(fig)
            # if self._i_frame % 10 == 0:
            #     print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)
        print("Ray shooting complete!")

    def rotate_to_front(self, angle1, angle2, n_rotate=100):
        """
        rotate a static simulation with different transparency to be in direct projection to the fron

        :param angle1: float, angle of the start of the rotation
        :param angle2: float, angle of the start of the rotation
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
            plt.savefig(fname=filename, dpi=96)
            plt.gca()
            plt.close(fig)
            # if self._i_frame % 10 == 0:
            #     print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)
        print("Rotate to front complete!")

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
            print("WARNING!: MP4 File not detected. Make sure you have FFMPEG installed and in your PATH.")

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

    def gif(self):
        # build gif
        movie_name = self._folder_path + self._movie_name + ".gif"

        images = []
        for file_name in self._filename_list:
                images.append(imageio.imread(file_name))
        imageio.mimwrite(movie_name, images, format='.gif', fps=1)
