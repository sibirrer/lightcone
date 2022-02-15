from lightcone.plot_3d import Plot3d
import numpy as np
import os
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

        """
        filename = self._folder_path + self._movie_name + str(self._i_frame) + '.png'
        return filename

    def ray_shooting(self, angle1, angle2):
        # ray-trace
        n_list = np.linspace(0, self._n_z_bins - 1, self._n_z_bins)
        for n in n_list[::-1]:
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
            if self._i_frame % 10 == 0:
                print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)

    def rotate_to_front(self, angle1, angle2, n_rotate=100):
        angle1_front = 0
        angle2_front = -180

        # rotate angle to angle2 = -180, angle1=0
        angle1_list = np.linspace(angle1, angle1_front, n_rotate)
        angle2_list = np.linspace(angle2, angle2_front, n_rotate)
        # alpha_lens_list = np.logspace(-4.5, 0, n_rotate)
        # alpha_lens_list = np.zeros(int(n_rotate*0.8))
        alpha_lens_list = np.append(np.zeros(int(n_rotate*0.3)), np.logspace(-2, 0, n_rotate - int(n_rotate*0.3)))
        for n in range(n_rotate):
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
            if self._i_frame % 10 == 0:
                print(self._i_frame)
            self._i_frame += 1
            self._filename_list.append(filename)

    def mp4(self, fps=20):

        def save():
            os_string = "ffmpeg -r " + str(fps) + " -i " + self._folder_path + self._movie_name + \
                        "%01d.png -vcodec mpeg4 -y " + self._folder_path + self._movie_name + ".mp4"
            os.system(os_string)

        save()

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

        # with imageio.get_writer(movie_name, mode='I') as writer:
        #     for filename in self._filename_list:
        #         image = imageio.imread(filename)
        #         writer.append_data(image, meta=dict({'fps': 0.5}))


