#from scenario import Scenario
import numpy as np
import matplotlib.pyplot as plt

class Home:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.bulbs = np.zeros((self.height, self.width))
        self.luminosity = np.zeros((self.height, self.width,))
        self.presence = np.zeros((self.height, self.width,))

    def create_plot(self, location, title, data, plot_cmap='gray_r', plot_interpolation='nearest', plot_vmin=0, plot_vmax=1, plot_animated=False):

        plot = self.fig.add_subplot(location)
        plot.set_title(title, y=1.08)
        plot.xaxis.tick_top()
        plt.gca().invert_yaxis()
        plot.set_xticks(np.arange(self.width)+0.5)
        plot.set_yticks(np.arange(self.height)+0.5)
        plot.set_yticklabels([])
        plot.set_xticklabels([])
        image = plt.imshow(data, cmap=plot_cmap, interpolation=plot_interpolation, vmin=plot_vmin, vmax=plot_vmax, animated=plot_animated)
        plt.grid()
        return image

    def init_figures(self):

        self.fig = plt.figure(figsize=(1, 4))
        self.create_plot(location=141, title='Presence', data=self.presence, plot_cmap='gray_r', plot_interpolation='nearest', plot_vmin=0, plot_vmax=1)
        self.create_plot(location=142, title='Faulty Bulbs', data=self.bulbs, plot_cmap='Reds_r', plot_interpolation='nearest', plot_vmin=-1, plot_vmax=0)
        self.im = self.create_plot(location=143, title='Turned ON Bulbs', data=self.bulbs, plot_cmap='Blues', plot_interpolation='nearest', plot_vmin=0, plot_vmax=1, plot_animated=True)
        self.luminosity_im = self.create_plot(location=144, title='Luminosity', data=self.bulbs, plot_cmap='inferno', plot_interpolation='bilinear', plot_vmin=0, plot_vmax=1, plot_animated=True)

    def control(self):
        pass
