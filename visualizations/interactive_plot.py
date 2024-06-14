"""
Contains the functions to plot the interactive density plot
"""

import matplotlib.pyplot as plt
import numpy as np
# pylint: disable=E0401
from utils import data_utils, extract_data


# pylint: disable=R0903
class InteractivePlot:
    """
    Plots the interactive density plot
    """

    def __init__(self, pruned_neighbors_list):
        """
        Initializes with the required values
        """
        self.data = data_utils.get_data(extract_data.get_raw_data_path())
        self.pruned_neighbors_list = pruned_neighbors_list
        self.selected_index = None
        self.fig, self.axes = plt.subplots()
        self.sc_plot = self.axes.scatter(self.data[:, 0], self.data[:, 1], s=10, c='black')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.title('Click on a point to select it')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    def onclick(self, event):
        """
        Plots the interactive density plot
        """
        if event.inaxes != self.axes:
            return
        x_val, y_val = event.xdata, event.ydata
        selected_point = np.array([x_val, y_val])
        distances = np.linalg.norm(self.data - selected_point, axis=1)
        index = np.argmin(distances)

        if self.selected_index == index:
            self.selected_index = None
            self.axes.clear()
            self.axes.scatter(self.data[:, 0], self.data[:, 1], s=10, c='black')
        else:
            self.selected_index = index
            pruned_neighbors = [n for n in self.pruned_neighbors_list[index] if n != 0]
            self.axes.clear()
            self.axes.scatter(self.data[:, 0], self.data[:, 1], s=10, c='black', alpha=0.1)
            self.axes.scatter(self.data[pruned_neighbors][:, 0],
                              self.data[pruned_neighbors][:, 1], s=10,
                              c='blue')
            self.axes.scatter(self.data[index, 0], self.data[index, 1], s=15, c='red')

        self.axes.set_title('Click on a point to select it')
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.axes.grid(True)
        self.fig.canvas.draw()
