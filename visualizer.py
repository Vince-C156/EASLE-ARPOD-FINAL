import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import os
from functools import partial
from time import sleep

class render_visual:

    def __init__(self, data_dir = 'runs'):
        style.use('fast')
        matplotlib.use('WebAgg')
        dir_path = os.getcwd()
        data_path = os.path.join(dir_path, data_dir)
        self.data_path = data_path

        if os.path.exists(data_path) == False:
            os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'w') as f:
            #f.write('0,0,0')
            pass
        """

        self.fig = plt.figure(figsize = (8,8))
        self.ax = plt.axes(projection='3d')
        self.ax.set_zlim([-1000, 1000]) 
        self.ax.set_xlim([-1000, 1000]) 
        self.ax.set_ylim([-1000, 1000])

    def animate(self, i, file_name):
        data = os.path.join(self.data_path, file_name)

        if os.path.exists(data) == False:
            #file_path = os.path.join(data_path, file_name)
            with open(data, 'w') as f:
                #f.write('0,0,0')
                pass


        with open(data) as f:
            lines = f.readlines()
        xs, ys, zs = np.array([]), np.array([]), np.array([])
        for line in lines:
            line = line.strip()
            if len(line) > 1:
                x, y, z = line.split(',')
                x, y, z = np.float64(x), np.float64(y), np.float64(z)
                xs = np.append(xs, [x])
                ys = np.append(ys, [y])
                zs = np.append(zs, [z])
        self.ax.plot3D(xs, ys, zs, color = 'blue', linewidth=1)
        self.ax.set_xlabel('x', labelpad=20)
        self.ax.set_ylabel('y', labelpad=20)
        self.ax.set_zlabel('z', labelpad=20)
        plt.draw()
        sleep(2.0)


    def render_animation(self, file_name):
        """
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection='3d')
        ax.set_zlim([-1000, 1000]) 
        ax.set_xlim([-1000, 1000]) 
        ax.set_ylim([-1000, 1000])
        """
        #plt.ion()
        #data = os.path.join(self.data_path, file_name)
        ani = animation.FuncAnimation(self.fig, partial(self.animate, file_name=file_name), repeat=False, cache_frame_data=True, interval=5000)
        plt.ion()
        plt.show(block=False)
        #sleep(2.0)




