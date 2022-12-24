import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
from matplotlib import cm
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

        self.dock_point = np.array([0, 60, 0])
        self.theta = 60.0
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

        """
        Plot target (cylinder) and LOS (cone)
        """
        x_center, y_center, radius, height = 0, 0, 60, 120
        Xc, Yc, Zc = self.data_for_cylinder_along_z(x_center, y_center, radius, height)
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

        X_los, Y_los, Z_los = self.data_for_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1])
        self.ax.plot_surface(X_los, Y_los, Z_los, alpha=0.5)
        plt.draw()

    def data_for_cylinder_along_z(self, center_x, center_y, radius, height_z):
        #height_z
        z = np.linspace((height_z//2)*-1, height_z, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid

    def data_for_cone_along_y(self, center_x, center_z, theta, height_y):
        #y = np.linspace(0, height_y, 50)
        #f = lambda z, x, theta : ((np.tanh(theta)**2.0) * (center_z**2.0)) + (center_x**2.0)

        def cone_formula(z, x, theta):
            ys = np.square(np.tanh(theta)) * np.square(z) + np.square(x)
            return ys

        #u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:80j]
        """
        x = np.cos(u)*np.sin(v)
        z = np.sin(u)*np.sin(v)

        x = x * 500.0
        z = z * 500.0
        """

        x = np.arange(-50.0, 50, 5)
        z = np.arange(-50.0, 50, 5)
        X, Z = np.meshgrid(x,z)
        Y = cone_formula(Z, X, theta)
        Y += height_y
        return X, Y, Z


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
        ani = animation.FuncAnimation(self.fig, partial(self.animate, file_name=file_name), repeat=False, cache_frame_data=True, interval=1000)
        plt.ion()
        plt.show(block=False)
        #sleep(2.0)




