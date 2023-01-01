import numpy as np
import matplotlib.pyplot as plt
from dynamics import chaser_discrete
import os
from visualizer import render_visual
from time import sleep
#os.chdir('runs')
path = os.getcwd()
data_dir = os.path.join(path, 'runs')
num = len(os.listdir(data_dir))

if num > 0:
    num -= 1

data_file_name = f'chaser{num}.txt'
vis_obj = render_visual()
vis_obj.render_animation(data_file_name)
