import numpy as np
import matplotlib.pyplot as plt
from dynamics import chaser_discrete
import os
from visualizer import render_visual
from time import sleep


data_file_name = 'chaser16.txt'
vis_obj = render_visual()
vis_obj.render_animation(data_file_name)
