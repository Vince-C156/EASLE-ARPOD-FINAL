import numpy as np
import matplotlib.pyplot as plt
from dynamics import chaser_discrete
import os
from visualizer import render_visual
from time import sleep

def write2text(chaser, data_dir, file_name, step):
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, data_dir)
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        """
        with open(file_path, 'w') as f:
            #f.write('0,0,0\n')
            pass

    state_arr = np.asarray(chaser.get_state_trace())
    print(f'state arr shape {state_arr.shape}')
    pos_arr = state_arr[step:, 0:3]

    lines2write = ['\n'+str(pos[0])+','+str(pos[1])+','+str(pos[2]) for pos in pos_arr]
    print(f'WRITING TO {file_path}')
    print('----------------------')
    print('DATA WRITTING')
    print(lines2write)
    file = open(file_path, 'a')
    file.writelines(lines2write)
    file.close()


chaser = chaser_discrete()
action = np.array([0.0, 0.0, 0.0], dtype=np.float64)

data_file_name = 'chaser16.txt'
"""
vis_obj = render_visual()
vis_obj.render_animation(data_file_name)
"""
for t in range(1000):
    x_next = chaser.get_next(action)
    chaser.update_state(x_next)
    write2text(chaser, 'runs', data_file_name, t)
    sleep(0.5)
    #print(chaser.get_state_trace())
"""
#chaser.reset()
print(chaser.get_state_trace())
write2text(chaser, 'runs')
vis_obj = render_visual()

vis_obj.render_animation('chaser6.txt')
#plt.show()
"""
