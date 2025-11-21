import ikpy.chain
import ikpy.utils.plot as plot_utils

import numpy as np
import math

import matplotlib.pyplot as plt
import serial

my_chain = ikpy.chain.Chain.from_urdf_file("actual_arm_urdf.urdf", active_links_mask=[False, True, True, True, True, True])

target_position = [0, 0.2, 0.1]

target_orientation = [-1, 0, 0]

ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y")
print("The angles of each joints are: ", list(map(lambda r: math.degrees(r), ik.tolist())))
computed_position = my_chain.forward_kinematics(ik)
print("Computed position:", [f"{val:.2f}" for val in computed_position[:3, 3]])

fig, ax = plot_utils.init_3d_figure()
fig.set_figheight(18)  
fig.set_figwidth(26)  
my_chain.plot(ik, ax, target=target_position)
plt.xlim(-0.5, 0.5)
plt.ylim(-0.5, 0.5)
ax.set_zlim(0, 0.6)
plt.ion()



def doIK():
    global ik
    old_position = ik.copy()
    ik = my_chain.inverse_kinematics(target_position, target_orientation, orientation_mode="Y", initial_position=old_position)

def updatePlot():
    ax.cla()
    my_chain.plot(ik, ax, target=target_position)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.6)
    fig.canvas.draw_idle()
    
def move(x,y,z):
    global target_position
    target_position = [x,y,z]
    doIK()
    updatePlot()


move(0, 0.1, 0.2)