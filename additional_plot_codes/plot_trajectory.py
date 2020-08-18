"""
This module contains the code to plot the chaotic trajectory.
the initial neural ativity is 0.22
discrimination threshold = 0.96
"""
import numpy as np
import matplotlib.pyplot as plt
from trajectory_codes import skewtent_onestep


N = 500
initial_value = 0.22
threshold = 0.96
out = []
out.append(initial_value)
firingtime = np.arange(0, N, 1)
for iterations in range(1, N):
    out.append(skewtent_onestep(out[iterations-1], threshold))
    
plt.figure(figsize=(15, 15))
plt.plot(firingtime, out, '-r', markersize=12,)
plt.plot(firingtime, [threshold] * N, '--k', linewidth = 2.0, markersize=12, label='b = 0.96')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Trajectory', fontsize=30)
plt.legend(fontsize = 25)
plt.savefig("occd_trajectory.jpg", format='jpg', dpi=200)
plt.show()
    
    