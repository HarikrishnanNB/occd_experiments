'''
Plot corresponding to hyperparameter tuning for OCCD.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
DATA_NAME = "concentric_circle_noise"
PATH = os.getcwd()
RESULT_PATH = PATH + '/HYPERPARAMETER-TUNING/'  + DATA_NAME + '/NEUROCHAOS-RESULTS/'

print("Loading Hyperparameter Tuning Results")
FSCORE = np.load(RESULT_PATH + 'h_fscore.npy')
EPSILON = np.load(RESULT_PATH + 'h_EPS.npy')

plt.figure(figsize=(10,10))
plt.plot(EPSILON[0, 0, :],FSCORE[0, 0, :],'-*k', markersize = 12)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel('$\epsilon$', fontsize=25)
plt.ylabel('Average F1-score', fontsize=25)
plt.ylim(0, 1.01)
plt.tight_layout()
plt.savefig(RESULT_PATH+"/concentric-circle-noise-hyperparameter-tuning.jpg", format='jpg', dpi=200)
plt.show()