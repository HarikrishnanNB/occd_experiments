import os
import numpy as np
import matplotlib.pyplot as plt


classification_type = "concentric_circle_noise"
PATH = os.getcwd()
RESULT_PATH = PATH + '/NEUROCHAOS-LTS-RESULTS/'  + classification_type + '/OCCD-LTS/'


FSCORE_NEUROCHAOS = np.load(RESULT_PATH+"/fscore_neurochaos.npy" )    
STD_FSCORE_NEUROCHAOS = np.load(RESULT_PATH+"/std_fscore_neurochaos.npy") 
 
FSCORE_SVM = np.load(RESULT_PATH+"/fscore_svm_rbf.npy" )    
STD_FSCORE_SVM = np.load(RESULT_PATH+"/std_fscore_svm_rbf.npy")   
SAMPLES_PER_CLASS = np.load(RESULT_PATH+"/samples_per_class.npy")


# Plotting F1-score vs. Samples per class
plt.figure(figsize=(45,10))

plt.plot(SAMPLES_PER_CLASS ,FSCORE_NEUROCHAOS[:,0], linewidth = 2.0,  linestyle='solid', color ='r', marker='s',ms=10, label="Neurochaos-SVM")
plt.plot(SAMPLES_PER_CLASS ,FSCORE_SVM[:,0], linewidth = 2.0, linestyle='--', color ='k', marker='o', ms=10, label="SVM (linear)")

plt.xticks(SAMPLES_PER_CLASS,fontsize=22)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel('Number of training samples per class', fontsize=30)
plt.ylabel('Average F1-score', fontsize=30)
plt.legend(loc="lower right", fontsize=22)
plt.savefig(RESULT_PATH+ "/occd_F1_low_training_sample_regime_binary_classification.jpg", format='jpg', dpi=200)
plt.show()

# Plotting Standard deviation of F1-score vs. smaples per class
plt.figure(figsize=(45,10))
plt.plot(SAMPLES_PER_CLASS ,STD_FSCORE_NEUROCHAOS[:,0], linewidth = 2.0,  linestyle='solid', color ='r', marker='s',ms=10, label="Neurochaos-SVM")
plt.plot(SAMPLES_PER_CLASS ,STD_FSCORE_SVM[:,0], linewidth = 2.0, linestyle='--', color ='k', marker='o', ms=10, label="SVM (linear)")


plt.xticks(SAMPLES_PER_CLASS,fontsize=22)
plt.yticks(fontsize=25)
plt.grid(True)
plt.xlabel('Number of training samples per class', fontsize=30)
plt.ylabel('Standard deviation of F1-scores', fontsize=30)
plt.legend(loc="upper right", fontsize=22)
plt.savefig(RESULT_PATH+ "/occd_standard_deviation_low_training_sample_regime_binary_classification.jpg", format='jpg', dpi=200)
plt.show()    

    