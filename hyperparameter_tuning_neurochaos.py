# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:54:39 2020

@author: harik
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
import os
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.svm import LinearSVC
from load_data_synthetic import get_data
import ChaosFEX.feature_extractor as CFX

DATA_NAME = "concentric_circle_noise"
TRAINDATA, TRAINLABEL, X_TEST, Y_TEST = get_data(DATA_NAME)

INITIAL_NEURAL_ACTIVITY = [0.22]
DISCRIMINATION_THRESHOLD = [0.96]
EPSILON = np.arange(0.01, 0.201,0.001)


ACCURACY = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
Q = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
B = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
EPS = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))

# In[6]:


KF = KFold(n_splits= 3, random_state=42, shuffle=True) # Define the split - into 2 folds 
KF.get_n_splits(TRAINDATA) # returns the number of splitting iterations in the cross-validator
print(KF) 
KFold(n_splits= 3, random_state=42, shuffle=True)
ROW = -1
COL = -1
WIDTH = -1
for DT in DISCRIMINATION_THRESHOLD:
    ROW = ROW+1
    COL = -1
    WIDTH = -1
    for INA in INITIAL_NEURAL_ACTIVITY:
        COL =COL+1
        WIDTH = -1
        for EPSILON_1 in EPSILON:
            WIDTH = WIDTH + 1
            
            ACC_TEMP =[]
            FSCORE_TEMP=[]
        
            for TRAIN_INDEX, VAL_INDEX in KF.split(TRAINDATA):
                
                X_TRAIN, X_VAL = TRAINDATA[TRAIN_INDEX], TRAINDATA[VAL_INDEX]
                Y_TRAIN, Y_VAL = TRAINLABEL[TRAIN_INDEX], TRAINLABEL[VAL_INDEX]
    
                
                
    
             
                
                # Extract features
                FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)            
            
                CLASSIFIER = LinearSVC(random_state=0, tol=1e-5, dual = False)
                
                CLASSIFIER.fit(FEATURE_MATRIX_TRAIN, Y_TRAIN[:,0])
                Y_PRED = CLASSIFIER.predict(FEATURE_MATRIX_VAL)
    
                ACC = accuracy_score(Y_VAL, Y_PRED)*100
                RECALL = recall_score(Y_VAL, Y_PRED , average="macro")
                PRECISION = precision_score(Y_VAL, Y_PRED , average="macro")
                F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                '''
                print("ACCURACY")
                print("%.3f" %ACC)
                print("PRECISION")
                print("%.3f" %PRECISION)
                print("RECALL")
                print("%.3f" %RECALL)
                print("f1score")
                print("%.3f" %F1SCORE)
                
                CONFUSION_MAT = cm(Y_VAL,Y_PRED)  
                print("Confusion matrix\n", CONFUSION_MAT)
                '''               
                
                ACC_TEMP.append(ACC)
                FSCORE_TEMP.append(F1SCORE)
            Q[ROW, COL, WIDTH ] = INA # Initial Neural Activity
            B[ROW, COL, WIDTH ] = DT # Discrimination Threshold
            EPS[ROW, COL, WIDTH ] = EPSILON_1 
            ACCURACY[ROW, COL, WIDTH ] = np.mean(ACC_TEMP)
            FSCORE[ROW, COL, WIDTH ] = np.mean(FSCORE_TEMP)
            print("Mean F1-Score for Q = ", Q[ROW, COL, WIDTH ],"B = ", B[ROW, COL, WIDTH ],"EPSILON = ", EPS[ROW, COL, WIDTH ]," is  = ",  np.mean(FSCORE_TEMP)  )

print("Saving Hyperparameter Tuning Results")

PATH = os.getcwd()
RESULT_PATH = PATH + '/HYPERPARAMETER-TUNING/'  + DATA_NAME + '/NEUROCHAOS-RESULTS/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/h_fscore.npy", FSCORE )    
np.save(RESULT_PATH+"/h_accuracy.npy", ACCURACY ) 
np.save(RESULT_PATH+"/h_Q.npy", Q ) 
np.save(RESULT_PATH+"/h_Q.npy", B )
np.save(RESULT_PATH+"/h_EPS.npy", EPS )               


MAX_FSCORE = np.max(FSCORE)
Q_MAX = []
B_MAX = []
EPSILON_MAX = []

for ROW in range(0, len(DISCRIMINATION_THRESHOLD)):
    for COL in range(0, len(INITIAL_NEURAL_ACTIVITY)):
        for WID in range(0, len(EPSILON)):
            if FSCORE[ROW, COL, WID] == MAX_FSCORE:
                Q_MAX.append(Q[ROW, COL, WID])
                B_MAX.append(B[ROW, COL, WID])
                EPSILON_MAX.append(EPS[ROW, COL, WID])

print("BEST F1SCORE", MAX_FSCORE)
print("BEST INITIAL NEURAL ACTIVITY = ", Q_MAX)
print("BEST DISCRIMINATION THRESHOLD = ", B_MAX)
print("BEST EPSILON = ", EPSILON_MAX)


plt.figure(figsize=(10,10))
plt.plot(EPSILON,FSCORE[0,0,:],'-*k', markersize = 10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.xlabel('$\epsilon$', fontsize=20)
plt.ylabel('Average F1-score', fontsize=20)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULT_PATH+"/concentric-circle-noise-hyperparameter-tuning.jpg", format='jpg', dpi=200)
plt.show()

