# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:52:33 2020

@author: harik
"""
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from load_data_synthetic import get_data





classification_type = "concentric_circle_noise"



folder_name = "SVM-RBF"
target_names = ['class-0', 'class-1']
path = os.getcwd()

result_path_svm_rbf = path + '/HYPERPARAMETER-RESULTS/'  + classification_type + '/' + folder_name +'-svm_rbf/'

# Creating Folder to save the results
try:
    os.makedirs(result_path_svm_rbf)
except OSError:
    print("Creation of the result directory %s failed" % result_path_svm_rbf)
else:
    print("Successfully created the result directory %s" % result_path_svm_rbf)


## TEST DATA
occd_train_data, occd_train_label, occd_test_data, occd_test_label = get_data(classification_type)

num_classes = len(np.unique(occd_train_label)) # Number of classes
print("**** Genome data details ******")

for class_label in range(np.max(occd_train_label)+1):
    print("Total Data instance in Class -", class_label, " = ", occd_train_label.tolist().count([class_label]))
    print(" train data = ", (occd_train_data.shape[0]))
    print("test data  = ", (occd_test_data.shape[0]))



kernel1 = ['linear', 'rbf']
gamma1 = ['scale', 'auto']



accuracy_matrix = np.zeros((len(kernel1), len(gamma1)))
f1score_matrix = np.zeros((len(kernel1), len(gamma1)))

   
k_fold = KFold(n_splits=5, random_state=42, shuffle=True)

# returns the number of splitting iterations in the cross-validator
k_fold.get_n_splits(occd_train_data)
print(k_fold)

KFold(n_splits=5, random_state=42, shuffle=True)
row = -1
col = -1


for k1 in kernel1:
    row = row+1
    col = -1

    for g1 in gamma1:
        col = col+1
        acc_temp = []
        fscore_temp = []

        for train_index, val_index in k_fold.split(occd_train_data):
            print(train_index[0:10])
            print(val_index[0:10])
            train_occd_data = occd_train_data[train_index]
            val_occd_data = occd_train_data[val_index]
            train_occd_label = occd_train_label[train_index]
            val_occd_label = occd_train_label[val_index]
            print(" train data (%) = ",
                  (train_occd_data.shape[0]/occd_train_data.shape[0])*100)
            print("val data (%) = ",
                  (val_occd_data.shape[0]/occd_train_data.shape[0])*100)

            if k1 == 'linear': 
            # Neurochaos-SVM with linear kernel.

                classifier_svm_linear = LinearSVC(random_state=0, tol=1e-5, dual=False)
                classifier_svm_linear.fit(train_occd_data,
                                              train_occd_label[:, 0])
                predicted_val_label = classifier_svm_linear.predict(val_occd_data)
            else: 
                classifier_svm_rbf = svm.SVC(C=1.0, kernel=k1, gamma=g1)
                classifier_svm_rbf.fit(train_occd_data, train_occd_label[:, 0])
                predicted_val_label = classifier_svm_rbf.predict(val_occd_data)
            # Accuracy
            acc_svm = accuracy_score(val_occd_label, predicted_val_label)*100
            # Macro F1- Score
            f1score_svm = f1_score(val_occd_label, predicted_val_label, average="macro")

            acc_temp.append(acc_svm)
            fscore_temp.append(f1score_svm)

        
        # Average Accuracy
        accuracy_matrix[row, col] = np.mean(acc_temp)
        # Average Macro F1-score
        f1score_matrix[row, col] = np.mean(fscore_temp)


        print("Three fold Average F-SCORE %.3f" %f1score_matrix[row, col])

        print('--------------------------')
# Creating a result path to save the results.
path = os.getcwd()
result_path = path + '/SVM_RBF-HYPERPARAMETER/'  + classification_type + '/CROSS_VALIDATION/'


try:
    os.makedirs(result_path)
except OSError:
    print("Creation of the result directory %s failed" % result_path)
else:
    print("Successfully created the result directory %s" % result_path)

print("Saving Hyperparameter Tuning Results")
np.save(result_path + 'H_ACCURACY.npy', accuracy_matrix)
np.save(result_path + 'H_FSCORE.npy', f1score_matrix)

# =============================================================================
# best hyperparameters
# =============================================================================
# Computing the maximum F1-score obtained during crossvalidation.
maximum_fscore = np.max(f1score_matrix)

