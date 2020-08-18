


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX
from load_data_synthetic import get_data
#from Codes import classification_report_csv_



"""
This module does the five_fold_crossvalidation and saves the classifcation
report. At present the results for binary classification and five class classification is saved.
Author: Harikrishnan N B
Email: harikrishnannb07@gmail.com
Dtd: 2 - August - 2020

Parameters
----------
classification_type : string
    DESCRIPTION - classification_type == "binary_class" loads binary classification artificial data.
    classification_type == "multi_class" loads multiclass artificial data
epsilon : scalar, float
    DESCRIPTION - A value in the range 0 and 0.3. for eg. epsilon = 0.1835
initial_neural_activity : scalar, float
    DESCRIPTION - The chaotic neurons has an initial neural activity.
    Initial neural activity is a value in the range 0 and 1.
discrimination_threshold : scalar, float
    DESCRIPTION - The chaotic neurons has a discrimination threhold.
    discrimination threshold is a value in the range 0 and 1.
folder_name : string
    DESCRIPTION - the name of the folder to store results. For eg., if
    folder_name = "hnb", then this function will create two folder "hnb-svm"
    and "hnb-neurochaos" to save the classification report.
target_names : array, 1D, string
    DESCRIPTION - if there are two classes, then target_names = ['class-0', class-1]
    Note- At the present version of the code, the results for binary classification
    and five class classification will be saved.
Returns
-------
mean_fold_accuracy_neurochaos, mean_fold_fscore_neurochaos, mean_fold_accuracy_svm, mean_fold_fscore_svm

The above are the average accuracy and f1 score for the five fold validation for neurochaos and svm respectively.
"""

classification_type = "concentric_circle_noise"

epsilon = 0.018 #0.018
initial_neural_activity = 0.22
discrimination_threshold = 0.96

folder_name= "full-testdata-result"
target_names = ['class-0', 'class-1']
path = os.getcwd()

result_path_neurochaos = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/' + folder_name +'-neurochaos/'

# Creating Folder to save the results
try:
    os.makedirs(result_path_neurochaos)
except OSError:
    print("Creation of the result directory %s failed" % result_path_neurochaos)
else:
    print("Successfully created the result directory %s" % result_path_neurochaos)


full_artificial_data, full_artificial_label, full_artificial_test_data, full_artificial_test_label = get_data(classification_type)



num_classes = len(np.unique(full_artificial_label)) # Number of classes
print("**** Genome data details ******")

for class_label in range(np.max(full_artificial_label)+1):
    print("Total Data instance in Class -", class_label, " = ", full_artificial_label.tolist().count([class_label]))



    print(" train data = ", (full_artificial_data.shape[0]))
    print("val data  = ", (full_artificial_test_data.shape[0]))

    print("initial neural activity = ", initial_neural_activity, "discrimination threshold = ", discrimination_threshold, "epsilon = ", epsilon)

# Extracting Neurochaos features from the data
neurochaos_train_data_features = CFX.transform(full_artificial_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)
neurochaos_val_data_features = CFX.transform(full_artificial_test_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)

# Start of Neurochaos classifier
neurochaos_classifier = LinearSVC(random_state=0, tol=1e-5, dual=False)

neurochaos_classifier.fit(neurochaos_train_data_features[:, 4:6], full_artificial_label[:, 0])
predicted_neurochaos_val_label = neurochaos_classifier.predict(neurochaos_val_data_features[:, 4:6])

acc_neurochaos = accuracy_score(full_artificial_test_label, predicted_neurochaos_val_label)*100
f1score_neurochaos = f1_score(full_artificial_test_label, predicted_neurochaos_val_label, average="macro")
report_neurochaos = classification_report(full_artificial_test_label, predicted_neurochaos_val_label, target_names=target_names)


print(report_neurochaos)

#classification_report_csv_(report_neurochaos, num_classes).to_csv(result_path_neurochaos+'neurochaos_report_'+ str(iterations) +'.csv', index=False)
   

confusion_matrix_neurochaos = cm(full_artificial_test_label, predicted_neurochaos_val_label)
print("Confusion matrixfor Neurochaos\n", confusion_matrix_neurochaos)

# ploting traindata
class_0 = full_artificial_label == 0
class_1 = full_artificial_label == 1
class_0_data = full_artificial_data[class_0[:,0],:]
class_1_data = full_artificial_data[class_1[:,0],:]
class_0_neurochaos_data = neurochaos_train_data_features[class_0[:,0],:]
class_1_neurochaos_data = neurochaos_train_data_features[class_1[:,0],:]

# Original data
plt.figure(figsize=(15, 15))
plt.plot(class_0_data[:, 0],class_0_data[:, 1], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1_data[:, 0],class_1_data[:, 1], 'or', markersize=12, label = 'Class-1')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r'$f_1$', fontsize=30)
plt.ylabel(r'$f_2$', fontsize=30)
plt.legend(fontsize = 25)
#plt.savefig(result_path_neurochaos +"/train-data-occd.jpg", format='jpg', dpi=200)
plt.show()


# Firing Rate
plt.figure(figsize=(15, 15))
plt.plot(class_0_neurochaos_data[:, 0],class_0_neurochaos_data[:, 1], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1_neurochaos_data[:, 0],class_1_neurochaos_data[:, 1], 'or', markersize=12, label = 'Class-1')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r'Firing rate of $f_1$', fontsize=30)
plt.ylabel(r'Firinig rate of $f_2$', fontsize=30)
plt.legend(fontsize = 25)
#plt.savefig(result_path_neurochaos +"/firing-rate-occd.jpg", format='jpg', dpi=200)
plt.show()


# Energy
plt.figure(figsize=(15, 15))
plt.plot(class_0_neurochaos_data[:, 2],class_0_neurochaos_data[:, 3], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1_neurochaos_data[:, 2],class_1_neurochaos_data[:, 3], 'or', markersize=12, label = 'Class-1')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r'Energy of $f_1$', fontsize=30)
plt.ylabel(r'Energy of $f_2$', fontsize=30)
plt.legend(fontsize = 25)
#plt.savefig(result_path_neurochaos +"/energy-occd.jpg", format='jpg', dpi=200)
plt.show()


# Firing Time
plt.figure(figsize=(15, 15))
plt.plot(class_0_neurochaos_data[:, 4],class_0_neurochaos_data[:, 5], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1_neurochaos_data[:, 4],class_1_neurochaos_data[:, 5], 'or', markersize=12, label = 'Class-1')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r'Firing time of $f_1$', fontsize=30)
plt.ylabel(r'Firinig time of $f_2$', fontsize=30)
plt.legend(fontsize = 25)
#plt.savefig(result_path_neurochaos +"/firing-time-occd.jpg", format='jpg', dpi=200)
plt.show()


# Entropy
plt.figure(figsize=(15, 15))
plt.plot(class_0_neurochaos_data[:, 6],class_0_neurochaos_data[:, 7], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1_neurochaos_data[:, 6],class_1_neurochaos_data[:, 7], 'or', markersize=12, label = 'Class-1')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r'Entropy of $f_1$', fontsize=30)
plt.ylabel(r'Entropy of $f_2$', fontsize=30)
plt.legend(fontsize = 25)
#plt.savefig(result_path_neurochaos +"/entropy-occd.jpg", format='jpg', dpi=200)
plt.show()

