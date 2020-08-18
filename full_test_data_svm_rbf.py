"""
This module give the classification results for test data using SVM with RBF
kernel.
Email: harikrishnannb07@gmail.com
Dtd: 2 - August - 2020


Parameters
----------
classification_type : string
    DESCRIPTION - classification_type == "binary_class" loads binary classification artificial data.
    classification_type == "multi_class" loads multiclass artificial data
folder_name : string
    DESCRIPTION - the name of the folder to store results. For eg., if
    folder_name = "hnb", then this function will create two folder "hnb-svm"
    and "hnb-svm_rbf" to save the classification report.
target_names : array, 1D, string
    DESCRIPTION - if there are two classes, then target_names = ['class-0', class-1]
    Note- At the present version of the code, the results for binary classification
    and five class classification will be saved.
Returns : None
-------
Computes the accuracy_svm_rbf, fscore_svm_rbf

"""


import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from load_data_synthetic import get_data
#from Codes import classification_report_csv_


classification_type = "concentric_circle_noise"

folder_name = "full-testdata"
target_names = ['class-0', 'class-1']
path = os.getcwd()

result_path_svm_rbf = path + '/NEUROCHAOS-RESULTS/'  + classification_type + '/' + folder_name +'-svm_rbf/'

# Creating Folder to save the results
try:
    os.makedirs(result_path_svm_rbf)
except OSError:
    print("Creation of the result directory %s failed" % result_path_svm_rbf)
else:
    print("Successfully created the result directory %s" % result_path_svm_rbf)


full_artificial_data, full_artificial_label, full_artificial_test_data, full_artificial_test_label = get_data(classification_type)

num_classes = len(np.unique(full_artificial_label)) # Number of classes
print("**** Genome data details ******")

for class_label in range(np.max(full_artificial_label)+1):
    print("Total Data instance in Class -", class_label, " = ", full_artificial_label.tolist().count([class_label]))
    print(" train data = ", (full_artificial_data.shape[0]))
    print("val data  = ", (full_artificial_test_data.shape[0]))

# Start of svm_rbf classifier
svm_rbf_classifier = svm.SVC(kernel='rbf', gamma='scale')
svm_rbf_classifier.fit(full_artificial_data, full_artificial_label[:, 0])
predicted_svm_rbf_val_label = svm_rbf_classifier.predict(full_artificial_test_data)

acc_svm_rbf = accuracy_score(full_artificial_test_label, predicted_svm_rbf_val_label)*100
f1score_svm_rbf = f1_score(full_artificial_test_label, predicted_svm_rbf_val_label, average="macro")
report_svm_rbf = classification_report(full_artificial_test_label, predicted_svm_rbf_val_label, target_names=target_names)

# Saving the classification report to csv file for svm_rbf classifier.
print(report_svm_rbf)

#classification_report_csv_(report_svm_rbf, num_classes).to_csv(result_path_svm_rbf+'svm_rbf_report_'+ str(iterations) +'.csv', index=False)
confusion_matrix_svm_rbf = cm(full_artificial_test_label, predicted_svm_rbf_val_label)
print("Confusion matrixfor svm_rbf\n", confusion_matrix_svm_rbf)

# End of svm_rbf classifier.
# saving the f1-score
np.save(result_path_svm_rbf + 'f1score.npy', f1score_svm_rbf)
