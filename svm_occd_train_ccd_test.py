'''
Performance of SVM with RBF kernel. Training with OCCD and testing with CCD.
We found the best hyperparameter using a validation data from ccd distribution.
C = 1.0 and gamma = 0.1 works for this case. 
'''

import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
from load_data_synthetic import get_data
#from Codes import classification_report_csv_


classification_type_test = "concentric_circle"
classification_type_train = "concentric_circle_noise"

folder_name = "svm_occd-train_ccd-test"
target_names = ['class-0', 'class-1']
path = os.getcwd()

result_path_svm_rbf = path + '/NEUROCHAOS-RESULTS/' + folder_name +'/'

# Creating Folder to save the results
try:
    os.makedirs(result_path_svm_rbf)
except OSError:
    print("Creation of the result directory %s failed" % result_path_svm_rbf)
else:
    print("Successfully created the result directory %s" % result_path_svm_rbf)


## TEST DATA
ccd_train_data, ccd_train_label, ccd_test_data, ccd_test_label = get_data(classification_type_test)
## TRAIN DATA
occd_train_data, occd_train_label, occd_test_data, occd_test_label = get_data(classification_type_train)

num_classes = len(np.unique(ccd_train_label)) # Number of classes
print("**** Sythetic data data details ******")

for class_label in range(np.max(ccd_train_label)+1):
    print("Total Data instance in Class -", class_label, " = ", ccd_train_label.tolist().count([class_label]))
    print(" OCCD train data = ", (occd_train_data.shape[0]))
    print("CCD validation data  = ", (ccd_train_data.shape[0]))


# Start of svm_rbf classifier

svm_rbf_classifier = svm.SVC(C = 1.0, kernel='rbf', gamma = 0.1)
svm_rbf_classifier.fit(occd_train_data, occd_train_label[:, 0])
predicted_svm_rbf_val_label = svm_rbf_classifier.predict(ccd_train_data)
acc_svm_rbf = accuracy_score(ccd_train_label, predicted_svm_rbf_val_label)*100
f1score_svm_rbf = f1_score(ccd_train_label, predicted_svm_rbf_val_label, average="macro")
report_svm_rbf = classification_report(ccd_train_label, predicted_svm_rbf_val_label, target_names=target_names)
  
print(report_svm_rbf)



confusion_matrix_svm_rbf = cm(ccd_train_label, predicted_svm_rbf_val_label)
print("Confusion matrixfor SVM RBF\n", confusion_matrix_svm_rbf)




