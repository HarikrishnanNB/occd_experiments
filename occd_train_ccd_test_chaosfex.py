"""
Training with OCCD and testing with CCD.
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
Returns : None
-------
Computes the accuracy_neurochaos, fscore_neurochaos

"""

import os
import numpy as np

from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX
from load_data_synthetic import get_data


classification_type_test = "concentric_circle"
classification_type_train = "concentric_circle_noise"


epsilon = 0.18
initial_neural_activity = 0.34
discrimination_threshold = 0.499


folder_name = "occd-train_ccd-test"
target_names = ['class-0', 'class-1']
path = os.getcwd()

result_path_neurochaos = path + '/NEUROCHAOS-NOISE-RESULTS/'  + classification_type_test + '/' + folder_name +'-neurochaos/'

# Creating Folder to save the results
try:
    os.makedirs(result_path_neurochaos)
except OSError:
    print("Creation of the result directory %s failed" % result_path_neurochaos)
else:
    print("Successfully created the result directory %s" % result_path_neurochaos)

## TEST DATA
ccd_train_data, ccd_train_label, ccd_test_data, ccd_test_label = get_data(classification_type_test)
## TRAIN DATA
occd_train_data, occd_train_label, occd_test_data, occd_test_label = get_data(classification_type_train)

num_classes = len(np.unique(ccd_test_label)) # Number of classes
print("**** Genome data details ******")

for class_label in range(np.max(ccd_train_label) + 1):
    print("Total Data instance in Class -", class_label, " = ", ccd_train_label.tolist().count([class_label]))
    print(" train data = ", (occd_train_data.shape[0]))
    print("test data  = ", (ccd_train_data.shape[0]))

    print("initial neural activity = ", initial_neural_activity, "discrimination threshold = ", discrimination_threshold, "epsilon = ", epsilon)

# Extracting Neurochaos features from the data
neurochaos_train_data_features = CFX.transform(occd_train_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)
neurochaos_val_data_features = CFX.transform(ccd_train_data, initial_neural_activity, 20000, epsilon, discrimination_threshold)

# Start of Neurochaos classifier
neurochaos_classifier = LinearSVC(random_state=0, tol=1e-5, dual=False)

neurochaos_classifier.fit(neurochaos_train_data_features, occd_train_label[:, 0])
predicted_neurochaos_val_label = neurochaos_classifier.predict(neurochaos_val_data_features)

acc_neurochaos = accuracy_score(ccd_train_label, predicted_neurochaos_val_label)*100
f1score_neurochaos = f1_score(ccd_train_label, predicted_neurochaos_val_label, average="macro")
report_neurochaos = classification_report(ccd_train_label, predicted_neurochaos_val_label, target_names=target_names)

# Saving the classification report to csv file for neurochaos classifier.
print(report_neurochaos)

#classification_report_csv_(report_neurochaos, num_classes).to_csv(result_path_neurochaos+'neurochaos_report_'+ str(iterations) +'.csv', index=False)


confusion_matrix_neurochaos = cm(ccd_train_label, predicted_neurochaos_val_label)
print("Confusion matrix for Neurochaos\n", confusion_matrix_neurochaos)


# End of ChaosFEX.
np.save(result_path_neurochaos + 'initial_neural_activity.npy', initial_neural_activity)
np.save(result_path_neurochaos + 'discrimination_threshold.npy', discrimination_threshold)
np.save(result_path_neurochaos + 'EPS.npy', epsilon)
np.save(result_path_neurochaos + 'f1score.npy', f1score_neurochaos)

