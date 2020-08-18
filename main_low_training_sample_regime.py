# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 22:01:52 2020

@author: harik
"""
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report
import ChaosFEX.feature_extractor as CFX
from quad_data_picking import get_data, class_wise_data, data_in_quadrants

# Loading the overlapping concentric circle data (not normalized)
classification_type = "concentric_circle_noise"
traindata, trainlabel, testdata, testlabel = get_data(classification_type)

# Normalizing the test data
X_test_norm = (testdata - np.min(testdata, 0))/(np.max(testdata, 0) - np.min(testdata, 0))


# Hyperparameters found using three fold validation
initial_neural_activity = 0.22
discrimination_threshold = 0.96
epsilon = 0.018

# Extracting Neurochaos features from normalized test data
neurochaos_test_data_features = CFX.transform(X_test_norm, initial_neural_activity, 20000, epsilon, discrimination_threshold)

# Binary Classification problem
target_names = ['class-0', 'class-1']
# Extracting traindata belonging to class-0 and class-1 separately
class_0, class_1 = class_wise_data(traindata, trainlabel)
# Extracting the data belonging to 4 quadrants from class-0
quad_1_c0, quad_2_c0, quad_3_c0, quad_4_c0 = data_in_quadrants(class_0)
# Extracting the data belonging to 4 quadrants from class-1
quad_1_c1, quad_2_c1, quad_3_c1, quad_4_c1 = data_in_quadrants(class_1)

# Finding the L2 norm (distance) of data instances belonging to quadrant 1,
# 2, 3 and 4 for the class-0 data.
radius_1_c0 = np.sqrt(np.sum(quad_1_c0 **2, axis=1))
radius_2_c0 = np.sqrt(np.sum(quad_2_c0 **2, axis=1))
radius_3_c0 = np.sqrt(np.sum(quad_3_c0 **2, axis=1))
radius_4_c0 = np.sqrt(np.sum(quad_4_c0 **2, axis=1))
# Finding the L2 norm (distance) of data instances belonging to quadrant 1,
# 2, 3 and 4 for the class-1 data.
radius_1_c1 = np.sqrt(np.sum(quad_1_c1 **2, axis=1))
radius_2_c1 = np.sqrt(np.sum(quad_2_c1 **2, axis=1))
radius_3_c1 = np.sqrt(np.sum(quad_3_c1 **2, axis=1))
radius_4_c1 = np.sqrt(np.sum(quad_4_c1 **2, axis=1))

# Sorting from decresing to increasing based on the length (L2-norm)
# of data instances belonging to each quadrant.
radius_1_c0_sort_index = np.argsort(radius_1_c0)
radius_2_c0_sort_index = np.argsort(radius_2_c0)
radius_3_c0_sort_index = np.argsort(radius_3_c0)
radius_4_c0_sort_index = np.argsort(radius_4_c0)

radius_1_c1_sort_index = np.argsort(radius_1_c1)
radius_2_c1_sort_index = np.argsort(radius_2_c1)
radius_3_c1_sort_index = np.argsort(radius_3_c1)
radius_4_c1_sort_index = np.argsort(radius_4_c1)


max_percentage = 30
# Class-0
# Count of the maximum length (L2 norm) data belonging to quadrant 1 of class-0.
max_count_1_c0 = int(max_percentage*len(radius_1_c0_sort_index)/100)

max_quad_1_c0 = quad_1_c0[radius_1_c0_sort_index[len(radius_1_c0_sort_index) - max_count_1_c0 :], :]
min_quad_1_c0 = quad_1_c0[radius_1_c0_sort_index[0: len(radius_1_c0_sort_index) - max_count_1_c0], :]

# Class-0
# Count of the maximum length (L2 norm) data belonging to quadrant 2 of class-0.
max_count_2_c0 = int(max_percentage*len(radius_2_c0_sort_index)/100)

max_quad_2_c0 = quad_2_c0[radius_2_c0_sort_index[len(radius_2_c0_sort_index) - max_count_2_c0 :], :]
min_quad_2_c0 = quad_2_c0[radius_2_c0_sort_index[0: len(radius_2_c0_sort_index) - max_count_2_c0], :]

# Class-0
# Count of the maximum length (L2 norm) data belonging to quadrant 3 of class-0.

max_count_3_c0 = int(max_percentage*len(radius_3_c0_sort_index)/100)

max_quad_3_c0 = quad_3_c0[radius_3_c0_sort_index[len(radius_3_c0_sort_index) - max_count_3_c0 :], :]
min_quad_3_c0 = quad_3_c0[radius_3_c0_sort_index[0: len(radius_3_c0_sort_index) - max_count_3_c0], :]

# Class-0
# Count of the maximum length (L2 norm) data belonging to quadrant 4 of class-0.
max_count_4_c0 = int(max_percentage*len(radius_4_c0_sort_index)/100)

max_quad_4_c0 = quad_4_c0[radius_4_c0_sort_index[len(radius_4_c0_sort_index) - max_count_4_c0 :], :]
min_quad_4_c0 = quad_4_c0[radius_4_c0_sort_index[0: len(radius_4_c0_sort_index) - max_count_4_c0], :]

# Class-1
# Count of the maximum length (L2 norm) data belonging to quadrant 1 of class-1.

max_count_1_c1 = int(max_percentage*len(radius_1_c1_sort_index)/100)

max_quad_1_c1 = quad_1_c1[radius_1_c1_sort_index[len(radius_1_c1_sort_index) - max_count_1_c1 :], :]
min_quad_1_c1 = quad_1_c1[radius_1_c1_sort_index[0: len(radius_1_c1_sort_index) - max_count_1_c1], :]

# Class-1
# Count of the maximum length (L2 norm) data belonging to quadrant 2 of class-1.

max_count_2_c1 = int(max_percentage*len(radius_2_c1_sort_index)/100)

max_quad_2_c1 = quad_2_c1[radius_2_c1_sort_index[len(radius_2_c1_sort_index) - max_count_2_c1 :], :]
min_quad_2_c1 = quad_2_c1[radius_2_c1_sort_index[0: len(radius_2_c1_sort_index) - max_count_2_c1], :]

# Class-1
# Count of the maximum length (L2 norm) data belonging to quadrant 3 of class-1.

max_count_3_c1 = int(max_percentage*len(radius_3_c1_sort_index)/100)

max_quad_3_c1 = quad_3_c1[radius_3_c1_sort_index[len(radius_3_c1_sort_index) - max_count_3_c1 :], :]
min_quad_3_c1 = quad_3_c1[radius_3_c1_sort_index[0: len(radius_3_c1_sort_index) - max_count_3_c1], :]

# Class-1
# Count of the maximum length (L2 norm) data belonging to quadrant 4 of class-1.

max_count_4_c1 = int(max_percentage*len(radius_4_c1_sort_index)/100)

max_quad_4_c1 = quad_4_c1[radius_4_c1_sort_index[len(radius_4_c1_sort_index) - max_count_4_c1 :], :]
min_quad_4_c1 = quad_4_c1[radius_4_c1_sort_index[0: len(radius_4_c1_sort_index) - max_count_4_c1], :]

# Next we will remove 30 % data belong to quadrant 1, 2, 3 and 4 for both class-0
# and class-1.


trials = 200 # No of random trials of training in the low training sample regime.
iterations = - 1
num_samples_per_quad = np.arange(1, 184, 4)# No. of samples extracted from each quadrant for class-0 and class-1
SAMPLES_PER_CLASS = num_samples_per_quad * 4
# Initialization of arrays to store results
std_neurochaos = np.zeros((len(num_samples_per_quad), 1))
std_svm = np.zeros((len(num_samples_per_quad), 1))
fscore_svm = np.zeros((len(num_samples_per_quad), 1))
fscore_neurochaos = np.zeros((len(num_samples_per_quad), 1))

for num_samples in num_samples_per_quad:
    # Number of training samples per each quadrant for class-0 and class-1
    iterations = iterations + 1

    f1_svm = []
    f1_neurochaos = []
    for num_trials in range(0, trials):
        # We do 200 random trials of training
        print("numner of samples per quadrant = ", num_samples, "trials= ", num_trials)

        # From each quadrant of class-0 we take -> num_samples. So total number
        # of samples in class-0 = 4*num_samples(since there are four quadrant).
        index_min = np.random.randint(min_quad_1_c0.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_1_c0.shape[0], size=(1, int(num_samples)))

        data_quad_1_c0 = min_quad_1_c0[index_min[0, :]]

        index_min = np.random.randint(min_quad_2_c0.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_2_c0.shape[0], size=(1, int(num_samples)))

        data_quad_2_c0 = min_quad_2_c0[index_min[0, :]]

        index_min = np.random.randint(min_quad_3_c0.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_3_c0.shape[0], size=(1, int(num_samples)))

        data_quad_3_c0 = min_quad_3_c0[index_min[0, :]]

        index_min = np.random.randint(min_quad_4_c0.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_4_c0.shape[0], size=(1, int(num_samples)))

        data_quad_4_c0 = min_quad_4_c0[index_min[0, :]]

        class_0_low_data = np.concatenate((data_quad_1_c0, data_quad_2_c0, data_quad_3_c0, data_quad_4_c0))
        class_0_low_data_label = np.zeros((class_0_low_data.shape[0], 1))
        #  class-1
        index_min = np.random.randint(min_quad_1_c1.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_1_c1.shape[0], size=(1, int(num_samples)))
        data_quad_1_c1 = min_quad_1_c1[index_min[0, :]]

        index_min = np.random.randint(min_quad_2_c1.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_2_c1.shape[0], size=(1, int(num_samples)))
        data_quad_2_c1 = min_quad_2_c1[index_min[0, :]]

        index_min = np.random.randint(min_quad_3_c1.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_3_c1.shape[0], size=(1, int(num_samples)))
        data_quad_3_c1 = min_quad_3_c1[index_min[0, :]]

        index_min = np.random.randint(min_quad_4_c1.shape[0], size=(1, int(num_samples)))
        #index_max = np.random.randint(max_quad_4_c1.shape[0], size=(1, int(num_samples)))
        data_quad_4_c1 = min_quad_4_c1[index_min[0, :]]

        class_1_low_data = np.concatenate((data_quad_1_c1, data_quad_2_c1, data_quad_3_c1, data_quad_4_c1))
        class_1_low_data_label = np.ones((class_1_low_data.shape[0], 1))
        # Plotting the train data
        '''
        plt.figure(figsize=(15, 15))
        plt.plot(class_0_low_data[:, 0], class_0_low_data[:, 1], '*k', markersize=12, label = 'Class-0')
        plt.plot(class_1_low_data[:, 0], class_1_low_data[:, 1], 'or', markersize=12, label = 'Class-1')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.grid(True)
        plt.xlim([-0.86, 0.86])
        plt.ylim([-0.86, 0.86])
        plt.xlabel(r' $f_1$', fontsize=30)
        plt.ylabel(r' $f_2$', fontsize=30)
        plt.legend(fontsize = 22)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.savefig(RESULT_PATH +"/data_"_+str(num_samples)+"_samples_"+str(num_trials)+"_.jpg", format='jpg', dpi=200)
        plt.show()
        '''
        final_traindata = np.concatenate((class_0_low_data, class_1_low_data))
        final_trainlabel = np.concatenate((class_0_low_data_label, class_1_low_data_label))

        X_train_norm = (final_traindata - np.min(final_traindata, 0))/(np.max(final_traindata, 0) - np.min(final_traindata, 0))


        neurochaos_train_data_features = CFX.transform(X_train_norm, initial_neural_activity, 20000, epsilon, discrimination_threshold)


        # Start of Neurochaos classifier
        neurochaos_classifier = LinearSVC(random_state=0, tol=1e-5, dual=False)

        neurochaos_classifier.fit(neurochaos_train_data_features, final_trainlabel[:, 0])
        predicted_neurochaos_test_label = neurochaos_classifier.predict(neurochaos_test_data_features)

        acc_neurochaos = accuracy_score(testlabel, predicted_neurochaos_test_label)*100
        f1score_neurochaos = f1_score(testlabel, predicted_neurochaos_test_label, average="macro")
        report_neurochaos = classification_report(testlabel, predicted_neurochaos_test_label, target_names=target_names)

        # Saving the classification report to csv file for neurochaos classifier.
        #print(report_neurochaos)
        f1_neurochaos.append(f1score_neurochaos)
        confusion_matrix_neurochaos = cm(testlabel, predicted_neurochaos_test_label)
        #print(confusion_matrix_neurochaos)


        # Low training sample using SVM with RBF kernel

        svm_classifier = svm.SVC(kernel='rbf', gamma='scale')
        svm_classifier.fit(X_train_norm, final_trainlabel[:, 0])
        predicted_svm_test_label = svm_classifier.predict(X_test_norm)

        acc_svm = accuracy_score(testlabel, predicted_svm_test_label)*100
        f1score_svm = f1_score(testlabel, predicted_svm_test_label, average="macro")
        report_svm = classification_report(testlabel, predicted_svm_test_label, target_names=target_names)
        f1_svm.append(f1score_svm)
        confusion_matrix_svm = cm(testlabel, predicted_svm_test_label)
        #print(confusion_matrix_svm)


    fscore_neurochaos[iterations, 0] = np.mean(f1_neurochaos)
    std_neurochaos[iterations, 0] = np.std(f1_neurochaos)
    fscore_svm[iterations, 0] = np.mean(f1_svm)
    std_svm[iterations, 0] = np.std(f1_svm)

# Saving the results
PATH = os.getcwd()
RESULT_PATH = PATH + '/NEUROCHAOS-LTS-RESULTS/'  + classification_type + '/OCCD-LTS/'


try:
    os.makedirs(RESULT_PATH)
except OSError:
    print("Creation of the result directory %s failed" % RESULT_PATH)
else:
    print("Successfully created the result directory %s" % RESULT_PATH)

np.save(RESULT_PATH+"/fscore_neurochaos.npy", fscore_neurochaos)
np.save(RESULT_PATH+"/std_fscore_neurochaos.npy", std_neurochaos)
np.save(RESULT_PATH+"/fscore_svm_rbf.npy", fscore_svm)
np.save(RESULT_PATH+"/std_fscore_svm_rbf.npy", std_svm)
np.save(RESULT_PATH+"/samples_per_class.npy", SAMPLES_PER_CLASS)
