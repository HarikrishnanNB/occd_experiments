
import numpy as np
import matplotlib.pyplot as plt
from quad_data_picking import get_data, class_wise_data, data_in_quadrants

# Loading the overlapping concentric circle data (not normalized)
classification_type = "concentric_circle_noise"
traindata, trainlabel, testdata, testlabel = get_data(classification_type)

total_data = np.concatenate((traindata, testdata))
total_label = np.concatenate((trainlabel, testlabel))

class_0, class_1 = class_wise_data(total_data, total_label)

# Firing Rate
plt.figure(figsize=(15, 15))
plt.plot(class_0[:, 0],class_0[:, 1], '*k', markersize=12, label = 'Class-0')
plt.plot(class_1[:, 0],class_1[:, 1], 'or', markersize=12, label = 'Class-1')

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel(r' $f_1$', fontsize=30)
plt.ylabel(r' $f_2$', fontsize=30)
plt.legend(fontsize = 25)
plt.savefig("total_occd_data.jpg", format='jpg', dpi=200)
plt.show()