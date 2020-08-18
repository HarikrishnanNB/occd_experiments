"""
This module contains the necessary function used in 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging




def get_data(DATA_NAME):
    """
    Parameters
    ----------
    DATA_NAME : TYPE : string
        DESCRIPTION.
        DATA_NAME can take any of two input as follows.
        DATA_NAME == "concentric_circle" -- will load the data set corresponding to concentric circle data (CCD)
        DATA_NAME == "cocentric_circle_noise" -- will load the data set corresponding to overlapping concentic circle data (OCCD).

    Returns
    -------
    X_TRAIN : array, 2D
        DESCRIPTION: traindata (not normalized)
    Y_TRAIN : array, 2D
        DESCRIPTION: trainlabel (0 and 1)
    X_TEST : array, 2D
        DESCRIPTION: testdata (not normalized)
    Y_TEST : array, 2D
        DESCRIPTION: testlabel (0 and 1)

    """
    if DATA_NAME == "concentric_circle":
        folder_path = "Data/" + DATA_NAME + "/"


        X_TRAIN = np.array( pd.read_csv(folder_path+"X_train.csv", header=None))
        Y_TRAIN =  np.array( pd.read_csv(folder_path+"Y_train.csv", header=None))
        X_TEST = np.array( pd.read_csv(folder_path+"X_test.csv", header=None))
        Y_TEST = np.array( pd.read_csv(folder_path+"Y_test.csv", header=None))

        return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


    elif DATA_NAME == "concentric_circle_noise":
        folder_path = "data/" + DATA_NAME + "/"


        X_TRAIN = np.array( pd.read_csv(folder_path+"X_TRAIN.csv", header=None))
        Y_TRAIN = np.array( pd.read_csv(folder_path+"Y_TRAIN.csv", header=None))
        X_TEST = np.array( pd.read_csv(folder_path+"X_TEST.csv", header=None))
        Y_TEST = np.array( pd.read_csv(folder_path+"Y_TEST.csv", header=None))

        return X_TRAIN, Y_TRAIN, X_TEST, Y_TEST


def class_wise_data(TRAINDATA, TRAINLABEL):
    """
    

    Parameters
    ----------
    TRAINDATA : TYPE -  array, 2D
        DESCRIPTION - traindata
    TRAINLABEL :TYPE -  array, 2D
        DESCRIPTION - trainlabel

    Returns
    -------
    CLASS_0 : TYPE - array, 2D
        DESCRIPTION - Data belonging to class-0
    CLASS_1 : TYPE - array, 2D
        DESCRIPTION - Data belonging to class-1

    """
    NUM_Z = TRAINLABEL.tolist().count([0])
    NUM_O = TRAINLABEL.tolist().count([1])
    CLASS_0 = np.zeros((NUM_Z, TRAINDATA.shape[1]))
    CLASS_1 = np.zeros((NUM_O, TRAINDATA.shape[1]))

    i =-1
    j = 0
    p = 0
    for lab in TRAINLABEL:
        i = i+1
        if lab == [0]:
            CLASS_0[j,:] = TRAINDATA[i,:]
            j = j+1
        elif lab == [1]:
            CLASS_1[p,:] = TRAINDATA[i,:]
            p = p+1
    return CLASS_0, CLASS_1


def data_in_quadrants(DATA):
    """

    Parameters
    ----------
    DATA : array, 2D
        DESCRIPTION: Input data

    Returns
    -------
    QUAD1 : array,2D
        DESCRIPTION : Data lieing in quadrant 1.
    QUAD2 : array,2D
        DESCRIPTION : Data lieing in quadrant 2.
    QUAD3 : array,2D
        DESCRIPTION : Data lieing in quadrant 3.
    QUAD4 : array,2D
        DESCRIPTION : Data lieing in quadrant 4.

    """
    SIGN = np.sign(DATA)
    ## Splitting into quadrants
    QUAD1 = np.zeros((SIGN.tolist().count([1, 1]), DATA.shape[1]))
    QUAD2 = np.zeros((SIGN.tolist().count([-1, 1]), DATA.shape[1]))
    QUAD3 = np.zeros((SIGN.tolist().count([-1, -1]), DATA.shape[1]))
    QUAD4 = np.zeros((SIGN.tolist().count([1, -1]), DATA.shape[1]))

    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    for i in range(0, SIGN.shape[0]):
        if (SIGN[i, :] - [1, 1] == [0, 0]).all():
            QUAD1[i1,:] = DATA[i,:]
            i1 = i1+1

        elif (SIGN[i, :] - [-1, 1] == [0, 0]).all():
            QUAD2[i2,:] = DATA[i,:]
            i2 = i2+1

        elif (SIGN[i, :] - [-1, -1] == [0, 0]).all():
            QUAD3[i3,:] = DATA[i,:]
            i3 = i3+1

        elif (SIGN[i, :] - [1, -1] == [0, 0]).all():
            QUAD4[i4,:] = DATA[i,:]
            i4 = i4+1
    return QUAD1, QUAD2, QUAD3, QUAD4
