"""
Created on Wed Mar 8, 2023

Author:  Shubham Raheja (shubhamraheja1999@gmail.com)
         Deeksha Sethi  (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of CFX+LSTM on the Ionosphere dataset.

Dataset Source: https://archive.ics.uci.edu/ml/datasets/ionosphere

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Codes import k_cross_validation

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    b (Bad)      -     0
    g (Good)     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    
    ionosphere      -   Complete Ionosphere dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
_______________________________________________________________________________

CFX hyperparameter description:
_______________________________________________________________________________

    INITIAL_NEURAL_ACTIVITY         -   Initial Neural Activity.
    EPSILON                         -   Noise Intensity.
    DISCRIMINATION_THRESHOLD        -   Discrimination Threshold.
    
    Source: Harikrishnan N.B., Nithin Nagaraj,
    When Noise meets Chaos: Stochastic Resonance in Neurochaos Learning,
    Neural Networks, Volume 143, 2021, Pages 425-435, ISSN 0893-6080,
    https://doi.org/10.1016/j.neunet.2021.06.025.
    (https://www.sciencedirect.com/science/article/pii/S0893608021002574)
_______________________________________________________________________________

DL hyperparameter description:
_______________________________________________________________________________

    LSTM units - Positive integer, dimensionality of the output space. 
    Source : Keras 
    (https://keras.io/api/layers/recurrent_layers/lstm/)
    
    Dense layer units - Positive integer, dimensionality of the output space.
    Source : Keras 
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dense layer activation - Activation function of the dense layer.
    Source : Keras
    (https://keras.io/api/layers/core_layers/dense/)
    
    Dropout rate - Float between 0 and 1. Fraction of the input units to drop.
    The dropout layer randomly sets input units to 0 with a frequency of rate 
    at each step during training time, which prevents overfitting. 
    Source : Keras
    (https://keras.io/api/layers/regularization_layers/dropout/)
    
    Learning rate - The learning rate is a hyperparameter that controls how 
    much to change the model in response to the estimated error each time the 
    odel weights are updated.
    
    Here the learning rate is being set for the Adam optimizer. 
    
    Adam optimization is a stochastic gradient descent method 
    that is based on adaptive estimation of first-order and second-order 
    moments.
    Source : Keras
    (https://keras.io/api/optimizers/adam/)
    
    
    The above mentioned hyperparameters are tuned using KerasTuner. It is a 
    general-purpose hyperparameter tuning library.
    Source : Keras
    (https://keras.io/guides/keras_tuner/)
    
_____________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and finds  their 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
    
_______________________________________________________________________________

'''    
# Import the IONOSPHERE Dataset 
ionosphere = np.array(pd.read_csv('ionosphere_data.txt', sep=",", header=None))

# Reading data and labels from the dataset
X, y = ionosphere[:,range(0,ionosphere.shape[1]-1)], ionosphere[:,ionosphere.shape[1]-1].astype(str)

# Norm: B -> 0;  G -> 1
y = y.reshape(len(y),1)
y = np.char.replace(y, 'b', '0', count=None)
y = np.char.replace(y, 'g', '1', count=None)
y = y.astype(int)

# Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Normalisation - Column-wise
X_train_norm = X_train
X_train_norm[:,range(2,X_train.shape[1])] = (X_train[:,range(2,X_train.shape[1])]-np.min(X_train[:,range(2,X_train.shape[1])],0))/(np.max(X_train[:,range(2,X_train.shape[1])],0)-np.min(X_train[:,range(2,X_train.shape[1])],0))
X_train_norm = X_train_norm.astype(float)
X_test_norm = X_test
X_test_norm[:,range(2,X_test.shape[1])] = (X_test[:,range(2,X_test.shape[1])]-np.min(X_test[:,range(2,X_test.shape[1])],0))/(np.max(X_test[:,range(2,X_test.shape[1])],0)-np.min(X_test[:,range(2,X_test.shape[1])],0))
X_test_norm = X_test_norm.astype(float)


# Validation
INITIAL_NEURAL_ACTIVITY = [0.680]
DISCRIMINATION_THRESHOLD = [0.969]
EPSILON = [0.164]
k_cross_validation(X_train_norm, y_train, X_test_norm, y_test, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON)


