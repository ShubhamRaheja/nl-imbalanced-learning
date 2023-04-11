"""
Created on Mon Mar 13, 2023

Author:  Shubham Raheja (shubhamraheja1999@gmail.com)
         Deeksha Sethi  (deeksha.sethi03@gmail.com)
Code Description: A python code to tune the hyperparameters of stand-alone LSTM on the Haberman's Survival dataset.
Dataset Source: https://archive.ics.uci.edu/ml/datasets/haberman's+survival

"""

import os
seed_value = 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
import numpy as np
import tensorflow as tf
import random
np.random.seed(39)
random.seed(1256)
tf.random.set_seed(91)

import pandas as pd
from sklearn.metrics import (f1_score,accuracy_score)
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import callbacks

'''
_______________________________________________________________________________

Rule used for renaming class labels: (Class Label - Numeric Code)
_______________________________________________________________________________

    1 (< 5yrs)      -     0
    2 (>= 5yrs)     -     1
_______________________________________________________________________________

Variable description:
_______________________________________________________________________________
    
    haberman        -   Complete Haberman's Survival dataset.    
    X               -   Data attributes.
    y               -   Corresponding labels for X.
    X_train         -   Data attributes for training (80% of the dataset).
    y_train         -   Corresponding labels for X_train.
    X_test          -   Data attributes for testing (20% of the dataset).
    y_test          -   Corresponding labels for X_test.
    X_train_norm    -   Normalizised training data attributes (X_train).
    X_test_norm     -   Normalized testing data attributes (X_test).
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
    
_______________________________________________________________________________

Performance metric used:
_______________________________________________________________________________

    Macro F1-score (F1SCORE/ f1) -
    The F1 score can be interpreted as a harmonic mean of the precision and 
    recall, where an F1 score reaches its best value at 1 and worst score at 
    0. The relative contribution of precision and recall to the F1 score are 
    equal; 'macro' calculates metrics for each label, and find stheir 
    unweighted mean. This does not take label imbalance into account.
    Source: Scikit Learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
_______________________________________________________________________________

'''    
# Import the HABERMAN'S SURVIVAL Dataset 
haberman = np.array(pd.read_csv('haberman.txt', sep=",", header=None))

# Reading data and labels from the dataset
X, y = haberman[:,range(0,haberman.shape[1]-1)], haberman[:,haberman.shape[1]-1].astype(str)
y = np.char.replace(y, '1', '0', count=None)
y = np.char.replace(y, '2', '1', count=None)
y = y.astype(int)
y = y.reshape(len(y),1)

# Binary matrix representation of the labels
y = to_categorical(y)

# Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Normalisation - Column-wise
X_train_norm = (X_train - np.min(X_train,0))/(np.max(X_train,0) - np.min(X_train,0))
X_test_norm = (X_test - np.min(X_test,0))/(np.max(X_test,0) - np.min(X_test,0))

# Reshaping as tensor for LSTM algorithm.
X_train_norm = np.reshape(X_train_norm ,(X_train_norm.shape[0], 1, X_train_norm.shape[1])) 
X_test_norm = np.reshape(X_test_norm ,(X_test_norm.shape[0], 1, X_test_norm.shape[1])) 

# Algorithm - LSTM

PATH = os.getcwd()
RESULT_PATH = PATH + '/SA-TUNING/RESULTS/' 
RESULT_PATH_DL = PATH + '/SA-logs/check-points/'

# Load the tuned hyperparamaters
units_ = np.load(RESULT_PATH+"/h_Units.npy")
dense = np.load(RESULT_PATH+"/h_Dense.npy")
dropout_rate = np.load(RESULT_PATH+"/h_DropoutRate.npy").item()
learning_rate_ = np.load(RESULT_PATH+"/h_LearningRate.npy")
best_epoch = np.load(RESULT_PATH+"/h_BestEpoch.npy")
F1SCORE_train = np.load(PATH+"/TESTING-RESULTS/SA-RESULT/SA_Train_F1SCORE.npy")

try:
    os.makedirs(RESULT_PATH)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_DL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_DL)
    
# Build the model with best hyperparameters
def model_builder(): 
    model = Sequential()       
    model.add(LSTM(units=units_, input_shape=(X_train_norm.shape[1],X_train_norm.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=dense,activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=learning_rate_),
                  metrics = ['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath=RESULT_PATH_DL + "checkpoint.hdf5", verbose=1, monitor='loss', mode='min', save_best_only=True)
    model.fit(X_train_norm,
              y_train,
              epochs = best_epoch,
              batch_size=32,
              verbose=1,
              shuffle=True,
              callbacks=[checkpointer])
              #validation_split = 0.2)
    return model

# Get the model
model = model_builder()
model.load_weights(RESULT_PATH_DL + "checkpoint.hdf5")

# Make predictions with trained model on test data
y_pred_testdata = np.argmax(model.predict(X_test_norm), axis=1)
y_test= np.argmax(y_test,axis=1)
ACC = accuracy_score(y_test, y_pred_testdata)*100
F1SCORE = f1_score(y_test, y_pred_testdata, average="macro")
print('TRAIN: F1 SCORE = ', F1SCORE_train)
print('TEST: F1 SCORE = ', F1SCORE)

# Create a path for saving the testing F1 Score
RESULT_PATH_FINAL = PATH + '/' +'TESTING-RESULTS/SA-RESULT'

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)
    
# Save the F1 Score for Standalone LSTM Algorithm
np.save(RESULT_PATH_FINAL+"/SA_Test_F1SCORE.npy", (F1SCORE)) 