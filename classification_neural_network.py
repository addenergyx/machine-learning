#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Classification Neural Network

#Preparing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc

filename = 'classification_neural_network_Merged26.csv'

#Importing dataset
dataset = pd.read_csv(filename)
dataset = dataset.drop(dataset.columns[-1], axis=1)

#Using one hot encoding with drop first columns to avoid dummy variable trap
dataset = pd.get_dummies(dataset,drop_first=True)
length_X = len(dataset.columns) + 1

outputset = pd.read_csv(filename, names=['ins/dels'], header=0)

#Input layer
X = dataset.iloc[: , 0:length_X].values
del dataset

#Output layer
#Y = dataset.iloc[: , 112:113].values

input_dim = len(X[0])

#Encoding pairs
from sklearn.preprocessing import LabelEncoder
#OneHotEncoder, LabelBinarizer

labelencoder_output = LabelEncoder()
#onehotencode = OneHotEncoder()
#binary = LabelBinarizer()

#one hot encoder does not work with negative numbers
#en_y = binary.fit_transform(outputset)
#in_y = binary.inverse_transform(en_y)

Y_vector = labelencoder_output.fit_transform(outputset)
del outputset

Y_vector = Y_vector.reshape(-1,1)

from keras.utils import np_utils

# one-vs-all
dummy_y = np_utils.to_categorical(Y_vector)
del Y_vector

#onehotencoder = OneHotEncoder()

#Y_array = onehotencoder.fit_transform(Y_array).toarray()


#cols_2_encode = range(0,112)

#for col in cols_2_encode:
#    X[:,col] = labelencoder_pairs.fit_transform(X[:,col])  

#X = onehotencoder.fit_transform(X).toarray()

#Spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

# Number of catagories
y_catagories = len(Y_test[0])
#number of rows - outcome_size = len(Y_test)

# delete references
del X, dummy_y 

#manually invoke garbage collection
gc.collect()

#Feature scaling
'''
don't need to feature scale because all results are binary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

import keras 

#Initialises Neural Network
from keras.models import Sequential

#Creates layers in the Neural Network
from keras.layers import Dense 

#Regularization
from keras.layers import Dropout

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from keras.wrappers.scikit_learn import KerasClassifier

#Neural Network architecture
def build_classifier():
    #Initialising neural network
    classifier = Sequential()
    
    #Input layer and first hidden layer with dropout
    classifier.add(Dense(units=200, kernel_initializer='uniform',activation='relu',input_dim=input_dim))
    """General tip for the number of nodes in the input layer is that it should be the
    average of the number of nodes in the input and output layer. However this may
    be changed later when parameter tuning using cross validation"""
    #regressor.add(Dropout(rate=0.1))
    
    #Hidden layer 2 with dropout
    classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
    """Rectifer function is good for hidden layers and sigmoid function good for output
    layers. Uniform initialises the weights randomly to small numbers close to 0"""
    #regressor.add(Dropout(rate=0.1))
    
    #Output layer, Densely-connected NN layer
    #sigmoid vs softmax
    classifier.add(Dense(units=y_catagories, kernel_initializer='uniform', activation='softmax'))
    
    #Complie model
    # Categorical crossentropy didn't make sense, it should be a sum of 
    # binary crossentropies because it is not a probability distribution over the labels but individual probabilities over every label individually    
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint 

from time import strftime

date = strftime("%d-%m-%y")

tensorboard = TensorBoard(log_dir='./logs/tensorboard/classification/' + date, histogram_freq=0, write_graph=True, write_images=True)

classifier = KerasClassifier(build_fn=build_classifier, epochs=24, batch_size=10000)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(classifier, X_train, Y_train, cv=kfold, n_jobs=-1)
print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

classifier.fit(X_train,Y_train, callbacks=[tensorboard])

#probability of different outcomes
y_prob = classifier.predict_proba(X_test)

# pred_test = labelencoder_output.inverse_transform(Y_train)

# most likely output, not really useful as wildtype (0) will always be most likely
y_pred = classifier.predict(X_test)

#classes = classifier.classes_
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, y_pred)



















