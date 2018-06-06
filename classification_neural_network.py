#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Classification Neural Network

#Preparing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#filename = 'classification_neural_network_Merged26.csv'
filename = 'csv/classification_50bp_miseq26_merged_data.csv'

#Importing dataset
dataset = pd.read_csv(filename)

#y column
# .iloc is purely integer-location based indexing for selection by position
# instead of by column name dataset["ins/dels"]
outputset = dataset.iloc[:,-1]

dataset = dataset.drop(dataset.columns[-1], axis=1)

#Using one hot encoding with drop first columns to avoid dummy variable trap
non_dropout = pd.get_dummies(dataset, drop_first=False)
dataset = pd.get_dummies(dataset,drop_first=True)
headers = list(dataset)
full_headers = list(non_dropout)

length_X = len(dataset.columns) + 1

#outputset = pd.read_csv(filename, names=['ins/dels'], header=0)

#ordered unique output possibilities
myset = set(outputset)
mylist = list(myset)
mylist.sort()

output_dict = {}

for x in range(len(myset)):
    key = x
    value =  min(myset)
    myset.remove(value)
    output_dict[key] = value

#Input layer
X = dataset.iloc[: , 0:length_X].values
#del dataset

#Output layer
#Y = dataset.iloc[: , 112:113].values

input_dim = len(X[0])

#Encoding pairs
from sklearn.preprocessing import LabelEncoder #OneHotEncoder, LabelBinarizer

labelencoder_output = LabelEncoder()
#onehotencode = OneHotEncoder()
#binary = LabelBinarizer()

#one hot encoder does not work with negative numbers
#en_y = binary.fit_transform(outputset)
#in_y = binary.inverse_transform(en_y)

Y_vector = labelencoder_output.fit_transform(outputset)
#del outputset

Y_vector = Y_vector.reshape(-1,1)

from keras.utils.np_utils import to_categorical
import keras
from numpy import argmax

# one-vs-all
dummy_y = to_categorical(Y_vector)

a = argmax(dummy_y, axis=1)

#del Y_vector

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
#del X, dummy_y 

#manually invoke garbage collection
#gc.collect()

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

from sklearn.metrics import confusion_matrix

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
    classifier.add(Dropout(rate=0.1))
    
    #Hidden layer 2 with dropout
    classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
    """Rectifer function is good for hidden layers and sigmoid function good for output
    layers. Uniform initialises the weights randomly to small numbers close to 0"""
    classifier.add(Dropout(rate=0.1))
    
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

classifier = KerasClassifier(build_fn=build_classifier, epochs=24, batch_size=1000)

'''
kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(classifier, X_train, Y_train, cv=kfold, n_jobs=-1)
print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

classifier.fit(X_train,Y_train, callbacks=[tensorboard])

#probability of different outcomes
y_prob = classifier.predict_proba(X_test)

# pred_test = labelencoder_output.inverse_transform(Y_train)

# most likely output, not really useful as wildtype (0) 
#will probably always be most likely
y_pred = classifier.predict(X_test)
#y_pred = y_pred.reshape(-1,1)

encoded_top5 = (-y_prob).argsort()[:,0:5]
encoded_y_true = np.argsort(Y_test)[:,-1]

# Vectorise function to map back to true values
def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

vfunc = np.vectorize(map_func)

top5 = vfunc(encoded_top5, output_dict)
y_true = vfunc(encoded_y_true, output_dict).reshape(-1,1) 
y_prediction = vfunc(y_pred, output_dict).reshape(-1,1)

# Mapping sequence

def seq_map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

vfunc2 = np.vectorize(seq_map_func)

seq_dict = {}

for x in range(len(headers)):
    key = x
    value = headers[x]
    seq_dict[key] = value

# Test sequence: TGAGAAAACCAAACAGGGTGTGGCAGAAGCAGCAGGAAAGACAAAAGAGG
#a_seq = [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
# For testing swap x in mapping function enumerate(x) with a_seq 
# and execute lines inside function (not the function itself) 

# Index only searches through the list for the first instance 
#To find more matches using list comprehension
#a_seq.index(0)

import re

missing_base_dict = {}

# Finds all the first occurences of each base possibility
for x in range(1,26):
    key = x
    value = next (i for i in full_headers if re.match(str(x),i))
    missing_base_dict[key] = value
    

def mapping(x):
    bases = [i for i, e in enumerate(x) if e == 1]
    ordered_bases = vfunc2(bases,seq_dict)
    
    for i in range(1,26):
        try:
            if re.match('^{0}\D.*'.format(i),ordered_bases[i-1]) is None: 
                ordered_bases = np.insert(ordered_bases, i-1, missing_base_dict.get(i))
        except (IndexError):
            ordered_bases = np.append(ordered_bases, missing_base_dict.get(i))


    for base in range(len(ordered_bases)):
        ordered_bases[base] = re.sub("\d+_","",ordered_bases[base])
    seq = ''.join(ordered_bases)
    return seq

# This process takes a long time, should look into a quickier method
print("Remapping data...")
sequences = np.apply_along_axis( mapping, axis=1, arr=X_test).reshape(-1,1)
print("Done")

# Dataset of sequence with top 5 predicted in/del
pred_set = np.concatenate((sequences,y_true,top5),axis=1)
pred_set_headers = ['Sequence', 'Actual Result','Predicted Most Likely in/del','2nd','3rd','4th','5th']
frame = pd.DataFrame(pred_set, columns=pred_set_headers)

'''
could look into making the display more meaningful data such as a colour coding 
for the percentage given by the y_prob matrix or colour coding based on the 
range of the dataset similar to when viewing the top5 variable in spyder
''' 

from turicreate import SFrame
sf = SFrame(frame)
sf.explore()
sf.show()



'''
x = 1
X_test[1,x]
full_headers[x]

import re
for row in range(len(X_test)):
    for column in range(len(X_test[0])):
        if X_test[row,column] == 1:
            seq = headers[column]
            seq = re.sub("\d+_","", seq)

for i in range(len(y_pred)):
    y_pred[i] = output_dict.get(int(y_pred[i]))

frame = pd.DataFrame(y_prob, columns=mylist)
'''

#classes = classifier.classes_
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_test, y_pred)

#first , X_test = X_test[0], X_test[0:-1,:]
















