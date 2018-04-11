"""This algorithm will be used to decide which parameters
will be best for the model based on the data
Seperating tuning from the neural network as this shouldn't 
always be run when the network runs"""

#Preparing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv('csv/Neural_network_Example_summary.csv')
dataset = dataset.drop(dataset.columns[3:6], axis=1)

#Input layer
X = dataset.iloc[: , 1:14].values
#Output layer
Y = dataset.iloc[: , 14:15].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder

#Encoding NHEJ
labelencoder_X_nhej = LabelEncoder()
X[:,0] = labelencoder_X_nhej.fit_transform(X[:,0])

#Encoding UNMODIFIED
labelencoder_X_unmodified = LabelEncoder()
X[:,1] = labelencoder_X_unmodified.fit_transform(X[:,1])

#Encoding HDR
#labelencoder_X_hdr = LabelEncoder()
#X[:,2] = labelencoder_X_hdr.fit_transform(X[:,2])

#Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras

#Initialises Neural Network
from keras.models import Sequential

#Creates layers in the Neural Network
from keras.layers import Dense 

#Regularization
from keras.layers import Dropout

#Parameter tuning
from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import GridSearchCV
from keras import backend as k

#Neural Network architecture
def build_regressor(optimizer, initializer, nodes1, nodes2, dropout, hidden_layers):
    #Initialising neural network
    regressor = Sequential()

    #Input layer and first hidden layer
    regressor.add(Dense(units=nodes1, kernel_initializer=initializer, activation='relu',input_dim=13))
    regressor.add(Dropout(rate=dropout))

    #Tuning number of hidden layers
    for i in range(hidden_layers):
        regressor.add(Dense(units=nodes2, kernel_initializer=initializer, activation='relu'))
        regressor.add(Dropout(rate=dropout))

    #Output layer
    regressor.add(Dense(units=1, kernel_initializer=initializer))

    def root_mean_squared_error(y_true, y_pred):
        return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))

    #Gradient Descent
    regressor.compile(optimizer=optimizer,loss='mean_squared_error', metrics=[root_mean_squared_error])

    return regressor

regressor = KerasRegressor(build_fn=build_regressor)

#Grid search
parameters = {'batch_size' : [50], 
              'epochs' : [100], 
              'optimizer' : ['adam'],
              'initializer' : ['uniform', 'glorot_uniform'],
              'nodes1' : [6],
              'nodes2' : [6],
              'hidden_layers' : [2,3,4,5,6],
              'dropout' : [0.1,0.2,0.3]
              }
              
grid_search = GridSearchCV(estimator=regressor, param_grid=parameters, cv=10, n_jobs=-1, scoring='mean_squared_error')

grid_search = grid_search.fit(X_train, Y_train)

#Displays best parameters
best_parameters =  grid_search.best_params_
best_accuracy = grid_search.best_score_
