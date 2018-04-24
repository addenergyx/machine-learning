#Command-Line Option and Argument Parsing
import argparse
parser = argparse.ArgumentParser(prog='Deletion/Insertion Spread Neural Network', 
                                 description='This is a program to predict insertions and deletions on a sequence based on given data')

parser.add_argument('-c','--cpu', action="store", type=int, default=-1, 
                    help="The number of CPUs to use to do the computation (default: -1 ‘all CPUs')")
parser.add_argument('--sample', action='store', default='Neural_network_Example_summary.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'Neural_network_Example_summary.csv')")
parser.add_argument('-t','--tensorboard', action="store_true", 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
parser.add_argument('-s','--save', action="store_true", help="Save model to disk")
parser.add_argument('-v','--verbose', action="store_true", help="Verbose")
parser.add_argument('-p','--predict', nargs='+',
                    help="Sample data for model to make a prediction on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%Reads. For example: 0 1 0 0 68 77 39 94 68 2 1 -106.400001525879 26 278 0 1684 34.9885726158")

#Later will need to add arguments for user to predict data
args = parser.parse_args()
n_cpu = args.cpu
sample = args.sample
board = args.tensorboard
verbose = args.verbose
cp = args.save
user_observation = args.predict

#Neural Network

#Preparing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import strftime
#from turicreate import SFrame

#Importing dataset
dataset = pd.read_csv(sample)
list(dataset)
dataset = dataset.drop(dataset.columns[0], axis=1)

#Looking into sframe as an alternative to pandas
#sf = SFrame(data=dataset)
#sf.explore()

#Input layer
X = dataset.iloc[: , 0:17].values
#Output layer
Y = dataset.iloc[: , 17:18].values

#Length of input, will be used when building model
input_dim = len(X[0])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder

#Encoding values, assigning each catagory a number

#Encoding NHEJ
labelencoder_nhej = LabelEncoder()
X[:,0] = labelencoder_nhej.fit_transform(X[:,0])

#Encoding UNMODIFIED
labelencoder_unmodified = LabelEncoder()
X[:,1] = labelencoder_unmodified.fit_transform(X[:,1])

#Encoding frameshift
labelencoder_frameshift = LabelEncoder()
X[:,14] = labelencoder_frameshift.fit_transform(X[:,14])

#All results are false in this sample so don't need this feature
#Encoding HDR
labelencoder_hdr = LabelEncoder()
X[:,2] = labelencoder_hdr.fit_transform(X[:,2])

#Spliting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

real_input = X_test

#Feature scaling -1 to +1 because there will be alot of parallel computations
#can use standardisation or normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Must fit object to training set then transform it
X_train = sc.fit_transform(X_train)

#Scaled on same basis as X_train because of sc, however not influenced by training data range 
X_test = sc.transform(X_test)

"""Algorithm will converge much faster with feature scalling
Don't need to apply feature scaling on Y if it's a classifaction problem with a 
catagorical dependent variable. Will need to apply feature scaling in a regression output"""

"""Use alot of objects so in the future it would be easy to implement a different algorithm
Would only need to change this library"""
import keras 

#Initialises Neural Network
from keras.models import Sequential

#Creates layers in the Neural Network
from keras.layers import Dense 

#Regularization
from keras.layers import Dropout

#K-fold cross valdation, reduces variance and bias
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error 

#Currently using tensorflow backend
from keras import backend as k

#Keras does not have root mean squared error so had to create a custom loss
#to match Amazon's ML model loss function
"""Custom metrics can be passed at the compilation step. 
The function would need to take (y_true, y_pred) as arguments and return a single tensor value."""
def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))

#Neural Network architecture
def build_regressor():
    #Initialising neural network
    regressor = Sequential()
    #Input layer and first hidden layer with dropout
    regressor.add(Dense(units=6, kernel_initializer='uniform',activation='relu',input_dim=input_dim))
    """General tip for the number of nodes in the input layer is that it should be the
    average of the number of nodes in the input and output layer. However this may
    be changed later when parameter tuning using cross validation"""
    #regressor.add(Dropout(rate=0.1))
    #Hidden layer 2 with dropout
    regressor.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    """Rectifer function is good for hidden layers and sigmoid function good for output
    layers. Uniform initialises the weights randomly to small numbers close to 0"""
    #regressor.add(Dropout(rate=0.1))
    #Output layer, Densely-connected NN layer
    regressor.add(Dense(units=1, kernel_initializer='uniform'))
    #Complie model    
    regressor.compile(optimizer='adam', loss='mse', metrics=[root_mean_squared_error])
    return regressor

model = KerasRegressor(build_fn=build_regressor, epochs=100, batch_size=10 )
#Accuracy is the 10 accuracies returned by k-fold cross validation
#Most of the time k=10

from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True)

accuracy = cross_val_score(estimator=model, X = X_train, y = Y_train, cv=kfold, n_jobs=n_cpu, verbose=verbose)
#n_jobs is number of cpu's, -1 is all

#Mean accuracies and variance
loss_mean = accuracy.mean()
loss_variance = accuracy.std()

date = strftime("%d-%m-%y")

#Callbacks
from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint 

#Save loss function progress
history = History()

#Model stops if loss function is nan                          
terminate_on_nan = TerminateOnNaN()

#Save model
Checkpoint = ModelCheckpoint("./snapshots/%s_trained_model.h5" % date, monitor='root_mean_squared_error', verbose=verbose, save_best_only=True)

#Creates tensorboard
tensorboard = TensorBoard(log_dir='./logs/tensorboard/' + date, histogram_freq=0, write_graph=True, write_images=True)

#Python doesn't have switch statements so will use a dictionary later
if board and not cp:
    callbacks=[history, tensorboard, terminate_on_nan]
elif cp and not board:
    callbacks=[history, terminate_on_nan, Checkpoint]
elif board and cp:
    callbacks=[history, terminate_on_nan, Checkpoint, tensorboard]
else:
    callbacks=[history, terminate_on_nan]
    
#Have to fit data to model again after cross validation
history = model.fit(X_train, Y_train, callbacks=callbacks)

#print(history.history.keys())

plt.title('Model accuracy (RMSE)')
plt.plot(history.history['root_mean_squared_error'])
#plt.plot(history.history['val_root_mean_squared_error'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

#plt.plot(history.history['loss'])
#plt.show()

#Prediction on test data, needed to reshape array to subtract element wise
y_pred = model.predict(X_test).reshape(-1,1)
y_difference = np.subtract(y_pred, Y_test)

#True results
plt.title('Actual Results')
plt.plot(Y_test)
plt.show()

#graphs comparing predicted and actual results
plt.title('Acutal results vs Predicted results')
plt.plot(Y_test)
plt.plot(y_pred)
plt.legend(['Actual', 'Predicted'], loc='best')
plt.show()

plt.title('Predicted results vs Actual results')
plt.plot(y_difference)
plt.show()

from math import sqrt

#Results variance and mean, best possible score is 1.0 for variance
variance = explained_variance_score(Y_test, y_pred)
rmse_value = sqrt(mean_squared_error(Y_test, y_pred))

#Single prediction
if user_observation:

    str_observation = ','.join(user_observation)
    import re
    fixed_observation = re.sub('true', '1', str_observation, flags=re.IGNORECASE)
    fixed_observation = re.sub('false', '0', fixed_observation, flags=re.IGNORECASE)

    #list comprehension didn't work
    #user_observation = [1 if x is 'True' else x for x in user_observation]
    #user_observation = [0 if x is 'False' else x for x in user_observation]

    list_observation = [float(i) for i in fixed_observation.split(',')]
    new_prediction = model.predict(sc.transform(np.array([list_observation])))
    print ("Prediction is %s" % new_prediction)

#Open tensorboard if user creates new model
import subprocess

if board:
    subprocess.call(['tensorboard', '--logdir', './logs/tensorboard/' + date])

#save model
#history.save('%s_trained_model.h5' % date)

#load model
# if load_model <- code
def load_model():
    from keras.models import load_model
    loaded_model = load_model('./snapshots/08-03-18_best_model.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error })
    print("Loaded model from disk")
    loaded_model.compile(loss='mse', optimizer='adam', metrics=[root_mean_squared_error])
    score = loaded_model.evaluate(X_test, Y_test, verbose=verbose)
    print("%s: %.2f" % (loaded_model.metrics_names[1], score[1]))
    return

