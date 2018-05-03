#Multiplatform home environment
from os.path import expanduser
home = expanduser("~")
from time import strftime
date = strftime("%d-%m-%y")
import configargparse
#Command-Line Option and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.nn_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='config file path')
config.add_argument('-c','--cpu', action="store", type=int, default=1, 
                    help="The number of CPUs to use to do the computation (default: -1 'all CPUs')")
config.add_argument('--sample', action='store', default=home + '/machine-learning/csv/Neural_network_Example_summary.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'Neural_network_Example_summary.csv')")
config.add_argument('-t','--tensorboard', nargs='?', const=home + '/machine-learning/logs/tensorboard/regression' + date, 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const=home + "/machine-learning/snapshots/%s_trained_model.h5" % date, help="Save model to disk")
config.add_argument('-v','--verbose', action="store_true", help="Verbose")
config.add_argument('-p','--predict', nargs='+',
                    help="Sample data for model to make a prediction on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0 1 0 0 68 77 39 94 68 2 1 -106.400001525879 26 278 0 1684 34.9885726158")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/03-05-18_best_model.h5', help="Path to saved model", nargs='?')

options = config.parse_args()
n_cpu = options.cpu
sample = options.sample
path_to_tensorboard = options.tensorboard
verbose = options.verbose
cp = options.save
user_observation = options.predict
#load = options.load
saved_model = options.load

print("\n")
print(options)
print("\n----------\n")
print(config.format_values())
print("\n----------\n")

#Neural Network

#Preparing data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import strftime
#from turicreate import SFrame

#Importing dataset
dataset = pd.read_csv(sample)

#Drop aligned sequence
#dataset = dataset.drop(dataset.columns[0], axis=1)

#Column names
header = list(dataset)

#Looking into sframe as an alternative to pandas, has s3 support
#Tensorflow also has s3 and GCP support if you install from source and enable it
#sf = SFrame(data=dataset)
#sf.explore()

#Length of input, will be used when building model
input_dim = len(dataset.columns) - 1

#Input layer
X = dataset.iloc[: , 0:input_dim].values
#Output layer
Y = dataset.iloc[: , input_dim:(input_dim + 1)].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder

#Encoding values, assigning each catagory a number

#Encoding NHEJ
labelencoder_nhej = LabelEncoder()
X[:,1] = labelencoder_nhej.fit_transform(X[:,1])

#Encoding UNMODIFIED
labelencoder_unmodified = LabelEncoder()
X[:,2] = labelencoder_unmodified.fit_transform(X[:,2])

#Encoding frameshift
labelencoder_frameshift = LabelEncoder()
X[:,15] = labelencoder_frameshift.fit_transform(X[:,15])

#Encoding HDR
labelencoder_hdr = LabelEncoder()
X[:,3] = labelencoder_hdr.fit_transform(X[:,3])

#Spliting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#real_input = X_test

#Feature scaling -1 to +1 because there will be alot of parallel computations
#can use standardisation or normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Dropping aligned sequence here instead of at the beginning so it can later be
#appened to prediction table
test_aligned_sequence, X_test = X_test[:,0],X_test[:,1:input_dim]
X_train = np.delete(X_train,0,1)

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
from keras.utils.vis_utils import plot_model

#Currently using tensorflow backend
from keras import backend as k

#Keras does not have root mean squared error so had to create a custom loss
#to match Amazon's ML model loss function
"""Custom metrics can be passed at the compilation step. 
The function would need to take (y_true, y_pred) as arguments and return a single tensor value."""
def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))

def visualisation(y_pred,Y_test):
    #print(history.history.keys())
    
    if saved_model is None:
        plt.title('Model accuracy (RMSE)')
        plt.plot(history.history['root_mean_squared_error'])
        #plt.plot(history.history['val_root_mean_squared_error'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()
    
    #plt.plot(history.history['loss'])
    #plt.show()

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
    return

if saved_model is not None:
    #load model
    from keras.models import load_model
    model = load_model(saved_model, custom_objects={'root_mean_squared_error': root_mean_squared_error })
    print("Loading model %s from disk" % saved_model)
    model.compile(loss='mse', optimizer='adam', metrics=[root_mean_squared_error])
    score = model.evaluate(X_test, Y_test, verbose=verbose)
    print("%s: %.2f" % (model.metrics_names[1], score[1]))
    
    #predict
    y_pred = model.predict(X_test)
    y_difference = np.subtract(y_pred, Y_test)
    visualisation(y_pred,Y_test)
    
else:
#Neural Network architecture
    def build_regressor():

        #Initialising neural network
        regressor = Sequential()

        #Input layer and first hidden layer with dropout
        regressor.add(Dense(units=9, kernel_initializer='uniform',activation='relu',input_dim=input_dim - 1 ))

        """General tip for the number of nodes in the input layer is that it should be the
        average of the number of nodes in the input and output layer. However this may
        be changed later when parameter tuning using cross validation"""
        #regressor.add(Dropout(rate=0.1))

        #Hidden layer 2 with dropout
        regressor.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))
        """Rectifer function is good for hidden layers and sigmoid function good for output
        layers. Uniform initialises the weights randomly to small numbers close to 0"""
        #regressor.add(Dropout(rate=0.1))

        #Output layer, Densely-connected NN layer
        regressor.add(Dense(units=1, kernel_initializer='uniform'))

        #Visualize model
        plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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

    #Callbacks
    from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint

    #Save loss function progress
    history = History()

    #Model stops if loss function is nan
    terminate_on_nan = TerminateOnNaN()
    
    #Save model
    Checkpoint = ModelCheckpoint(cp, monitor='root_mean_squared_error', verbose=verbose, save_best_only=True)

    #Create tensorboard
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)

    #Python doesn't have switch statements so will use a dictionary later
    if path_to_tensorboard and not cp:
        callbacks=[history, tensorboard, terminate_on_nan]
    elif cp and not path_to_tensorboard:
        callbacks=[history, terminate_on_nan, Checkpoint]
    elif path_to_tensorboard and cp:
        callbacks=[history, terminate_on_nan, Checkpoint, tensorboard]
    else:
        callbacks=[history, terminate_on_nan]
    
    #Have to fit data to model again after cross validation
    history = model.fit(X_train, Y_train, callbacks=callbacks)

    #Prediction on test data, needed to reshape array to subtract element wise
    y_pred = model.predict(X_test).reshape(-1,1)
    y_difference = np.subtract(y_pred, Y_test)
    visualisation(y_pred,Y_test)

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
    print("\n----------\n")
    print ("Prediction is %s" % new_prediction)

#Opens tensorboard in browser if user saves new tensorboard
import subprocess

if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])

#Building complete prediction table
def build_csv(test_aligned_sequence):
    inverse_x = sc.inverse_transform(X_test)
    test_aligned_sequence = np.reshape(test_aligned_sequence, (-1,1))
    pred_set = np.concatenate((test_aligned_sequence,inverse_x,Y_test,y_pred),axis=1)
    header.append('Prediction')
    frame = pd.DataFrame(pred_set, columns=header)
    print(frame.to_string())
    return frame

#correlation matrix
#plt.matshow(frame.corr())

if cp:
    print("\n----------\n")
    input("Press Enter to continue...")

    frame = build_csv(test_aligned_sequence)
    print("\n----------\n")
    save_csv = input("Do you wish to save csv of data? [Y/N] ")

    if save_csv.lower() is 'y' or 'yes':
        default = '{0}/machine-learning/predictions/{1}_miseq_predictions'.format(home, date)
        file_name = input('Type path and file name [Press enter to keep default: {0}]:'.format(default))
        frame.to_csv(file_name or default ,index=False)






