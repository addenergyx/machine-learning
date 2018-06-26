#!/usr/bin/env python3

from os.path import expanduser, isfile 
from os import listdir
from random import shuffle
from time import strftime
import configargparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from turicreate import SFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential #Initialises Neural Network
from keras.layers import Dense #Creates layers in the Neural Network
from keras.layers import Dropout #Regularization
from keras.wrappers.scikit_learn import KerasRegressor 
#from sklearn.model_selection import cross_val_score #K-fold cross valdation, reduces variance and bias
from sklearn.metrics import explained_variance_score, mean_squared_error 
from keras.utils.vis_utils import plot_model
from keras import backend as k #Currently using tensorflow backend
from bokeh.plotting import figure, output_file, show
from keras.models import load_model 
from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint
from math import sqrt
import subprocess
import re

'''
Putting moudles at the top instead of within functions as the latter will make 
calls to the function take longer. However can look into the cost of importing modules in
optional functions and if statments like generator_batch()  
'''

#Multiplatform home environment
home = expanduser("~")
date = strftime("%d-%m-%y")

#Command-Line Options and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.nn_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='Configuration file path, command-line values override config file values')
config.add_argument('-c','--cpu', action="store", type=int, default=-1, 
                    help="The number of CPUs to use to do the computation (default: -1 'all CPUs')")
config.add_argument('--sample', action='store', default=home + '/machine-learning/csv/regression/Neural_network_Example_summary.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'Neural_network_Example_summary.csv')")
config.add_argument('-t','--tensorboard', nargs='?', const='{0}/machine-learning/tensorboard/regression/{1}_regression_tensorboard'.format(home, date), 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const=home + "/machine-learning/snapshots/regression/%s_regression_trained_model.h5" % date, help="Save model to disk")
config.add_argument('-v','--verbose', action="store_true", help="Verbosity mode")
config.add_argument('-p','--predict', nargs='+',
                    help="Can parse a single observation or file containing multiple observations to make predictions on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0,1,0,0,68,77,39,94,68,2,1,-106.400001525879,26,278,0,1684,34.988572615")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/regession/03-05-18_best_model.h5', help="Path to saved model", nargs='?')
config.add_argument('-b', '--batch', nargs='?', const=home + '/machine-learning/smallsamplefiles', 
                    help="Path to directory containing multiple files with data in the correct format. Default: ~/machine-learning/smallsamplefiles/")
config.add_argument('-m','--multivariate', action="store_true", help="Multivariance mode")

# Configuration variables
options = config.parse_args()

n_cpu = options.cpu
sample = options.sample
path_to_tensorboard = options.tensorboard
verbose = options.verbose
cp = options.save
user_observation = options.predict
saved_model = options.load
path_to_batch = options.batch
multivariate = options.multivariate

if (path_to_batch is not None and sample is not None ):
    # Batch flag will override sample
    sample = None

print("\n")
print(options)
print("\n----------\n")
print(config.format_values())
print("\n----------\n")

#Neural Network

#Preparing data

#Encoding categorical data
labelencoder = LabelEncoder()

#Feature scaling -1 to +1 because there will be alot of parallel computations
#can use standardisation or normalisation
sc = StandardScaler()
    
def build_data(sample):
    
    #Importing dataset
    dataset = pd.read_csv(sample, error_bad_lines=False)

    #Drop aligned sequence
    #dataset = dataset.drop(dataset.columns[0], axis=1)
    
    #Column names
    headers = list(dataset)
    
    #Length of input, will be used when building model
    input_dim = len(headers) - (3 if multivariate else 2) 
        
    #Encoding values, assigning each catagory a number
        
    #Encoding NHEJ
    dataset['NHEJ'] = labelencoder.fit_transform(dataset['NHEJ'])
    
    #Encoding UNMODIFIED
    dataset['UNMODIFIED'] = labelencoder.fit_transform(dataset['UNMODIFIED'])
    
    #Encoding frameshift
    dataset['frameshift'] = labelencoder.fit_transform(dataset['frameshift'])
    
    #Encoding HDR
    dataset['HDR'] = labelencoder.fit_transform(dataset['HDR'])
    
    #Input layer
    X = dataset.iloc[:, 0:input_dim + 1].values
            
    #Output layer
    Y = dataset.iloc[:, input_dim + 1:20].values if multivariate else dataset.iloc[: , (input_dim + 1):(input_dim + 2)].values 
    
    #Spliting dataset into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #Dropping aligned sequence here instead of at the beginning so it can later be
    #appened to prediction table
    # input_dim or input_dim + 1
    test_aligned_sequence, X_test = X_test[:,0],X_test[:,1:input_dim + 1]
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
    return X_test, Y_test, input_dim, X_train, Y_train, test_aligned_sequence, headers

#Building complete prediction table
def build_csv(test_aligned_sequence):
    inverse_x = sc.inverse_transform(X_test)
    test_aligned_sequence = np.reshape(test_aligned_sequence, (-1,1))
    pred_set = np.concatenate((test_aligned_sequence,inverse_x,Y_test,y_pred),axis=1)
    
    if multivariate:
       if "Insertion prediction" and "Deletion prediction" not in headers:
           headers.append('Deletion prediction')
           headers.append('Insertion prediction')
    else:
        if "Prediction" not in headers:    
            headers.append('Prediction')
    
    frame = pd.DataFrame(pred_set, columns=headers)
    #print(frame.to_string())
    return frame

#Keras does not have root mean squared error so had to create a custom loss
#to match Amazon's ML model loss function
"""Custom metrics can be passed at the compilation step. 
The function would need to take (y_true, y_pred) as arguments and return a single tensor value."""
def root_mean_squared_error(y_true, y_pred):
    return k.sqrt(k.mean(k.square(y_pred - y_true), axis=-1))

def visualisation(y_pred, Y_test, y_difference, filename='vis.html'):
    #print(history.history.keys())

    #Big datasets are not represented by the graph very well so will only show first
    # 2000 results

    if len(y_difference) > 2000:
        y_pred = y_pred[0:2000]
        Y_test = Y_test[0:2000]
        y_difference = y_difference[0:2000]

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

    plt.title('Difference between predicted results and actual results')
    plt.plot(y_difference)
    plt.show()

    # output to static HTML file
    output_file(filename)
    # create a new plot with a title and axis labels
    plot = figure(title="Difference between predicted results and actual results", x_axis_label='x', y_axis_label='y')
    # add a line renderer with legend and line thickness
    plot.line(range(len(y_difference)), np.ravel(y_difference), legend="Difference", line_width=2)
    # show the results
    show(plot)

    return

if saved_model is not None:
    #load model
    model = load_model(saved_model, custom_objects={'root_mean_squared_error': root_mean_squared_error })
    print("Loading model %s from disk" % saved_model)
    model.compile(loss='mse', optimizer='adam', metrics=[root_mean_squared_error])
    X_test, Y_test = build_data(sample)[0:2]
    score = model.evaluate(X_test, Y_test, verbose=verbose)
    print("%s: %.2f" % (model.metrics_names[1], score[1]))

    #predict
    '''Currently this code is repeated twice (once for each model). Should look into having
    this outside the if/else so it runs for both at the end therefore not repeating code.
    Line 364-366'''
    y_pred = model.predict(X_test)
    y_difference = np.subtract(y_pred, Y_test)

    if multivariate:
        visualisation(y_pred=y_pred[:,0].reshape(-1,1),Y_test=Y_test[:,0].reshape(-1,1), y_difference=y_difference[:,0].reshape(-1,1))
        visualisation(y_pred=y_pred[:,1].reshape(-1,1),Y_test=Y_test[:,1].reshape(-1,1), y_difference=y_difference[:,1].reshape(-1,1))
    else:
        visualisation(y_pred,Y_test, y_difference)

    #Looking into sframe as an alternative to pandas, has s3 support
    #Tensorflow also has s3 and GCP support if you install from source and enable it
    sf = SFrame(build_csv(build_data(sample)[5], build_data(sample)[7]))
    sf.explore()
    sf.show()

else:
#Neural Network architecture

    def build_regressor():

        #Initialising neural network
        regressor = Sequential()

        #Input layer and first hidden layer with dropout
        regressor.add(Dense(units=10, kernel_initializer='uniform',activation='relu',input_dim=input_dim))

        """General tip for the number of nodes in the input layer is that it should be the
        average of the number of nodes in the input and output layer. However this may
        be changed later when parameter tuning using cross validation"""
        regressor.add(Dropout(rate=0.1))

        #Hidden layer 2 with dropout
        regressor.add(Dense(units=10, kernel_initializer='uniform', activation='relu'))
        """Rectifer function is good for hidden layers and sigmoid function good for output
        layers. Uniform initialises the weights randomly to small numbers close to 0"""
        regressor.add(Dropout(rate=0.1))

        #Output layer, Densely-connected NN layer
        #ReLU layer is a Linear layer that converts all negative values to 0.
        #This is necessary for multivariate model as all results should be positive
        if multivariate:
            regressor.add(Dense(units=output_dim, kernel_initializer='uniform', activation='relu' ))
        else:
            regressor.add(Dense(units=output_dim, kernel_initializer='uniform'))

        #Visualize model
        plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        #Complie model
        regressor.compile(optimizer='adam', loss='mse', metrics=[root_mean_squared_error])
        
        return regressor
        
    output_dim = 2 if multivariate else 1 

    #Callbacks

    #Save loss function progress
    history = History()

    #Model stops if loss function is nan, this happens when there is missing data
    terminate_on_nan = TerminateOnNaN()

    #Save model
    Checkpoint = ModelCheckpoint(cp, monitor='root_mean_squared_error', verbose=verbose, save_best_only=True)

    #Create tensorboard
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)     

    def tensorboard_callback():
        callbacks=[history, tensorboard, terminate_on_nan]
        print("\n----------")
        print("Saving tensorboard to {0}".format(path_to_tensorboard))
        print("----------\n")
        return callbacks

    def checkpoint_callback():
        callbacks=[history, terminate_on_nan, Checkpoint]
        print("\n----------")
        print("Saving model to {0}".format(cp))
        print("----------\n")
        return callbacks

    def both_callback():
        callbacks=[history, terminate_on_nan, Checkpoint, tensorboard]
        print("\n----------")
        print("Saving model to {0}\nSaving tensorboard to {1}".format(cp,path_to_tensorboard))
        print("----------\n")
        return callbacks

    def no_callback():
        callbacks=[history, terminate_on_nan]
        return callbacks

    callbacks_dict = {
            path_to_tensorboard and cp is None                         : tensorboard_callback,
            cp and path_to_tensorboard is None                         : checkpoint_callback,
            all (var is not None for var in [path_to_tensorboard, cp]) : both_callback,
            not path_to_tensorboard and not cp                         : no_callback
            }

    def switch_case_callbacks(x):
        return callbacks_dict[x]()

    def generate_batches(files, path_to_batch):
        counter = 0
        # This line is just to make the generator infinite, keras needs this
        while True:
            sample = files[counter]
            sample = path_to_batch + sample
            print(sample)
            counter = (counter + 1) % len(files)            

            # This method would require calling build_data() multiple times wasting cpu
            #X_train, Y_train = build_data(sample)[3:5]

            build = build_data(sample)
            X_train = build[3]
            Y_train = build[4]

            '''
            yield is the keyword in python used for generator expressions. 
            It means that the next time the function is called the execution will 
            start back up at the exact point it left off last time it was called
            This is important for interating chunks of lines in a file or batches 
            of files in a directory like above
            '''
            yield (X_train, Y_train)

    # Keras's scikit-learn wrapper doesn't work with fit_generator so had to separate them
    if path_to_batch is not None:
        
        # Files in batch
        files = listdir(path_to_batch)

        #shuffle file order to avoid overfitting
        shuffle(files)

        #Build model
        model = build_regressor()
        
        #Fit model to generated data
        '''When dealing with large datasets that can not fit into memory the best approach is to add the data to the model sequentially therefore only storing one file
        to memory at a time. It was not possible to merge the data into one file and run the model since memory would continue to increase
        during training. classification_neural_network_Merged26.csv is 734MB but used 14.2GB in memory when put into the 'dataset' variable.
        This is because '''
        history = model.fit_generator(generate_batches(files, path_to_batch),steps_per_epoch=10, epochs=100, callbacks=switch_case_callbacks(x=True))

    else:    
        #Build data
        X_test, Y_test, input_dim, X_train, Y_train, test_aligned_sequence, headers = build_data(sample)

        # According to Prof. Andrew Ng Coursera Course (Understanding mini-batch gradient descent) 
        # typically batch sizes are 64, 128, 256, 512 and 1024 (Powers of two)

        # Dictionaries are quickier than if statements and better optimized
        batch_dict = {
                (10000 > len(X_train) >= 5000): 512,
                (len(X_train) >= 10000)       : 1024,
                (len(X_train) < 5000)         : 10          
                }

        def switch_case_batch(x):
            return batch_dict.get(x)

        #Build Model
        model = KerasRegressor(build_fn=build_regressor, epochs=100, batch_size=switch_case_batch(x=True))

        #Accuracy is the 10 accuracies returned by k-fold cross validation
        #Most of the time k=10
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=10, shuffle=True)
        
        '''
        #Cross Validation
        accuracy = cross_val_score(estimator=model, X = X_train, y = Y_train, cv=kfold, n_jobs=n_cpu, verbose=verbose)

        #Mean accuracies and variance
        loss_mean = accuracy.mean()
        loss_mean = loss_mean*(-1)
        loss_variance = accuracy.std()
        print("\n----------\n")
        print("Cross Validation Results\nAverage loss: {0}\nloss function variance: {1}".format(loss_mean,loss_variance))
        print("\n----------\n")
        input("Press Enter to continue...")
        '''
        
        #Have to fit data to model again after cross validation
        history = model.fit(X_train, Y_train, callbacks=switch_case_callbacks(x=True))

    #Prediction on test data, needed to reshape array to subtract element wise
    y_pred = model.predict(X_test) if multivariate else model.predict(X_test).reshape(-1,1)
    y_difference = np.subtract(y_pred, Y_test)
    abs_pred = np.absolute(y_difference) 

    if multivariate:

        max_in_index, max_in_value = max(enumerate(abs_pred[:,1]), key=lambda p: p[1])
        max_del_index, max_del_value = max(enumerate(abs_pred[:,0]), key=lambda x: x[1])

        accuracy = ((np.sum(abs_pred <= 0.5)/len(Y_test))*50)
        mean_accuracy = y_difference.mean()

        visualisation(y_pred=y_pred[:,0].reshape(-1,1),Y_test=Y_test[:,0].reshape(-1,1), y_difference=y_difference[:,0].reshape(-1,1), filename='deletions.html')
        visualisation(y_pred=y_pred[:,1].reshape(-1,1),Y_test=Y_test[:,1].reshape(-1,1), y_difference=y_difference[:,1].reshape(-1,1), filename='insertions.html')
        print("Average model Accuracy: {0}".format(mean_accuracy))
        print("Largest insertion difference is {0:.2f} at position {1}\nModel predicted {2:.2f} whereas actual result was {3}".format(float(max_in_value), max_in_index, float(y_pred[:,1][max_in_index]), int(Y_test[:,1][max_in_index])))
        print("Largest deletion difference is {0:.2f} at position {1}\nModel predicted {2:.2f} whereas actual result was {3}".format(float(max_del_value), max_del_index, float(y_pred[:,0][max_del_index]), int(Y_test[:,0][max_del_index])))
        print("Model Performance: {0:.2f}%\nNumber of correct results (+/- 0.5): {1}/{2}".format(accuracy, np.sum(abs_pred <= 0.5), len(Y_test)*2))
   
    else:
        max_index, max_value = max(enumerate(abs_pred), key=lambda p: p[1])

        accuracy = ((np.sum(abs_pred <= 0.5)/len(Y_test))*100)
        mean_accuracy = y_difference.mean()

        visualisation(y_pred,Y_test, y_difference)
        #The closer to 0 the better the model
        print("Average model Accuracy: {0}".format(mean_accuracy))
        print("Largest difference is {0:.2f} at position {1}\nModel predicted {2:.2f} whereas actual result was {3}".format(float(max_value), max_index, float(y_pred[max_index]), int(Y_test[max_index])))
        print("Model Performance: {0:.2f}%\nNumber of correct results (+/- 0.5): {1}/{2}".format(accuracy, np.sum(abs_pred <= 0.5), len(Y_test)))

    print("Loading dataset and graphs in browser...")
    sf = SFrame(build_csv(test_aligned_sequence))
    sf.explore()
    sf.show()

#Results variance and mean, best possible score is 1.0 for variance
variance = explained_variance_score(Y_test, y_pred)
rmse_value = sqrt(mean_squared_error(Y_test, y_pred))

#Single prediction
if user_observation:

    if isfile(''.join(user_observation)):
        user_observation = ''.join(user_observation)

        data = pd.read_csv(user_observation)

        #headers = list(data)
        del data['ins/dels']

        #Encoding NHEJ
        data['NHEJ'] = labelencoder.fit_transform(data['NHEJ'])

        #Encoding UNMODIFIED
        data['UNMODIFIED'] = labelencoder.fit_transform(data['UNMODIFIED'])

        #Encoding frameshift
        data['frameshift'] = labelencoder.fit_transform(data['frameshift'])

        #Encoding HDR
        data['HDR'] = labelencoder.fit_transform(data['HDR'])

        data = data.iloc[:,:].values
        test_aligned_sequence, data = data[:,0:1],data[:,1:input_dim + 1]

        new_prediction = model.predict(sc.transform(data)).reshape(-1,1)
        pred_set = np.concatenate((test_aligned_sequence,data,new_prediction),axis=1)

        frame = pd.DataFrame(pred_set)
        frame.to_csv('{0}_batch_prediction.csv'.format(date), header=False, index=False)
        print("Predictions saved to file '{0}_batch_prediction.csv'".format(date))

    else:
        str_observation = ','.join(user_observation)
        fixed_observation = re.sub('true', '1', str_observation, flags=re.IGNORECASE)
        fixed_observation = re.sub('false', '0', fixed_observation, flags=re.IGNORECASE)

        #list comprehension didn't work
        #user_observation = [1 if x is 'True' else x for x in user_observation]
        #user_observation = [0 if x is 'False' else x for x in user_observation]

        list_observation = [float(i) for i in fixed_observation.split(',')]
        #Another square bracket around list observation so that it becomes a horizontal vector
        new_prediction = model.predict(sc.transform(np.array([list_observation])))
        print("\n----------\n")
        print ("Prediction is %s" % new_prediction)

#correlation matrix
#plt.matshow(frame.corr())

if cp:
    print("\n----------\n")
    input("Press Enter to continue...")

    frame = build_csv(test_aligned_sequence, sc)
    print("\n----------\n")
    save_csv = input("Do you wish to save csv of data? ([Y]/N) ")

    if save_csv.lower() is 'y' or 'yes':
        default = '{0}/machine-learning/predictions/regression/{1}_miseq_predictions'.format(home, date)
        file_name = input('Type path and file name [Press enter to keep default: {0}]:'.format(default))
        frame.to_csv(file_name or default ,index=False)


#feature importance
'''
Keras doesn't natively currently support feature importance
Probably to simpliest way to do this is to take the absoulte weights of the 
first layer of each variable and assume more important features have higher weights.
However the result may be misleading as it is completely possible for weights
to change in subsequent layers
'''

weights = build_regressor().get_weights()
layer_1_weights = np.sum(abs(weights[0]),axis=1).tolist()
feature_importance = np.column_stack((headers[1:18], layer_1_weights))
feature_importance = feature_importance[np.argsort(feature_importance[:, -1])][::-1]

#Opens tensorboard in browser if user saves new tensorboard
if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])









