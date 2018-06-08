#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
import time #from time import strftime
import configargparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint 
import re
import animation
import subprocess

'''
Putting moudles at the top instead of within functions as the latter will make 
calls to the function take longer. However can look into the cost of importing modules in
optional functions and if statments like sframe()  
'''

#Multiplatform home environment
home = expanduser("~")
date = time.strftime("%d-%m-%y")

#Command-Line Option and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.cn_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='Configuration file path, command-line values override config file values')
config.add_argument('-c','--cpu', action="store", type=int, default=-1, 
                    help="The number of CPUs to use to do the computation (default: -1 'all CPUs')")
config.add_argument('--sample', action='store', default=home + '/machine-learning/csv/classification_50bp_miseq26_merged_data.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'classification_50bp_miseq26_merged_data.csv')")
config.add_argument('-t','--tensorboard', nargs='?', const='{0}/machine-learning/logs/tensorboard/{1}_classification'.format(home, date), 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const=home + "/machine-learning/snapshots/%s_trained_model.h5" % date, help="Save model to disk")
config.add_argument('-v','--verbose', action="store_true", help="Verbosity mode")
config.add_argument('-p','--predict', nargs='+',
                    help="Can parse a single observation or file containing multiple observations to make predictions on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0,1,0,0,68,77,39,94,68,2,1,-106.400001525879,26,278,0,1684,34.988572615")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/03-05-18_best_model.h5', help="Path to saved model", nargs='?')
config.add_argument('-b', '--batch', nargs='?', const=home + '/machine-learning/smallsamplefiles', 
                    help="Path to directory containing multiple files with data in the correct format. Default: ~/machine-learning/smallsamplefiles/")

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

if (path_to_batch is not None and sample is not None ):
    # Batch flag will override sample
    sample = None

print("\n")
print(options)
print("\n----------\n")
print(config.format_values())
print("\n----------\n")

#Classification Neural Network

### Dataset ###
#Preparing data

#Importing dataset
dataset = pd.read_csv(sample)

#y column
# .iloc is purely integer-location based indexing for selection by position
# instead of by column name dataset["ins/dels"]
outputset = dataset.iloc[:,-1]

# Drop output column from dataset
dataset = dataset.drop(dataset.columns[-1], axis=1)

no_of_bases = len(dataset.columns)

#user_observation = ['TG','AG','AA','AA','CC','AA','AC','AG','GG','TG','TG','GC','AG','AA','GC','AG','CA','GG','AA','AG','AC','AA','AA','GA','GG']

# Appending user input to dataset so it can be formatted in the same way as the dataset
# Can only pasre data to classifier.predict encoded in the same way as the dataset 
'''
One issue with using this method for prediction based on user sequence 
is that predictions can't be made based on a loaded model. As a result the user
would have to wait for the model to be trained again to get the result.
This can take up to 10 minutes 
'''
if user_observation:
    # Single observations
    str_observation = ','.join(user_observation)
    user_observation = [i for i in str_observation.split(',')]
    # Have to name columns in dataframes to be able to append rows 
    columns = list(range(1,26))
    dataset.columns = columns
    user_series = pd.Series(user_observation, index=columns)
    dataset = dataset.append(user_series, ignore_index=True)

#Using one hot encoding with drop first columns to avoid dummy variable trap
non_dropout_dataset = pd.get_dummies(dataset, drop_first=False)
dataset = pd.get_dummies(dataset,drop_first=True)
headers = list(dataset)
full_headers = list(non_dropout_dataset)

length_X = len(dataset.columns) + 1

# Returns the encoded user observation and original dataset
if user_observation:
    dataset,user_observation=dataset.drop(dataset.tail(1).index),dataset.tail(1)

#ordered unique output possibilities
myset = set(outputset)
mylist = list(myset)
mylist.sort()

# Mapping encoded in/del to actual in/del 
output_dict = {}

for x in range(len(myset)):
    key = x
    value =  min(myset)
    myset.remove(value)
    output_dict[key] = value

'''
if user_observation:
    user_observation, X = dataset.iloc[-1:], dataset.iloc[:-1]
else:
'''

#Input layer
X = dataset.iloc[: , 0:length_X].values


input_dim = len(X[0])

#Encoding pairs

labelencoder_output = LabelEncoder()

Y_vector = labelencoder_output.fit_transform(outputset)

Y_vector = Y_vector.reshape(-1,1)

# one-vs-all
dummy_y = to_categorical(Y_vector)

#from numpy import argmax
#a = argmax(dummy_y, axis=1)

#Spliting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

# Number of catagories
y_catagories = len(Y_test[0])
#number of rows - outcome_size = len(Y_test)

#Feature scaling
'''
don't need to feature scale because all results are binary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
###^^ Build_data ###

#Neural Network architecture
def build_classifier():
    #Initialising neural network
    classifier = Sequential()
    
    #Input layer and first hidden layer with dropout
    classifier.add(Dense(units=200, kernel_initializer='uniform',activation='relu',input_dim=input_dim))
    classifier.add(Dropout(rate=0.1))
    
    #Hidden layer 2 with dropout
    classifier.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    
    #Output layer
    classifier.add(Dense(units=y_catagories, kernel_initializer='uniform', activation='softmax'))
    
    #Complie model
    # Categorical crossentropy wouldn't be appropriate here as it is the sum of 
    # binary crossentropies because it is not a probability distribution over the labels 
    # but individual probabilities over every label individually    
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

#Save loss function progress
history = History()

#Model stops if loss function is nan, this happens when there is missing data
terminate_on_nan = TerminateOnNaN()

#Save model
Checkpoint = ModelCheckpoint(cp, monitor='val_loss', verbose=verbose, save_best_only=True)

tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)

classifier = KerasClassifier(build_fn=build_classifier, epochs=24, batch_size=1000)

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

'''
# Cross validation
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(classifier, X_train, Y_train, cv=kfold, n_jobs=-1)
print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

start = time.time()
classifier.fit(X_train,Y_train, callbacks=switch_case_callbacks(x=True))
end = time.time()
time_completion = (end - start) / 60
print('Model completion time: {0}'.format(time_completion))

#probability of different outcomes
y_prob = classifier.predict_proba(X_test)

# result will probably always be a wildtype(0) 
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

seq_dict = {}

for x in range(len(headers)):
    key = x
    value = headers[x]
    seq_dict[key] = value

# Test sequence: TGAGAAAACCAAACAGGGTGTGGCAGAAGCAGCAGGAAAGACAAAAGAGG
#a_seq = [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
#b_seq = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
# For testing swap x in mapping function enumerate(x) with a_seq 
# and execute lines inside function (not the function itself) 

# Index only searches through the list for the first instance 
#To find more matches using list comprehension
#a_seq.index(0)

missing_base_dict = {}

# Finds all the first occurences of each base possibility
for x in range(1, no_of_bases + 1):
    key = x
    value = next (i for i in full_headers if re.match(str(x),i))
    missing_base_dict[key] = value
    
# Mapping sequence
def mapping(x):
    bases = {}
    for i, e in enumerate(x):
        if e == 1:
            parts = seq_dict[i].split("_")
            bases[int(parts[0])] = parts[1]
    results = [""] * 26
    for i in range(1, 26):
        if i in bases:
            results[i] = bases[i]
        else:
            results[i] = missing_base_dict.get(i).split("_")[1]
    seq = ''.join(results)
    return seq

'''
### old functions for mapping ###
def seq_map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

vfunc2 = np.vectorize(seq_map_func)
# This process takes a long time, should look into a quickier method
def old_mapping(x):
    bases = [i for i, e in enumerate(x) if e == 1]
    ordered_bases = vfunc2(bases,seq_dict)
        
    for i in range(1, no_of_bases + 1):
        #basestr = str(ordered_bases[i-1])
        try:
            #if str(ordered_bases[i-1]).startswith('{0}_'.format(i)) is False:
            if re.match('^{0}_'.format(i),ordered_bases[i-1]) is None: 
                ordered_bases = np.insert(ordered_bases, i-1, missing_base_dict.get(i))
        except IndexError:
            ordered_bases = np.append(ordered_bases, missing_base_dict.get(i))

    for base in range(len(ordered_bases)):
        ordered_bases[base] = re.sub("\d+_","",ordered_bases[base])
    seq = ''.join(ordered_bases)
    return seq
'''

print("\n----------\n")
wait = animation.Wait(text='Remapping data')
wait.start()
start = time.time()
sequences = np.apply_along_axis( mapping, axis=1, arr=X_test).reshape(-1,1)
end = time.time()
lapse = end - start
wait.stop()
print('Mapping medthod length of execution: {0}'.format(lapse))
print("\n----------\n")

'''
old_start = time.time()
sequences = np.apply_along_axis( old_mapping, axis=1, arr=X_test).reshape(-1,1)
old_end = time.time()
old_lapse = old_end - old_start
print('Old mapping medthod length of execution: {0}'.format(old_lapse))
'''

def sframe(frame):
    from turicreate import SFrame
    sf = SFrame(frame)
    sf.explore()
    sf.show()
    return

# Dataset of sequence with top 5 predicted in/del
pred_set = np.concatenate((sequences,y_true,top5),axis=1)
pred_set_headers = ['Sequence', 'Actual Result','Predicted Most Likely in/del','2nd','3rd','4th','5th']
frame = pd.DataFrame(pred_set, columns=pred_set_headers)
sframe(frame)

'''
could look into making the display more meaningful data such as a colour coding 
for the percentage given by the y_prob matrix or colour coding based on the 
range of the dataset similar to when viewing the top5 variable in spyder
''' 

if user_observation is not None:    
    #user_observation = [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
    user_proba = classifier.predict_proba(np.array(user_observation))
    encoded_user_top5 = (-user_proba).argsort()[:,0:5]
    user_top5 = vfunc(encoded_user_top5, output_dict)
    user_headers = ['Sequence','Most likely','2nd','3rd','4th','5th']
    user_sequence = np.apply_along_axis( mapping, axis=1, arr=user_observation).reshape(-1,1)
    user_set = np.concatenate((user_sequence,user_top5),axis=1)
    user_frame = pd.DataFrame(user_set,columns=user_headers)
    sframe(frame=user_frame)

if cp:
    print("\n----------\n")
    input("Press Enter to continue...")
    print("\n----------\n")
    save_csv = input("Do you wish to save csv of data? [Y/N] ")

    if save_csv.lower() is 'y' or 'yes':
        default = '{0}/machine-learning/predictions/classification/{1}_miseq_predictions'.format(home, date)
        file_name = input('Type path and file name [Press enter to keep default: {0}]:'.format(default))
        frame.to_csv(file_name or default ,index=False)


if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])





