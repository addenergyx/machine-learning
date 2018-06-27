#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
import time #from time import strftime
import configargparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.utils.np_utils import to_categorical
#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from sklearn.model_selection import cross_val_score, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint 
import re
import animation
import subprocess
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from keras.models import load_model
from turicreate import SFrame
import gc
import sys

'''
Put modules at the top instead of within functions as the latter will make 
calls to the function take longer. However can look into the cost of importing modules in
optional functions and if statments like sframe()  
'''

#Multiplatform home environment
home = expanduser("~")
date = time.strftime("%d-%m-%y")

#Command-Line Options and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.cn_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='Configuration file path, command-line values override config file values')
config.add_argument('-c','--cpu', action="store", type=int, default=-1, 
                    help="The number of CPUs to use to do the computation (default: -1 'all CPUs')")
config.add_argument('--sample', action='store', default=home + '/machine-learning/csv/classification/classification_50bp_miseq26_merged_data.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'classification_50bp_miseq26_merged_data.csv')")
config.add_argument('-t','--tensorboard', nargs='?', const='{0}/machine-learning/tensorboard/classification/{1}_classification_tensorboard'.format(home, date), 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const=home + "/machine-learning/snapshots/classification/%s_classification_trained_model.h5" % date, help="Save model to disk")
config.add_argument('-v','--verbose', action="store_true", help="Verbosity mode")
config.add_argument('-p','--predict', nargs='+',
                    help="Can parse a single observation or file containing multiple observations to make predictions on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0,1,0,0,68,77,39,94,68,2,1,-106.400001525879,26,278,0,1684,34.988572615")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/classification/13-06-18_classification_trained_model.h5', help="Path to saved model", nargs='?')
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
#dataset = pd.read_csv(sample)

#dask solves memory issue as it keeps it in disk
import dask.dataframe
print("Creating Dask Dataframe...\n")
daskset = dask.dataframe.read_csv('/home/ubuntu/data-preprocessing-worktree/data/merged_data/classification_52bp_miseq22_merged_data.csv')

#y column
# .iloc is purely integer-location based indexing for selection by position
# instead of by column name dataset["ins/dels"]
#outputset = dataset.iloc[:,-1]

outputset = daskset["ins/dels"]
daskset = daskset.drop(['ins/dels'], axis=1)
no_of_basepairs = len(daskset.columns)

# In memory
print("Computing output to memory...\n")
outputset = outputset.compute()

# Drop output column from dataset
#dataset = dataset.drop(dataset.columns[-1], axis=1)

#no_of_basepairs = len(dataset.columns)

#user_observation = ['TG','AG','AA','AA','CC','AA','AC','AG','GG','TG','TG','GC','AG','AA','GC','AG','CA','GG','AA','AG','AC','AA','AA','GA','GG']

# Appending user observation to dataset so it can be encoded in the same way as the dataset
# Can only pass data that has been encoded in the same way as the dataset to classifer.predict() 

if user_observation:
    # Single observations
    # Pandas
    #str_observation = ','.join(user_observation)
    #user_observation = [i for i in str_observation.split(',')]
    # Have to name columns in dataframes to be able to append rows 
    #columns = list(range(1,no_of_basepairs + 1))
    #dataset.columns = columns
    #user_series = pd.Series(user_observation, index=columns)
    #dataset = dataset.append(user_series, ignore_index=True)
    
    str_observation = ','.join(user_observation)
    user_observation = [i for i in str_observation.split(',')]
    # Have to name columns in dataframes to be able to append rows 
    columns = list(range(1, no_of_basepairs + 1))
    daskset.columns = columns
    user_series = pd.Series(user_observation, index=columns)
    daskset = daskset.append(user_series)
    
'''
#Using one hot encoding with drop first columns to avoid dummy variable trap
non_dropout_dataset = pd.get_dummies(dataset, drop_first=False)
dataset = pd.get_dummies(dataset,drop_first=True)
headers = list(dataset)
full_headers = list(non_dropout_dataset)
'''

# To use get_dummies with a dask dataframe, columns must be converted from a 
# column of strings to categorical data unlike with pd.get_dummies
print("Encoding Features...\n")
cat_dask = daskset.categorize()
non_dropout_daskset = dask.dataframe.get_dummies(cat_dask, drop_first=False)
daskset = dask.dataframe.get_dummies(cat_dask,drop_first=True)
headers = list(daskset.columns)
full_headers = list(non_dropout_daskset.columns)
#daskset.to_csv('machine-learning/tester')
del non_dropout_daskset
gc.collect()

#length_X = len(dataset.columns) + 1
length_X = len(daskset.columns) + 1

# Returns the encoded user observation and original dataset
if user_observation:
    #dataset,user_observation=dataset.drop(dataset.tail(1).index),dataset.tail(1)
    
    #dask df.drop does not work as of version 0.18.0
    #daskset, user_observation = daskset.drop([-1]),daskset.tail(1)
    user_observation = daskset.tail(1)
    # Turn dask dataframe into pandas dataframe
    # compute() takes awhile and uses alot of cpu
    print("Computing features to memory...\n")
    dataset = daskset.compute()
    dataset = dataset[:-1]
    print("Done")
else:
    print("Computing features to memory...\n")
    dataset = daskset.compute()
    print("Done")

del daskset 
gc.collect()

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

#Input layer
X = dataset.iloc[: , 0:length_X].values
dask_X = dask.array.from_array(X , (1000000))

del dataset
gc.collect()

###
#Number of inputs
input_dim = len(X[0])


#Encoding pairs
labelencoder_output = LabelEncoder()
Y_vector = labelencoder_output.fit_transform(outputset)
Y_vector = Y_vector.reshape(-1,1)

#np_outputset = np.reshape([outputset],(-1,1))

dask_Y_vector = dask.dataframe.from_array(Y_vector.astype(str))
Y_cat = dask_Y_vector.categorize()
dask_dummy_y = dask.dataframe.get_dummies(Y_cat)

#dask_dummy_y.to_csv('machine-learning/testery')

'''
import dask.array as da
from dask import compute

def to_dask_array(df):
    partitions = df.to_delayed()
    shapes = [part.values.shape for part in partitions]
    dtype = partitions[0].dtype

    results = compute(dtype, *shapes)  # trigger computation to find shape
    dtype, shapes = results[0], results[1:]

    chunks = [da.from_delayed(part.values, shape, dtype) 
              for part, shape in zip(partitions, shapes)]
    return da.concatenate(chunks, axis=0)
daskyy = to_dask_array(df=dask_dummy_y)

dask_dummy_y.compute()
'''
# one-vs-all
'''
This method does not one hot encode results
dask_Y_vector = dask.dataframe.from_array(outputset)
Y_cat = Y_vector.categorize()
dummy_y = dask.dataframe.get_dummies(Y_cat)
'''

# one-vs-all
dummy_y = to_categorical(Y_vector)

# Even though dummy_y is saved on disk memory usuage still increases, unsure why
dask_dummy_y = dask.array.from_array(to_categorical(Y_vector),(1000000))

#del Y_vector, outputset
gc.collect()

'''
#a = np.argmax(dummy_y, axis=1)
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import DummyEncoder
encoder = DummyEncoder()
yyy = encoder.fit_transform(Y_cat)
dummy_y = dummy_y.values
'''

#Spliting dataset
from dask_ml.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(dask_X, dask_dummy_y, random_state=42)

#del X, dummy_y, dask_X
gc.collect()

# Number of catagories
y_catagories = len(Y_test[0])
#number of rows - outcome_size = len(Y_test)

'''
#Feature scaling
don't need to feature scale because all data is binary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
###^^ Build_data ###

def sframe(frame):
    sf = SFrame(frame)
    sf.explore()
    sf.show()
    return

if saved_model is not None:
    classifier = load_model(saved_model)
    print("\nLoading model %s from disk" % saved_model)
    classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #X_test, Y_test = build_data(sample)[0:2]
    score = classifier.evaluate(X_test, Y_test, verbose=verbose)
    print("\n%s: %.2f" % (classifier.metrics_names[1], score[1]*100))
    '''print sframe'''    

else:
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
        classifier.add(Dense(units=y_catagories, kernel_initializer='uniform', activation='sigmoid'))
        
        #Complie model
        # Categorical crossentropy wouldn't be appropriate here as it looks at the probabilty
        # in relation of the other categories
        # Chose binary crossentropies because it is not a probability distribution over the labels 
        # but individual probabilities over every label    
        classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
        
        return classifier
    
    #Save loss function progress
    history = History()
    
    #Model stops if loss function is nan, this happens when there is missing data or memory error
    terminate_on_nan = TerminateOnNaN()
    
    #Save model
    Checkpoint = ModelCheckpoint(cp, monitor='acc', verbose=verbose, save_best_only=True)
    
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)
    
    classifier = KerasClassifier(build_fn=build_classifier, epochs=24, batch_size=1000000)
    
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
    
    #from memory_profiler import memory_usage
    #import dask.array
    #dask_X_train = dask.array.from_array(X_train, chunks=(100))
    #dask_Y_train = dask.array.from_array(Y_train, chunks=(100))
    
    #del X_train, Y_train
    gc.collect()
    
    start = time.time()
    #def f():
    classifier.fit(X_train, Y_train, callbacks=switch_case_callbacks(x=True))
    #   return
    
    #mem_usage = memory_usage(f, max_usage=True)
    #print('Maximum memory usage: %s' % max(mem_usage))
    
    end = time.time()
    time_completion = (end - start) / 60
    print('Model completion time: {0:.2f} minutes'.format(time_completion))

#probability of different outcomes
y_prob = classifier.predict_proba(X_test)

# result will probably always be a wildtype(0) 
y_pred = classifier.predict(X_test)
#y_pred = y_pred.reshape(-1,1)

#Index of top 5 indels 
encoded_top5 = (-y_prob).argsort()[:,0:5]

#Probability of top 5 indels
top5_prob = np.sort(-y_prob)[:,0:5] * -100

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
# To find more matches using list comprehension instead
#a_seq.index(0)

missing_base_dict = {}

# Finds all the first occurences of each base possibility
for x in range(1, no_of_basepairs + 1):
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
    results = [""] * (no_of_basepairs + 1)
    for i in range(1, (no_of_basepairs + 1)):
        if i in bases:
            results[i] = bases[i]
        else:
            results[i] = missing_base_dict.get(i).split("_")[1]
    seq = ''.join(results)
    return seq

def seq_to_crispr(x):
    crispr = str(x)[30:50]
    return crispr

'''
### old functions for mapping ###
def seq_map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

vfunc2 = np.vectorize(seq_map_func)
# This process takes a long time, should look into a quickier method
def old_mapping(x):
    bases = [i for i, e in enumerate(x) if e == 1]
    ordered_bases = vfunc2(bases,seq_dict)
        
    for i in range(1, no_of_basepairs + 1):
        #basestr = str(ordered_bases[i-1])
        try:
            if str(ordered_bases[i-1]).startswith('{0}_'.format(i)) is False:
            #if re.match('^{0}_'.format(i),ordered_bases[i-1]) is None: 
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
print("\nSequences mapped")
crisprs = np.apply_along_axis( seq_to_crispr, axis=1, arr=sequences).reshape(-1,1)
print("Crisprs found")
end = time.time()
lapse = end - start
wait.stop()
print('\nRemapping execution time: {0:.2f} seconds'.format(lapse))
print("\n----------\n")

'''
old_start = time.time()
sequences = np.apply_along_axis( old_mapping, axis=1, arr=X_test).reshape(-1,1)
old_end = time.time()
old_lapse = old_end - old_start
print('Old mapping method execution time: {0:.2f} seconds'.format(old_lapse))
'''

# Dataset of sequence with top 5 predicted in/del
pred_set = np.concatenate((crisprs,sequences,y_true,top5),axis=1)
pred_set_headers = ['Crispr','Reference Sequence', 'Actual Result','Predicted in/del','2nd','3rd','4th','5th']
frame = pd.DataFrame(pred_set, columns=pred_set_headers)
sframe(frame)

if user_observation is not None:    
    #user_observation = [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0]
    user_proba = classifier.predict_proba(np.array(user_observation))
    encoded_user_top5 = (-user_proba).argsort()[:,0:5]
    user_top5_prob = np.sort(-user_proba)[:,0:5] * -100
    user_top5 = vfunc(encoded_user_top5, output_dict)
    user_headers = ['Crispr','Reference Sequence','Most likely','%','2nd','2%','3rd','3%','4th','4%','5th','5%']
    user_sequence = np.apply_along_axis( mapping, axis=1, arr=user_observation).reshape(-1,1)
    user_crispr = np.apply_along_axis( seq_to_crispr, axis=1, arr=user_sequence).reshape(-1,1)
    
    arraysss = []
    for x in range(5):
        arraysss.extend([int(user_top5[:,x]), '{0}%'.format(int(user_top5_prob[:,x]))])

    user_set = np.concatenate((user_crispr, user_sequence, [arraysss]),axis=1)
    user_frame = pd.DataFrame(user_set,columns=user_headers)
    sframe(frame=user_frame)
        
    pred_percentage = user_proba * 100
    pred_percentage = pred_percentage.reshape(-1,1)
    
    # output to static HTML file
    output_file('classification.html')
    # create a new plot with a title and axis labels
    plot = figure(title="Chance of given in/del occuring", x_axis_label='x', y_axis_label='y')
    # add a line renderer with legend and line thickness
    plot.line(mylist, np.ravel(pred_percentage), legend="Percentage", line_width=2)
    # show the results
    show(plot)

if cp:
    print("\n----------\n")
    input("Press Enter to continue...\n")
    save_csv = input("Do you wish to save csv of data? ([Y]/N)")

    if save_csv.lower() is 'y' or 'yes':
        default = '{0}/machine-learning/predictions/classification/{1}_miseq_predictions'.format(home, date)
        file_name = input('Type path and file name [Press [ENTER] to keep default: {0}]:'.format(default))
        frame.to_csv(file_name or default ,index=False)

if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])





