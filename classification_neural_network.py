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
#import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from keras.models import load_model
from turicreate import SFrame #Currently no python3.7 support
import gc
import csv
#import pickle #Module for saving Objects by serialization

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
                    help="Data to train and test model created by data_preprocessing.pl (default: 'classification_50bp_miseq26_merged_data.csv') If using batch file must be encoded input data.")
config.add_argument('-t','--tensorboard', nargs='?', const='{0}/machine-learning/tensorboard/classification/{1}_classification_tensorboard'.format(home, date), 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const='{0}/data/machine-learning-data/machine-learning/snapshots/classification/{1}_classification_trained_model.h5'.format(home, date), help="Save model to disk")
config.add_argument('-v','--verbose', action="store_true", help="Verbosity mode")
config.add_argument('-p','--predict', nargs='+',
                    help="Can parse a single observation or file containing multiple observations to make predictions on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0,1,0,0,68,77,39,94,68,2,1,-106.400001525879,26,278,0,1684,34.988572615")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/classification/13-06-18_classification_trained_model.h5', help="Path to saved model", nargs='?')
config.add_argument('-b','--batch', action="store_true", help="Batch mode")
config.add_argument('-o','--output', nargs='?', const=home + 'encoded_output.csv', 
                    help="File containing encoded output, must be supplied when using batch. Input file goes with --sample")

# Configuration variables
options = config.parse_args()

n_cpu = options.cpu
sample = options.sample
path_to_tensorboard = options.tensorboard
verbose = options.verbose
cp = options.save
user_observation = options.predict
saved_model = options.load
batch = options.batch
encoded_output = options.output

print("\n")
print(options)
print("\n----------\n")
print(config.format_values())
print("\n----------\n")

if batch is False:
    #Classification Neural Network
    
    ### Dataset ###
    #Preparing data
    
    #Importing dataset
    print("Loading Data To Memory...\n")
    #dataset = pd.read_csv(sample)
    # dtype category is alot faster to go through then object and reduces memory 
    # usage by 26%
    dtypes = {'1':'category','2':'category','3':'category','4':'category','5':'category','6':'category',
              '7':'category','8':'category','9':'category','10':'category','11':'category','12':'category',
              '13':'category','14':'category','15':'category','16':'category','17':'category','18':'category',
              '19':'category','20':'category','21':'category','22':'category','23':'category','24':'category',
              '25':'category','26':'category','ins/dels':'int'} 
    dataset = pd.read_csv(sample, dtype=dtypes, error_bad_lines=False)
    
    #y column
    # .iloc is purely integer-location based indexing for selection by position
    # instead of by column name dataset["ins/dels"]
    outputset = dataset.iloc[:,-1]
    
    # Drop output column from dataset
    # Uses quite alot of memory when executed but returns it after
    dataset = dataset.drop(dataset.columns[-1], axis=1)
    
    no_of_basepairs = len(dataset.columns)
    
    #user_observation = ['TG','AG','AA','AA','CC','AA','AC','AG','GG','TG','TG','GC','AG','AA','GC','AG','CA','GG','AA','AG','AC','AA','AA','GA','GG']
    
    # Appending user observation to dataset so it can be encoded in the same way as the dataset
    # Can only pass data that has been encoded in the same way as the dataset to classifer.predict() 
    if user_observation:
        # Single observations
        str_observation = ','.join(user_observation)
        user_observation = [i for i in str_observation.split(',')]
        # Have to name columns in dataframes to be able to append rows 
        columns = list(range(1,no_of_basepairs + 1))
        dataset.columns = columns
        user_series = pd.Series(user_observation, index=columns)
        # This method makes a copy of the dataset and doesn't return memory back to 
        # system even when deleted. Taking up a considerable amount of memory when 
        # using a big dataset
        dataset = dataset.append(user_series, ignore_index=True)
        
    #Using one hot encoding with drop first columns to avoid dummy variable trap
    print("Encoding features...\n")
    # Pairs are encoded because strings can not be directly fed to a machine learning model
    # as machine learning models are based on mathematical equations
    non_dropout_dataset = pd.get_dummies(dataset, drop_first=False)
    full_headers = list(non_dropout_dataset)
    
    del non_dropout_dataset
    gc.collect()
    
    missing_base_dict = {}
    
    # Finds all the first occurences of each basepair possibility
    for x in range(1, no_of_basepairs + 1):
        key = x
        value = next (i for i in full_headers if re.match(str(x),i))
        missing_base_dict[key] = value
    
    dataset = pd.get_dummies(dataset,drop_first=True)
    headers = list(dataset)
    
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
        
    # Number of catagories
    #y_categories = len(Y_test[0])
    y_categories = len(mylist)
    #number of rows - outcome_size = len(Y_test)
    
    # Input layer
    X = dataset.iloc[: , 0:length_X].values
    
    # When dataframes are deleted the memory that was allocated to them can't be used
    # by the same program it was executed from unlike numpy arrays
    del dataset
    gc.collect()
    
    # Number of inputs
    input_dim = len(X[0])
    
    # Encoding output
    # Unlike other Deep Learning frameworks When modeling multi-class classification 
    # problems in keras using neural networks, it is best to reshape the output 
    # from a vector to a binary matrix. Keras does not use integer labels for 
    # crossentropy loss, unless using sparse_crossentropy_loss
    # to_categorical doesn't work with -ve numbers so have to label encode the in/del
    # column first
    print("Categorizing Possible Output...\n")
    labelencoder_output = LabelEncoder()
    Y_vector = labelencoder_output.fit_transform(outputset)
    Y_vector = Y_vector.reshape(-1,1)
    
    # one-vs-all
    # uses too much memory
    dummy_y = to_categorical(Y_vector)
    
    del Y_vector, outputset
    gc.collect()
    
    # Save encoded data to csv, large datasets need to be encoded first 
    # then close and reopen the program with the new dataset  
    # this will take a while
    # Char is the smallest integer data type using only 1 btye compared to int which is 2 or 4 bytes
    # As the file only contains 0 or 1 char would be optimal for storage
    #np.savetxt("encoded_output.csv", dummy_y, delimiter=",", fmt='%d')
    #np.savetxt("encoded_input.csv", X, delimiter=",", fmt='%d' )
    
    #from numpy import argmax
    #a = argmax(dummy_y, axis=1)
    
    #Spliting dataset
    # When using large datasets test size can be reduced. For example if a dataset 
    # has 10 million lines, taking out 3 million for testing would be overkill. 1 million or
    # even 500,000 would be enough for validation
    
    #outputset = np.reshape([outputset],(-1,1))
    
    print("Spliting testing and training data...\n")  
    X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=42)
    
    del X, dummy_y
    gc.collect()

'''
#Feature scaling
don't need to feature scale because all data is binary
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

def sframe(frame):
    sf = SFrame(frame)
    sf.explore()
    sf.show()
    return

def average(x, y):
  return int((x + y) / 2.0)

if saved_model is not None:
    classifier = load_model(saved_model)
    print("\nLoading model %s from disk" % saved_model)
    classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #X_test, Y_test = build_data(sample)[0:2]
    score = classifier.evaluate(X_test, Y_test, verbose=verbose)
    print("\n%s: %.2f" % (classifier.metrics_names[1], score[1]*100))
    #classifier.predict()
    '''print sframe'''    

else:
    #Neural Network architecture
    def build_classifier():
        #Initialising neural network
        classifier = Sequential()
        
        #Input layer and first hidden layer with dropout
        classifier.add(Dense(units=average(input_dim,y_categories), kernel_initializer='uniform',activation='relu',input_dim=input_dim))
        classifier.add(Dropout(rate=0.1))
        
        #Hidden layer 2 with dropout
        classifier.add(Dense(units=average(input_dim,y_categories), kernel_initializer='uniform', activation='relu'))
        classifier.add(Dropout(rate=0.1))
        
        #Output layer
        classifier.add(Dense(units=y_categories, kernel_initializer='uniform', activation='sigmoid'))
        
        #Complie model
        # Categorical crossentropy wouldn't be appropriate here as it looks at the probabilty
        # in relation of the other categories
        # Chose binary crossentropies because it is not a probability distribution over the labels 
        # but individual probabilities over every label
        classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
        
        return classifier
    
    # Save loss function progress
    history = History()
    
    # Model stops if loss function is nan, this happens when there is missing data or memory error
    terminate_on_nan = TerminateOnNaN()
    
    # Save model
    Checkpoint = ModelCheckpoint(cp, monitor='acc', verbose=verbose, save_best_only=True)
    
    # Creates tensorboard
    tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)
    
    # Wrapper for sklearn, need this to use cross validation
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
    
    def default_callback():
        callbacks=[history, terminate_on_nan]
        return callbacks
    
    callbacks_dict = {
            path_to_tensorboard and cp is None                         : tensorboard_callback,
            cp and path_to_tensorboard is None                         : checkpoint_callback,
            all (var is not None for var in [path_to_tensorboard, cp]) : both_callback,
            not path_to_tensorboard and not cp                         : default_callback
            }
    
    def switch_case_callbacks(x):
        return callbacks_dict[x]()
    
    '''
    # Cross validation
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(classifier, X_train, Y_train, cv=kfold, n_jobs=-1)
    print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    '''
    if batch is False:
        start = time.time()
        #def f():
        classifier.fit(X_train, Y_train, callbacks=switch_case_callbacks(x=True))
        #   return
        #mem_usage = memory_usage(f, max_usage=True)
        #print('Maximum memory usage: %s' % max(mem_usage))
        end = time.time()
        time_completion = (end - start) / 60
        print('Model completion time: {0:.2f} minutes'.format(time_completion))
    
    else:  
                
        #To get input and output dimensions for model
        with open(sample,'rt') as x, open(encoded_output,'rt') as y:
            # Row count takes a while
            row_count = sum(1 for row in open(sample))
            input_reader = csv.reader(x)
            output_reader = csv.reader(y)
            first_input = next(input_reader)
            first_output = next(output_reader)
            input_dim = len(first_input)
            y_categories = len(first_output)
        
        #https://keras.io/models/sequential/#fit_generator
        def generator(encoded_input,encoded_output):
            while True:
                with open(encoded_input,'rt') as x, open(encoded_output,'rt') as y:
                    for a,b in zip(x,y):
                        inputa = [int(i) for i in a.split(",")]
                        outputb = [int(j) for j in b.split(",")]
                        X_train = np.array(inputa).reshape(1,-1)
                        Y_train = np.array(outputb).reshape(1,-1)
                        yield (X_train,Y_train)
                x.close()
                y.close()      
                
        model = build_classifier()
    
        model.fit_generator(generator(encoded_input=sample, encoded_output=encoded_output),
                            steps_per_epoch=row_count, epochs=1, callbacks=switch_case_callbacks(x=True))
                 

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
    yyy = np.array(user_observation)
    user_proba = classifier.predict_proba(np.array(user_observation))
    encoded_user_top5 = (-user_proba).argsort()[:,0:5]
    user_top5_prob = np.sort(-user_proba)[:,0:5] * -100
    user_top5 = vfunc(encoded_user_top5, output_dict)
    user_headers = ['Crispr','Reference Sequence','Most likely','%','2nd','2%','3rd','3%','4th','4%','5th','5%']
    user_sequence = np.apply_along_axis( mapping, axis=1, arr=user_observation).reshape(-1,1)
    user_crispr = np.apply_along_axis( seq_to_crispr, axis=1, arr=user_sequence).reshape(-1,1)
    
    arraysss = [] # Top 5 indels and precentage into list
    for x in range(5):
        arraysss.extend([int(user_top5[:,x]), '{0}%'.format(int(user_top5_prob[:,x]))])

    user_set = np.concatenate((user_crispr, user_sequence, [arraysss]),axis=1)
    user_frame = pd.DataFrame(user_set,columns=user_headers)
    #user_frame.to_csv('user_pred data.csv',header=False,index=False)
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


if user_observation is not None:
    print("\n----------\n")
    input("Press Enter to continue...\n")
    save_csv = input("Do you wish to save csv of predicted data? ([Y]/N)")

    if save_csv.lower() == 'y' or 'yes':
        default = '{0}/machine-learning/predictions/classification/{1}_miseq_predictions'.format(home, date)
        file_name = input('Type path and file name [Press [ENTER] to keep default: {0}]:'.format(default))
        user_frame.to_csv(file_name or default ,index=False)


if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])

#pickle used to save classification of data - movce to flask_train.py later
#file saved in same location as script running it
#pickle_out = open("dict.pickle","wb")
#pickle.dump(output_dict, pickle_out)
#pickle_out.close()
#pickle_out = open("headers.pickle","wb")
#pickle.dump(headers, pickle_out)
#pickle_out.close()
