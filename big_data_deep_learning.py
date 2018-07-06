#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import expanduser
import time #from time import strftime
import configargparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
#from sklearn.model_selection import cross_val_score, KFold
from keras.callbacks import History, TensorBoard, TerminateOnNaN, ModelCheckpoint 
import re
import animation
import subprocess
#import matplotlib.pyplot as plt
import csv

'''
Put modules at the top instead of within functions as the latter will make 
calls to the function take longer. However can look into the cost of importing modules in
optional functions and if statments like sframe()  
'''

#Multiplatform home environment
home = expanduser("~")
date = time.strftime("%d-%m-%y")

#Command-Line Options and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.bd_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='Configuration file path, command-line values override config file values')
config.add_argument('--input', action='store', default='encoded_input.csv', 
                    help="Data to train and test model created by data_preprocessing.pl (default: 'classification_50bp_miseq26_merged_data.csv') If using batch file must be encoded input data.")
config.add_argument('-t','--tensorboard', nargs='?', const='{0}/machine-learning/tensorboard/classification/{1}_classification_tensorboard'.format(home, date), 
                    help="Creates a tensorboard of this model that can be accessed from your browser")
config.add_argument('-s','--save', nargs='?', const=home + "/machine-learning/snapshots/classification/%s_big_data_trained_model.h5" % date, help="Save model to disk [Default: {0}/machine-learning/snapshots/classification/{1}_big_data_trained_model.h5]".format(home,date))
config.add_argument('-p','--predict', nargs='+',
                    help="Can parse a single observation or file containing multiple observations to make predictions on. False = 0, True = 1. Must be in order: NHEJ,UNMODIFIED,HDR,n_mutated,a_count,c_count,t_count,g_count,gc_content,tga_count,ttt_count,minimum_free_energy_prediction,pam_count,length,frameshift,#Reads,%%Reads. For example: 0,1,0,0,68,77,39,94,68,2,1,-106.400001525879,26,278,0,1684,34.988572615")
config.add_argument('-l','--load', const=home + '/machine-learning/snapshots/classification/13-06-18_classification_trained_model.h5', help="Path to saved model", nargs='?')
config.add_argument('-o','--output', action='store', default='encoded_output.csv', 
                    help="File containing encoded output, must be supplied when using batch. Input file goes with --sample")

# Configuration variables
options = config.parse_args()

sample = options.input
path_to_tensorboard = options.tensorboard
cp = options.save
user_observation = options.predict
saved_model = options.load
encoded_output = options.output

def average(x, y):
  return int((x + y) / 2.0)

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=average(input_dim,y_categories), kernel_initializer='uniform',activation='relu',input_dim=input_dim))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=average(input_dim,y_categories), kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(units=y_categories, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])
    return classifier

history = History()
    
terminate_on_nan = TerminateOnNaN()
    
Checkpoint = ModelCheckpoint(cp, monitor='acc', save_best_only=True)
    
tensorboard = TensorBoard(log_dir=path_to_tensorboard, histogram_freq=0, write_graph=True, write_images=True)
        
def tensorboard_callback():
    callbacks=[history, tensorboard, terminate_on_nan, Checkpoint]
    print("\n----------")
    print("Saving tensorboard to {0}".format(path_to_tensorboard))
    print("----------\n")
    return callbacks

def default_callback():
    callbacks=[history, terminate_on_nan, Checkpoint]
    return callbacks

callbacks_dict = {
        path_to_tensorboard is None                         : default_callback,
        path_to_tensorboard is not None                     : tensorboard_callback,
        }

def switch_case_callbacks(x):
    return callbacks_dict[x]()

with open(sample,'rt') as x, open(encoded_output,'rt') as y:
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
                            steps_per_epoch=60000000, epochs=1, callbacks=switch_case_callbacks(x=True))

if path_to_tensorboard:
    subprocess.call(['tensorboard', '--logdir', path_to_tensorboard])









