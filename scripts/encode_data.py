#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.utils.np_utils import to_categorical
from os.path import expanduser
import time #from time import strftime
import configargparse

#Multiplatform home environment
home = expanduser("~")
date = time.strftime("%d-%m-%y")

#Command-Line Options and Argument Parsing
config = configargparse.ArgParser(default_config_files=[home + '/machine-learning/.ed_config.yml'],
                                  config_file_parser_class=configargparse.YAMLConfigFileParser)
config.add_argument('--config', is_config_file=True, help='Configuration file path, command-line values override config file values')
config.add_argument('--data', action='store', default=home + '/data-preprocessing-worktree/data/merged_data/miseq_classification_data.csv', 
                    help="Data created by data_preprocessing.pl that is to large to fit into memory so needs to be encoded first (default: 'miseq_classification_data.csv') This file is 5GB, 63 million+ lines")

# Configuration variables
options = config.parse_args()
data = options.data

print("Loading Data To Memory...\n")
# dtype category is alot faster to go through then object and reduces memory 
# usage by 26%
dataset = pd.read_csv(data, dtype='category', error_bad_lines=False)
    
outputset = dataset.iloc[:,-1]
    
dataset = dataset.drop(dataset.columns[-1], axis=1)
 
print("Encoding input...\n") 
dataset = pd.get_dummies(dataset,drop_first=True)
length_X = len(dataset.columns) + 1
X = dataset.iloc[: , 0:length_X].values

print("Categorizing Possible Output...\n")
labelencoder_output = LabelEncoder()
Y_vector = labelencoder_output.fit_transform(outputset)
Y_vector = Y_vector.reshape(-1,1)
dummy_y = to_categorical(Y_vector)

print("Saving Encoded output to 'encoded_output.csv'...\n") 
np.savetxt("encoded_output.csv", dummy_y, delimiter=",", fmt='%d')
print("Saving Encoded Input to 'encoded_input.csv'...\n") 
np.savetxt("encoded_input.csv", X, delimiter=",", fmt='%d' )