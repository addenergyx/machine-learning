#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from flask import jsonify
from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

#loads model
def get_model():
    global model, graph
    model = load_model('/home/ubu/flask_apps/data/flask_classification_trained_model.h5')
    graph = tf.get_default_graph() 
    print(" * Model loaded!")

def encodeData(userData):
    
    #https://stackoverflow.com/questions/9475241/split-string-every-nth-character
    pairing = 2
    userData = userData.upper()
    userData = [userData[i:i+pairing] for i in range(0, len(userData), pairing)]
    
    #userData = ['TG','AG','AA','AA','CC','AA','AC','AG','GG','TG','TG','GC','AG','AA','GC','AG','CA','GG','AA','AG','AC','AA','AA','GA','GG','GA']
    
    #User data needs to be encoded in the same way as the training data
    #cross ref sequence with headers list
        
    #Gets basepairs from original dataset to cross reference
    pickle_in = open("data/headers.pickle","rb")
    headers = pickle.load(pickle_in)
    
    #encodes user input
    x = []
    for i in range(26):
        for j in headers:
            if int(j.split("_")[0]) == i+1:
                
                if j.split("_")[1] == userData[i]:
                    x.append(1)
                else:
                    x.append(0)
    return x

# Vectorise function to map back to true values
def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val

vfunc = np.vectorize(map_func)

##loading model into memory
print(" * Loading Keras model...")
get_model()



@app.route("/predict", methods=['POST']) 
def predict():
    
    if request.method == 'POST':
        userData = request.form['comment']
        #userData = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        x = encodeData(userData)
        
        #Gets classification of indel from model
        pickle_in = open("data/dict.pickle","rb")
        output_dict = pickle.load(pickle_in)
        
        #Computational Graph
        #Flask uses multiple threads. As a result the tensorflow model can not be loaded and 
        #used in the same thread. One workaround is to force tensorflow to use the gloabl default graph .
        #https://stackoverflow.com/questions/51127344/tensor-is-not-an-element-of-this-graph-deploying-keras-model
        with graph.as_default():
            user_proba = model.predict_proba(np.array(x).reshape(1,-1))
            encoded_user_top5 = (-user_proba).argsort()[:,0:5]
            #user_top5_prob = np.sort(-user_proba)[:,0:5] * -100
            user_top5 = vfunc(encoded_user_top5, output_dict)    
            return render_template('result.html', 
                                   prediction = user_top5[0][0], 
                                   p2 = user_top5[0][1], 
                                   p3 = user_top5[0][2],
                                   p4 = user_top5[0][3],
                                   p5 = user_top5[0][4],)

    
if __name__ == '__main__':
#    print(" * Loading Model")
#    model = load_model('/home/ubu/flask_apps/flask_classification_trained_model.h5')
#    print(" * Model loaded!")
     #app.run(host='0.0.0.0', port=5000)
     app.run(debug=True)

    
    
    
    
