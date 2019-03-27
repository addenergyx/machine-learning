#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request #WebApp Framework
from bokeh.plotting import figure
from bokeh.embed import components #Used to embed plots in webapp
import pickle #Used to get stored python objects from the trained classification model

#By doing this don't need to train the model in the webapp.

#Initialises flask application
app = Flask(__name__)

#Home page
@app.route('/')
def home():
	return render_template('home.html')

#loads model
def get_model():
    global model, graph
    model = load_model('/home/ubu/flask_apps/data/flask_classification_trained_model.h5')
    #Flask uses multiple threads. As a result the tensorflow model can not be loaded and 
    #used in the same thread. One workaround is to force tensorflow to use the gloabl default graph .
    #https://stackoverflow.com/questions/51127344/tensor-is-not-an-element-of-this-graph-deploying-keras-model
    graph = tf.get_default_graph() 
    print(" * Model loaded!")

def encodeData(userData):
        
    #https://stackoverflow.com/questions/9475241/split-string-every-nth-character
    pairing = 2
    userData = userData.upper()
    userData = [userData[i:i+pairing] for i in range(0, len(userData), pairing)]
        
    #User data needs to be encoded in the same way as the training data
    #cross ref sequence with headers list
        
    #Gets basepairs from original dataset to cross reference
    pickle_in = open("data/headers.pickle","rb")
    headers = pickle.load(pickle_in)
    
    #encodes user input to match encoding of training data
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

# Create main plot
def create_figure(pred_percentage):
    # create a new plot with a title and axis labels
    plot = figure(title="Chance of given in/del occuring", x_axis_label='Insertion/Deletion', y_axis_label='Likelyhood')
    
    #Gets all possible outcomes from trained model
    pickle_in = open("data/outcomes.pickle","rb")
    mylist = pickle.load(pickle_in)
    
    # add a line renderer with legend and line thickness
    plot.line(mylist, np.ravel(pred_percentage), legend="Percentage", line_width=2)
    return plot

#Reults page
@app.route("/predict", methods=['POST']) 
def predict():
    
    if request.method == 'POST':
        userData = request.form['comment']
        x = encodeData(userData)
        
        # Gets classification of indel from output dictionary of the trained model
        pickle_in = open("data/dict.pickle","rb")
        output_dict = pickle.load(pickle_in)
        
        # Computational Graph
        with graph.as_default():
            
            # Performs prediction
            user_proba = model.predict_proba(np.array(x).reshape(1,-1))
            encoded_user_top5 = (-user_proba).argsort()[:,0:5]
            user_top5_prob = np.sort(-user_proba)[:,0:5] * -100
            
            # Coverts encoded output to human readable output
            user_top5 = vfunc(encoded_user_top5, output_dict)
                        
            # Results to be posted to webapp
            posts = {}
            for A, B in zip(user_top5, user_top5_prob):
                B = [int(round(x)) for x in B ]
                posts = dict(zip(A, B))
                
            # Likelihood of each possible outcome
            pred_percentage = user_proba * 100
            pred_percentage = pred_percentage.reshape(-1,1)
            
        # Creates figure
        plot = create_figure(pred_percentage)
        
        # Embed plot into HTML via Flask Render
        # This returns javascript code and an HTML div section for rendering Bokeh plots within an HTML page.
        # These two components are then passed to the template via render_template().
        script, div = components(plot)

        # Returns top 5 results to html
        return render_template('result.html', 
                               posts=posts,
                               script=script, 
                               div=div,)
        
    
if __name__ == '__main__':
#    print(" * Loading Model")
#    model = load_model('/home/ubu/flask_apps/flask_classification_trained_model.h5')
#    print(" * Model loaded!")
     #app.run(host='0.0.0.0', port=5000)
     app.run(debug=True) #runs on default port 5000
