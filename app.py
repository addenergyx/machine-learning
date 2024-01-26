#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for, flash
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.embed import components
import pickle

app = Flask(__name__)
app.secret_key = b'\x19\xb7\x0c\x12z\x0b\x1a\xcd\xb4\xc7\x13\xaa\xd84R\x1e\xa0\x04\x8c\x02!\xdc\x8f%'


@app.route('/', methods=['GET', 'POST'])
def home():
    print(" * Model loaded!")
    return render_template('home.html')


def get_model():
    return load_model('data/flask_classification_trained_model.h5')


def encodeData(userData):
    pairing = 2
    userData = userData.upper()
    userData = [userData[i:i + pairing] for i in range(0, len(userData), pairing)]

    with open("data/headers.pickle", "rb") as pickle_in:
        headers = pickle.load(pickle_in)

    x = []
    for i in range(26):
        for j in headers:
            if int(j.split("_")[0]) == i + 1:
                if j.split("_")[1] == userData[i]:
                    x.append(1)
                else:
                    x.append(0)
    return x


def map_func(val, dictionary):
    return dictionary[val] if val in dictionary else val


def is_valid_DNA(dna):
    return len(set(dna.upper()) - {'A', 'C', 'G', 'T'}) == 0 and len(dna) == 52


def create_figure(pred_percentage):
    print(" * Creating fig...")

    plot = figure(title="Chance of given in/del occuring", x_axis_label='Insertion/Deletion', y_axis_label='Likelihood',
                  sizing_mode="scale_width", max_width=600 )

    with open("data/outcomes.pickle", "rb") as pickle_in:
        mylist = pickle.load(pickle_in)

    plot.vbar(mylist, top=np.ravel(pred_percentage), legend_label="Percentage", width=0.9)
    plot.add_tools(HoverTool(tooltips=[("LOCATION", "@x"), ("PREDICTION", "@top%")]))

    return plot


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        print(" * Prediction...")

        userData = request.form['comment']

        if not is_valid_DNA(userData):
            flash("Invalid Sequence! Ensure sequence is 52 bases long and includes only 'A' 'T' 'C' & 'G'")
            return redirect(url_for('home'))

        x = encodeData(userData)

        with open("data/dict.pickle", "rb") as pickle_in:
            output_dict = pickle.load(pickle_in)

        print(" * Loading Keras model...")
        user_proba = get_model().predict(np.array(x).reshape(1, -1))
        encoded_user_top5 = (-user_proba).argsort()[:, 0:5]
        user_top5_prob = np.sort(-user_proba)[:, 0:5] * -100
        vfunc = np.vectorize(map_func)
        user_top5 = vfunc(encoded_user_top5, output_dict)

        posts = {}
        for A, B in zip(user_top5, user_top5_prob):
            B = [int(round(x)) for x in B]
            posts = dict(zip(A, B))

        pred_percentage = user_proba * 100
        pred_percentage = pred_percentage.reshape(-1, 1)

        plot = create_figure(pred_percentage)
        script, div = components(plot)

        print(" * Prediction done...")

        return render_template('result.html', posts=posts, script=script, div=div)


if __name__ == '__main__':
    app.run(debug=True)
