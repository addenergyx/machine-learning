# Machine Learning project for predicting insertion/deletion spread

[![Build Status](https://travis-ci.com/addenergyx/machine-learning.svg?token=wy5brpF4p1m4f7GJqQcL&branch=master)](https://travis-ci.com/addenergyx/machine-learning)

## Overview

This project uses machine learning to predict insertion/deletion spread of a given sequence from data provided by the Wellcome Sanger Institute. The deployed web application can be viewed [here.](https://indel-app.herokuapp.com/)

## Table of Contents

 1. [Running Webapp Locally](#running-webapp-locally)
 2. [Installation](#installation)
 3. [Deployment](#deployment)
 4. [Running Keras Models Locally](#running-keras-models-locally)
 5. [Built with](#built-with)
 6. [Description of files](#description-of-files)

## Running Webapp Locally
* Clone this repo
* Install requirements
* Run app.py
* Check  [http://localhost:5000](http://localhost:5000/)


Screenshots:

![ml-webapp](https://user-images.githubusercontent.com/22744727/55635730-5483c980-57b9-11e9-83bf-98dff6719d80.jpg)

![ml-webapp-prediction](https://user-images.githubusercontent.com/22744727/55635819-85fc9500-57b9-11e9-899a-7a5fd4cae6fe.jpg)

## Installation

### Clone repo
```shell
$ git https://github.com/addenergyx/machine-learning.git
```
### Install Webapp dependencies
```shell
$ pip install -r requirements.txt
```
### Run application
```shell
$ python app.py
```
### View application
Open [http://localhost:5000](http://localhost:5000/)

## Deployment
The model was deployed to production using **Heroku**. 

## Running Keras Models Locally
### **Need miseq data to run the models locally!!!**

### Data pre-processing script dependencies
```shell
$ sudo apt-get install make
$ sudo cpan App::cpanminus 
$ sudo cpanm Text::CSV::Slurp Try::Tiny Log::Log4perl Parallel::ForkManager Text::CSV::Separator Text::CSV_XS Getopt::Long
```

Vinna rna install
```shell
$ sudo apt-add-repository ppa:j-4/vienna-rna
$ sudo apt-get update
$ sudo apt-get install vienna-rna
```

**Advise using anaconda distribution because it comes with spyder and jupyter notebook and allows easy configuration of different environments. Jupyter notebok only used for visualization of data in a browser.**

### Clone repo
```shell
$ git https://github.com/addenergyx/machine-learning.git
```

### Regression Neural Network

##### Ensure the right data is pointing to the network in the config file .nn_config.yml or by using flags
```shell
$ vim ~/machine-learninng/.nn_config.yml
```
##### Run Neural Network
_If you are using a large dataset that cannot fit into memory split the data into separate csv files in one directory and point the --batch flag to the directory_
```shell
$ python ~/machine-learning/neural_network.py
```
### Classification Neural Network

##### Ensure the right data is pointing to the network in the config file .cn_config.yml or by using flags
```shell
vim ~/machine-learninng/.cn_config.yml
```
##### Run Neural Network
_If you are using a large dataset that cannot be encoded into memory, firstly encode the dataset and save that csv, then point the --batch flag to that file. This will read the file and add one line into memory and the model at a time using the python generator._
```shell
python ~/machine-learning/classification_neural_network.py
```
## Built with
* [Keras](https://keras.io/) - The main machine learning framework used. High level API for Neural Networks using Tensorflow in the backend.
* [Bootstrap](https://getbootstrap.com/) - Free front-end framework for faster and easier web development.
* [Scikit-learn](https://scikit-learn.org/stable/) - Used for data encoding and cross validation.
* [Heroku](https://www.heroku.com/) - Cloud platform used to deploy the model.
* [Flask](http://flask.pocoo.org/) - Python web framework.

## Description of files
Python scripts files:

|Filename|Description|
|--|--|
|classification_neural_network.py|Classification approach to use case, model used in the WebApp.|
|neural_network.py|Regression approach to use case.|
|.cn_config.yml|Configuration file for the classification model.|
|encode_data.py|Parse large dataset and produce encoded binary data for the classification model.|

Webapp files:

|Filename|Description|
|--|--|
|requirements.txt|All dependencies to run webapp on heroku.|
|app.py|Flask application.|
|Procfile|Specifies the commands that are executed by the app on startup.|
|tests/test.py|Unit testing on web application.|
|templates/home.html|Web app home page.|
|templates/results.html|Web app results page.|

Other files:

|Filename| Description |
|--|--|
|README.md|Markdown file description of the project.|
|data_preprocessing.pl|Feature extraction from miseq data.|
|classification_data_preprocessing.pl|Feature extraction from miseq data.|
|merge_csv.pl|Merges several data files into one.|
|static/css/styles.css|Styling for webapp.|

## TODO
* Tensorflow 2.0
* Google Charts
* Docker
* Figure filters
