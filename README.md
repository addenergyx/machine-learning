# Machine Learning project for predicting insertion/deletion spread

## Overview

This project uses machine learning to predict insertion/deletion spread of a given sequence from data provided by the Wellcome Sanger Institute. The deployed web application can be viewed [here.](https://indel-app.herokuapp.com/)

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
