
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time
import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("ticker", help = " please insert the ticker of any stock in the S&P500", type = str)
args = parser.parse_args()

ticker = args.ticker
directory = "morningcomplete2" + "/" + str(ticker) + ".csv"
#import matplotlib.pyplot as plt

""" # population script
priceList = []
with open("SP500.csv", "r") as spfile:
    for line in spfile:
        lister = line.split(",")
        price = lister[1]
        if(len(price) > 4):
            priceList.append(price)
print priceList

with open("closingSP.csv", "wb+") as closingfile:
    for price in priceList:
        closingfile.write(price)
"""


# files in morningcomplete2
def load_data(filename, normalise_window):
    dataset = []
    with open(filename, "r") as stockfile:
        stockfile.readline()  # skip first line
        for line in stockfile:
            lister = line.split(",")
            mediary = []
            for elem in lister:
                elem = elem.strip("/n")
                mediary.append(float(elem))
            dataset.append(mediary)
    if normalise_window:
        dataset= normalise_windows(dataset)

    dataset = np.array(dataset)
    rownumber = int(round(.9 * (dataset.shape[0])))
    print "$$$$$$$$$$$$$$$$$$$"
    print dataset

    train = dataset[:rownumber , : ]

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test  = dataset[rownumber:, :-1]
    y_test  = dataset[rownumber:, -1]

    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    lister =  [x_train, y_train, x_test, y_test]

    for elem in lister :
        print elem
        print "************"

    return lister

def normalise_windows(window): # this will give us prices that have been
    new_window = []
    for index, lister in enumerate(window):
        try:
            lister[0] = ((float(lister[0]) - float(window[index-1][0]))/ 100)
            new_window.append(lister)
        except:
            continue
    return new_window

def build_model(layers):
    model = Sequential()

    # FIRST LAYER
    model.add(Dense(

        input_dim = 4, #the dimension of the input vectors
        output_dim = layers[1], # the number of nodes in the first output layer
    )) # will return to next layer
    model.add(Dropout(0.2)) #this will help prevent overfitting by randomly cutting off some of the ties between neural networks

    # SECOND layer
    model.add(Dense(
        layers[2], # number of nodes in this layer
    )) # we don't wamt to feed sequences to another layer, we want to extract output from this layer

    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim = layers[3])) # this will be the final output that we are going to map using a linear activation, binary wouldn't make sense since we are not classifying anything
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss = "mse", optimizer = "rmsprop", metrics = ["accuracy"]) # we are using mean squared errors as the metric to optimize the model
    return model

layers = [4,50,100,1]
model =  build_model(layers)
x_train, y_train, x_test, y_test = load_data(directory, True)
model.fit(x_train, y_train, epochs = 10, batch_size = 1 )

predictions = model.predict(x_test)

scores = model.evaluate(x_test, y_test)
print "SCORES"
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
