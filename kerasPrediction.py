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
directory = "closingstock_CSV" + "/" + str(ticker) + ".csv"
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



def load_data(filename, sequence_length, normalise_window):

    sequence_length = sequence_length + 1
    data = []
    result = []
    with open(filename, "r") as timeseries:
        for line in timeseries:
            data.append(line.strip("\n")) # list of prices
    for index in range(len(data)- sequence_length):
        result.append(data[index: index  + sequence_length])


    if normalise_window:
        result = normalise_windows(result)




    result = np.array(result) # so we are making result a 2D array where each sublist is an interval of 50 prices

    row = round(0.9 * result.shape[0])

    train = result[:int(row), :] # all the columns, so each sublist is still 50 prices but we are only taking 90% of the lists so that we can train our model

    np.random.shuffle(train)

    x_train = train[:, :-1] # all the rows in train but not the last column, the 50th price in each interval
    y_train = train[:, -1] # all the rows, but only the last column


    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]



    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    print("#######################3")
    print x_test

    return [x_train, y_train, x_test, y_test]

def build_model(layers): # where layers will be a list of parameters that specify the shape of the neural net
    model = Sequential()

    # FIRST LAYER
    model.add(LSTM(

        input_dim = layers[0], #the number of input layers
        output_dim = layers[1], # the number of nodes in the first output layer
        return_sequences = True))
    model.add(Dropout(0.2)) #this will help prevent overfitting by randomly cutting off some of the ties between neural networks

    # SECOND layer
    model.add(LSTM(
        layers[2], # number of nodes in this layer
        return_sequences = False)) # we don't wamt to feed sequences to another layer, we want to extract output from this layer

    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim = layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss = "mse", optimizer = "rmsprop") # we are using mean squared errors as the metric to optimize the model


    return model

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

epochs = 1 # this is how many passes our model will have at the training examples, how many tries itll have to optimize
seq_len = 50

X_train, y_train, X_test, y_test = load_data(directory, seq_len, True)


print('> Data Loaded. Compiling...')
print X_test

model = build_model([1, 50, 100, 1])
model.fit(
    X_train,
	y_train,
	batch_size=512,
	nb_epoch=epochs,
	validation_split=0.05)

predictions = model.predict(X_test)
print"&&&&&&&&&&&&&&&&&"
print predictions
#predicted = lstm.predict_sequence_full(model, X_test, seq_
