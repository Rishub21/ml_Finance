
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
def load_data(filename, normalise_window, sequence_length):
    dataset = []
    final_dataset = []
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

    for index in range(len(dataset) - sequence_length):
        mediary  = []
        for index2 in range(index, index + sequence_length):
            mediary.append(dataset[index2])
        final_dataset.append(mediary)

    final_dataset = np.array(final_dataset)


    rownumber = int(round(.9 * (final_dataset.shape[0])))
    train = final_dataset[:rownumber,:,:]

    x_train = train[:, :, :-1]
    y_train = train[:, :, -1]
    x_test  = final_dataset[rownumber:, :, :-1]
    y_test  = final_dataset[rownumber:,:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    lister =  [x_train, y_train, x_test, y_test]

    #for elem in lister :
        #print elem
        #print "************"

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
    model.add(LSTM(
        input_dim=layers[0], # should be just one dimensional if we only have prices as our data input
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(50))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics = ["accuracy"])
    print("> Compilation Time : ", time.time() - start)
    return model

layers = [4,50,100,1]
model =  build_model(layers)
x_train, y_train, x_test, y_test = load_data(directory, True, 50)
print "////////////////////////"
print x_train.shape
print model.summary()
model.fit(x_train, y_train, epochs = 1, batch_size = 1 )

predictions = model.predict(x_test)
scores = model.evaluate(x_test, y_test)

print "SCORES"
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
