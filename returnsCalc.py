import pandas as pd
import numpy as np
from collections import Counter
from sklearn import svm, neighbors
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier



"""
the goal of this script is to predict if we should buy, sell, or hold a particular stock based off of the changes in value of every single other stock in the S&P 500
DISCLAIMERL: With this simple model we are going to get at best 40-50 percent accuracy

"""



def return_calc(ticker):
    df = pd.DataFrame.from_csv("master.csv")
    tickers = df.columns.values
    df.fillna(0, inplace = True)

    return_Period = 7 # any number you want
    for num in range(1,(return_Period+1) ):
        df[str(ticker) + "_" + str(num) ] = ((df[ticker].shift(-num) - df[ticker])/df[ticker])

    df.fillna(0, inplace = True)
    print (df)
    return tickers, df





def buy_sell_hold(*args): # where args is going to be the group of 7 columns that each represent the change in price over the course of the week
    cols = []
    for column in args:
        cols.append(column)
    requirement = 0.02 # the minimum percent change we need to start caring
    for c in cols: # for every single row in those columns (every single date )
        if c > requirement:
            return 1
        if c < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = return_calc(ticker)
    print "THIS is tickers"
    print tickers
    df[str(ticker) + "_target"] = list(map(buy_sell_hold, df[str(ticker) + "_1"],
                                                          df[str(ticker) + "_2"],
                                                          df[str(ticker) + "_3"],
                                                          df[str(ticker) + "_4"],
                                                          df[str(ticker) + "_5"],
                                                          df[str(ticker) + "_6"],
                                                          df[str(ticker) + "_7"]
                                            ))


    vals = df[str(ticker)+ "_target"].values.tolist() # list of the buy sell results in 1,0,-1
    str_vals = [str(i) for i in vals]

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[tick for tick in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)


    X = df_vals.values
    y = df[str(ticker) + "_target"].values  # the buy or sell signal column for every stock

    return X,y,df # where df is the full dataframe of prices for every stock for every day plus those additional 7 columns for the ticker in question
    # where X is a list of sublists where each sublist is the percentage change of every single stock for that day
    # where y is the list of buy, sell, or hold commands for the ticker in question based off of those last 7 cloumns

def ml(ticker):
    X,y,df = extract_featuresets(ticker)

    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = .25)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    # we are going to use skilit's ensemble learning package called Voting Cassifier to make predictions with three different models and then use them all to "vote" on the final result
    clf_ensemble = VotingClassifier([( "lsvc", svm.LinearSVC()), # the reason linear svc would make sesnse is that each of our x values is actually a list of values where each value is the price of a stock for that day, and the whole sublist is for every single stock
                                     ("knn", neighbors.KNeighborsClassifier()),
                                      ("rfor", RandomForestClassifier())])



   #      AFTER YOU FIT THE CLASSIFIER, YOU CAN PICKLE IT THEN  WHEN YOU WANT TO PREDICT WITH IT, JUST DO PICKLE LOAD AND THERES YOUR ANSWER. THIS WAY YOU DONT HAVE TO CONTINUALLY TRAIN THE MODEL


    clf_ensemble.fit(X_train, y_train)
    confidence =  clf_ensemble.score(X_test, y_test) # where the confidence measures the accuracy of this model
    predictions = clf_ensemble.predict(X_test)
    print ("predicted spread", Counter(predictions))
    print confidence
    return confidence

ml("GOOG") # enter whatever stock ticker in the s&p500 that you want
