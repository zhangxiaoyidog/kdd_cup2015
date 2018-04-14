# coding=utf-8
import pandas as pd
import numpy as np
import extractfeature
import model
from sklearn.cross_validation import train_test_split

#选取模型训练和预测
def getfeatures():
    X = extractfeature.loadfeatures()
    df3 = pd.read_csv('data/truth_train.csv', usecols=[1], names=["drop"])
    y = np.ravel(df3["drop"])
    return X, y

def dropPredict():
    print "getfeatures..."
    # X, y = loadPickleTrainData()
    X, y = getfeatures()

    print "training..."
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=148)
    model.method1(x_train, x_test, y_train, y_test)
    #model.xgboostpre(x_train, x_test, y_train, y_test)
    #model.randomForestpre(x_train, x_test, y_train, y_test)
    model.linearRegressionpre(x_train, x_test, y_train, y_test)
    #model.svc_clf(x_train, x_test, y_train, y_test)

dropPredict()