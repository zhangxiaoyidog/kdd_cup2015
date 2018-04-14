# coding=utf-8
import pandas as pd
import numpy as np

from sklearn.preprocessing import scale
#特征提取以及一部分预处理操作

import preprocessing


#得到log_train表中的特征
def getEnrollFeature(filename):
    df1 = pd.read_csv(filename, usecols=[0, 1, 2, 3, 4, 5], converters={1: preprocessing.timecut})
    print df1.head()
    gp1 = df1.groupby("enrollment_id")

    df1.event = pd.factorize(df1.event)[0]
    eventdf = gp1.event.describe().unstack()

    timedf = gp1.time.describe().unstack()
    timedf["span"] = timedf["max"] - timedf["min"]
    timedf = timedf.drop(["count"], axis='columns')

    df1.source = pd.factorize(df1.source)[0]
    sourcedf = gp1.source.describe().unstack()
    sourcedf = sourcedf.drop(['count'], axis='columns')

    df1.object = pd.factorize(df1.object)[0]
    objectdf = gp1.object.describe().unstack()
    objectdf = objectdf.drop(['count'], axis=1)

    df1.temp = pd.factorize(df1.temp)[0]
    tempdf = gp1.temp.describe().unstack()
    tempdf = tempdf.drop(['count'],axis =1)

    data = df1.pivot_table("source", index='enrollment_id', columns="event", aggfunc='count', fill_value=0)

    data = pd.concat([data, eventdf], axis=1)
    data = pd.concat([data, timedf], axis=1)
    data = pd.concat([data, sourcedf], axis=1)
    data = pd.concat([data, objectdf], axis=1)
    data = pd.concat([data, tempdf],axis=1)

    data = data.fillna(0)

    return data

#得到date表中的特征
def getCourseFeature():
    dataftempenrol = pd.read_csv('data/enrollment_train.csv', usecols=[0, 2], converters={2: preprocessing.coursecut})
    dataftempdate = pd.read_csv('data/date.csv', converters={0: preprocessing.coursecut, 1: preprocessing.timecut, 2:preprocessing.timecut})
    dataframecourse = pd.merge(dataftempenrol, dataftempdate, on='course_id', how='outer')
    dataframecourse = dataframecourse.sort_index(by='enrollment_id')

    print dataframecourse.tail(10)
    return dataframecourse


#快速读取已经存起来的特征
def loadfeatures():
    dataenroll = pd.read_csv("fastfeature.csv")

    X = dataenroll.values
    X = scale(X)

    return X


#合并特征
def getallfeature():
    dataframecourse = getCourseFeature()
    dataenroll = getEnrollFeature("data/bb.csv")

    dataenroll["course_id"] = dataframecourse["course_id"].values
    dataenroll["from"] = dataframecourse["from"].values
    dataenroll["to"] = dataframecourse["to"].values
    dataenroll = dataenroll.fillna(0)

    print "save features"
    dataenroll.to_csv("fastfeature.csv" ,index=False)

    X = dataenroll.values
    X = scale(X)

    return X