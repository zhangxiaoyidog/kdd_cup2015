# coding=utf-8

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import svm

from sklearn import metrics


#随机森林算法模型
def randomForestpre(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)[:, 1]

    yyy=clf.predict(x_test)
    print yyy
    sss=metrics.accuracy_score(y_test,yyy)
    scores = roc_auc_score(y_test, y_pred)
    print "Accurcy randomForest",sss
    print "AUC randomForest_scores: ", scores


#线性回归模型
def linearRegressionpre(x_train, x_test, y_train, y_test):
    clf = linear_model.LinearRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    scores = roc_auc_score(y_test, y_pred)
    print "linearRegressionpre_scores: ", scores


def method1(X_train, X_test, y_train, y_test):
    # XGBoost自带接口
    params = {
        'eta': 0.3,
        'max_depth': 3,
        'min_child_weight': 1,
        'gamma': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'nthread': 12,
        'scale_pos_weight': 1,
        'lambda': 1,
        'seed': 27,
        'silent': 0,
        'eval_metric': 'auc'
    }
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_valid = xgb.DMatrix(X_test, label=y_test)
    d_test = xgb.DMatrix(X_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    # sklearn接口
    clf = xgb.XGBClassifier(
        n_estimators=30,
        learning_rate=0.3,
        max_depth=3,
        min_child_weight=1,
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=12,
        scale_pos_weight=1,
        reg_lambda=1,
        seed=27)

    model_bst = xgb.train(params, d_train, 30, watchlist, early_stopping_rounds=500, verbose_eval=10)

    y_bst = model_bst.predict(d_test)
    clf.fit(X_train,y_train)
    yyy=clf.predict(X_test)

    print("XGBoost   Accurcy Score : %f" % metrics.accuracy_score(y_test, yyy))
    print("XGBoost   AUC Score : %f" % roc_auc_score(y_test, y_bst))


#SVM模型，一种有自动调参，一种没有
def svc_clf(x_train, x_test, y_train, y_test):
    tuned_parameters = [{'kernel': ['poly'], 'C': [10, 500, 1200]},
                        {'kernel': ['linear'], 'C': [200, 500, 800]}]
    clfone = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring="roc_auc")

    clf = svm.SVC(C=2.0,kernel="rbf",probability=True,random_state=42)
    clf.fit(x_train, y_train)
    clfone.fit(x_train, y_train)
    print "Best parameters set found : "
    print clf.best_params_
    y_pred = clf.predict_proba(x_test)[:, 1]
    scores = roc_auc_score(y_test, y_pred)
    precisionone=clf.predict(x_test)
    precisiononescore=metrics.accuracy_score(y_test,precisionone)
    y_predone = clfone.predict_proba(x_test)[:, 1]
    scoresone = roc_auc_score(y_test, y_predone)

    precisiontwo=clfone.predict(x_test)
    precisiontwoscore = metrics.accuracy_score(y_test, precisiontwo)

    print "AUC svm clf scores...", scores

    print "Accurcy svm clf scores...", precisiononescore

    print "Accurcy svmone clf scores...", precisiontwoscore

    print "AUC svmone clf scores...", scoresone