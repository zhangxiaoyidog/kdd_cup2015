# coding=utf-8
import pandas as pd
import numpy as np


#预处理文件，数值化特征
def timecut(x):
    rng = pd.date_range('2010-01-01', '2017-01-01')
    time_dict = pd.Series(np.arange(len(rng)), index=rng)
    x = x[:10]
    return time_dict[x]


def coursecut(x):
    dataftemp = pd.read_csv('data/date.csv', usecols=[0])
    course_map = pd.factorize(dataftemp.course_id)[1]
    course_cut = dict(zip(course_map, range(len(course_map))))
    return course_cut[x]