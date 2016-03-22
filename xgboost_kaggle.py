import pandas as pd
import pylab as pl
import numpy as np
import os
import csv
from threading import Thread
import collections
import numpy as np
import preprocess as pre
import xgboost as xgb

dir = os.getcwd()

'''
https://github.com/tqchen/xgboost/blob/master/demo/multiclass_classification/train.py
'''

categories = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']

train = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 100000)
#train = train.sample(n = 450000)
validate = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 10000,skiprows = range(1,300000))

test = pd.read_csv(dir+'/test.csv',sep = ',',header = 0,nrows = 5000)

train = pre.preprocess(train)
train = pre.encode(train)
print "Training data   preprocessed"

validate = pre.preprocess(validate)
validate = pre.encode(validate)


test= pre.preprocess(test)
test = pre.encode(test,True)
print "Test data preprocessed"
input_cols= train.columns[range(2,3)+range(4,7)+range(8,21)]
print input_cols

train_inp = train[input_cols].values
train_op = train['Category'].values
validate_inp = validate[input_cols].values
validate_op = validate['Category'].values
test_inp = test[input_cols].values

predictions = []
xg_train = xgb.DMatrix( train_inp, label=train_op)
xg_test = xgb.DMatrix( validate_inp,label = validate_op )
watchlist = [ (xg_train,'train'), (xg_test, 'test') ]

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 8
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 39

num_round = 2

bst = xgb.train(param, xg_train, num_round, watchlist )

result = bst.predict(xg_test)
print result






