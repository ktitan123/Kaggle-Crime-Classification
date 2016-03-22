import pandas as pd
#import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression 
import pylab as pl
import numpy as np
import os
import csv
from threading import Thread
import collections


categories = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']

'''
columns
[u'Category', u'Descript', u'DayOfWeek', u'PdDistrict', u'Resolution',
       u'Address', u'X', u'Y', u'Date', u'Time'],
      dtype='object')

https://www.kaggle.com/smerity/sf-crime/fighting-crime-with-keras/code
http://blog.yhat.com/posts/logistic-regression-and-python.html
'''



def preprocess(data):
	data['Date'] = data['Dates'].str.split(' ',expand = True)[0].astype(str)
	data['Time'] = data['Dates'].str.split(' ',expand = True)[1].astype(str)
	del data['Dates']
	return data


def encode(train,test = False):
	'''Selected columns Address, PdDistrict, X,Y, DayOfWeek, Time, Resolution'''
	if test == False:
		cat_encode = lambda x: categories.index(x)
		train['Category'] = train['Category'].apply(cat_encode)
		print len(categories)

	address = pd.unique(train.Address.ravel())
	districts = pd.unique(train.PdDistrict.ravel())
	days = pd.unique(train.DayOfWeek.ravel())
	
	
	meanx = np.mean(train['X'])
	meany = np.mean(train['Y'])
	day_encode = lambda x: np.where(days == x)[0][0]
	
	add_encode = lambda x: 1 if 'block' in x.lower() else 0 
	dist_encode = lambda x: np.where(districts == x)[0][0]
	x_encode = lambda x: ((x-meanx)**2)*10000
	y_encode = lambda x: ((x-meany)**2)*10000
	
	time_encode = lambda x: int(x.split(':')[0])/2

	
	train['Address'] = train['Address'].apply(add_encode) 
	print "address done"
	train['PdDistrict'] = train['PdDistrict'].apply(dist_encode) 
	print "district done"
	train['X'] = train['X'].apply(x_encode)
	print "x done"
	train['Y'] = train['Y'].apply(y_encode)
	print "y done"
	train['DayOfWeek'] = train['DayOfWeek'].apply(day_encode)
	print "day done"
	
	train['Time'] = train['Time'].apply(time_encode)
	
	return train


def write_results(categories):
	category_map = {}
	result = open(dir+'/result.csv','w')
	
	result.write('Id,')
	
	result.write(str(categories[0]))
	for category in categories[1:]:
		result.write(',' + str(category))
	result.write('\n')






dir = os.getcwd()

train1 = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 350000)
train2 = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 350000,skiprows = range(1,200000))

validate = pd.read_csv(dir+'/test.csv',sep = ',',header = 0)



write_results(categories)

train1 = preprocess(train1)
train1 = encode(train1)
print "Training data  1 preprocessed"

train2 = preprocess(train2)
train2 = encode(train2)
print "Training data  2 preprocessed"



validate = preprocess(validate)
validate = encode(validate,True)
print "Validation preprocessed"

result = open(dir+'/result.csv','a')


input_cols= train1.columns[range(2,4)+range(5,8)+range(9,10)]

print input_cols
train_inp1 = train1[input_cols]
train_inp2 = train2[input_cols]

train_op1 = train1['Category']
train_op2 = train2['Category']



test_inp = validate[input_cols]

logit1 = LogisticRegression()
logit1.fit(train_inp1,train_op1)

logit2 = LogisticRegression()
logit2.fit(train_inp2,train_op2)



print  "Fitting done"



for index in range(len(test_inp)):
	res1 = logit1.predict_proba(test_inp.iloc[index])
	res2 = logit2.predict_proba(test_inp.iloc[index])
	
	res1  = [x/sum(res1[0]) for x in res1[0]]
	res2  = [x/sum(res2[0]) for x in res2[0]]
	
	res = []
	for i in range(len(res1)):
		res.append((res1[i] + res2[i]  )/2.0)
	if index%50000 == 0:
		print index
	result.write(str(index)+',')
	result.write(",".join(map(str,res)))
	result.write('\n')










