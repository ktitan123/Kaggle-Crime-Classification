import pandas as pd
import os
import pylab as pl
import numpy as np

categories = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']
districts = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']


def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std

def get_streets(train):
	streets = []
	for index in range(len(train)):
		if '/' in train.iloc[index]['Address']:
			streets.append(train.iloc[index]['Address'].split('/')[1].strip())
		else:
			streets.append(train.iloc[index]['Address'].split(' ')[3].strip())
	
	streets = list(set(streets))
	print len(streets)

def preprocess(data):
	data['Date'] = data['Dates'].str.split(' ',expand = True)[0].astype(str)
	data['Time'] = data['Dates'].str.split(' ',expand = True)[1].astype(str)
	data['Month'] = data['Date'].str.split('-',expand = True)[1].astype(str)
	data['Day'] = data['Date'].str.split('-',expand = True)[2].astype(str)
	
	del data['Dates']
	return data


def encode(train,test = False):
	'''Selected columns Address, PdDistrict, X,Y, DayOfWeek, Time, '''
	if test == False:
		cat_encode = lambda x: categories.index(x)
		train['Category'] = train['Category'].apply(cat_encode)
		print len(categories)

	address = pd.unique(train.Address.ravel())
	#districts = pd.unique(train.PdDistrict.ravel())
	days = pd.unique(train.DayOfWeek.ravel())
	
	
	meanx = np.mean(train['X'])
	meany = np.mean(train['Y'])
	sdx = np.std(train['X'])
	sdy = np.std(train['Y'])
	day_encode = lambda x: np.where(days == x)[0][0]

	add_encode = lambda x:  1 if 'block' in x.lower() else 0 
	dist_encode = lambda x: [1 if x == d else 0 for d in districts]
	x_encode = lambda x: (((x-meanx)/sdx)**2)*1000
	y_encode = lambda x: (((x-meany)/sdy)**2)*1000
	
	time_encode = lambda x: int(x.split(':')[0])/2

	
	train['Address'] = train['Address'].apply(add_encode) 
	#print "address done"
	train['PdDistrict'] = train['PdDistrict'].apply(dist_encode) 
	#print "district done"
	train['X'] = train['X'].apply(x_encode)
	#print "x done"
	train['Y'] = train['Y'].apply(y_encode)
	#print "y done"
	train['DayOfWeek'] = train['DayOfWeek'].apply(day_encode)
	#print "day done"
	
	train['Time'] = train['Time'].apply(time_encode)
	
	for i in range(10):
		train['PD'+str(i)] = train['PdDistrict'].str[i]

	del train['PdDistrict']
	return train

'''
dir = os.getcwd()
train = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 10000)
address = pd.unique(train.Address.ravel())



train = preprocess(train)
train = encode(train)
print train.head()

print train.columns
input_cols= train.columns[range(2,3)+range(4,7)+range(8,21)]
print input_cols
print len(input_cols)
'''
