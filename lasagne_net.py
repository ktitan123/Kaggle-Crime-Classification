from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from theano.tensor.nnet import sigmoid,softmax
import theano
import pandas as pd
import pylab as pl
import numpy as np
import os
import csv
from threading import Thread
import collections
import numpy as np
import preprocess as pre
import math
from sklearn.utils import shuffle
'''

http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

'''

categories = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']


	

def write_results(net,test_inp):
	category_map = {}
	result = open(dir+'/result.csv','w')
	
	result.write('Id,')
	
	result.write(str(categories[0]))
	for category in categories[1:]:
		result.write(',' + str(category))
	result.write('\n')
	for index in range(len(test_inp)):
		inp = np.asarray(test_inp.iloc[index], dtype=np.float32)
		if index % 50000 == 0:
			print index
		res = net.predict(np.asarray([inp]))
		x = []

		agg = float(sum(res[0]))
		res = [x/agg for x in res[0]]

		for ind in range(len(res)):
			if str(res[ind]) == 'nan':
				res[ind] = 0.025641


		result.write(str(index)+',')
		result.write(",".join(map(str,res)))
		result.write('\n')
	result.close()



dir = os.getcwd()

train = pd.read_csv(dir+'/train.csv',sep = ',',header = 0)
train = train.sample(n = 50000)
train = pre.preprocess(train)
train = pre.encode(train)

print "Preprocessing done"

test = pd.read_csv(dir+'/test.csv',sep = ',',header = 0)

test = pre.preprocess(test)
test = pre.encode(test,True)

input_cols= train.columns[range(2,3)+range(4,7)+range(8,21)]
print input_cols

train_inp = train[input_cols]
train_op = train['Category']
test = test[input_cols]



X = []
Y = []
for index in range(len(train_inp)):
	X.append(list(train_inp.iloc[index]))
	val = train_op.iloc[index]
	op_list = [0.0]*39
	op_list[val] = 1.00
	Y.append(op_list)

learning_rate = theano.shared(np.float32(0.1))

X = np.asarray(X, dtype=np.float32)
Y = np.asarray(Y, dtype=np.float32)

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 17),  # 96x96 input pixels per batch
    hidden1_num_units=64,
    dropout1_p=0.4, 
    dropout2_p=0.4,
    hidden2_num_units=64,
    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=39,  # 39 target values
    

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=learning_rate,
    update_momentum=0.9,

   

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=44,  # we want to train this many epochs
    verbose=1,
    )

X, Y = shuffle(X, Y, random_state=123)
net1.fit(X,Y)
write_results(net1,test)

