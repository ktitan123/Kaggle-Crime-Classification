from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import preprocess as pre

dir = os.getcwd()

categories = ['WARRANTS', 'OTHER OFFENSES', 'LARCENY/THEFT', 'VEHICLE THEFT', 'VANDALISM', 'NON-CRIMINAL', 'ROBBERY', 'ASSAULT', 'WEAPON LAWS', 'BURGLARY', 'SUSPICIOUS OCC', 'DRUNKENNESS', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'SECONDARY CODES', 'TRESPASS', 'MISSING PERSON', 'FRAUD', 'KIDNAPPING', 'RUNAWAY', 'DRIVING UNDER THE INFLUENCE', 'SEX OFFENSES FORCIBLE', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'ARSON', 'FAMILY OFFENSES', 'LIQUOR LAWS', 'BRIBERY', 'EMBEZZLEMENT', 'SUICIDE', 'LOITERING', 'SEX OFFENSES NON FORCIBLE', 'EXTORTION', 'GAMBLING', 'BAD CHECKS', 'TREA', 'RECOVERED VEHICLE', 'PORNOGRAPHY/OBSCENE MAT']

def write_results(rf,test_inp):
	category_map = {}
	result = open(dir+'/result.csv','w')
	
	result.write('Id,')
	
	result.write(str(categories[0]))
	for category in categories[1:]:
		result.write(',' + str(category))
	result.write('\n')
	for index in range(len(test_inp)):
		inp = test_inp.iloc[index]
		
		if index%10000==0:
			print index
		res = rf.predict_proba(inp)[0]
		
		result.write(str(index)+',')
		result.write(",".join(map(str,res)))
		result.write('\n')
	result.close()



train = pd.read_csv(dir+'/train.csv',sep = ',',header = 0)
train = train.sample(n = 450000)
validate = pd.read_csv(dir+'/train.csv',sep = ',',header = 0,nrows = 10000,skiprows = range(1,500000))

test = pd.read_csv(dir+'/test.csv',sep = ',',header = 0)

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



train_inp = train[input_cols]
train_op = train['Category']
test = test[input_cols]

rf = RandomForestClassifier(n_jobs=25,n_estimators = 35)
rf.fit(train_inp,train_op)

print "rf score ", rf.score(validate[input_cols],validate['Category'])

write_results(rf,test)