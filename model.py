# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.utils import shuffle

dataset = pd.read_csv('heart.csv')
#x=dataset.iloc[:,dataset.columns!='target']
x = dataset.iloc[:,dataset.columns != 'target'].values
#print(x)
y=dataset.iloc[:,dataset.columns=='target']
#print(y)
x,y=shuffle(x,y)
#print(y)
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=100)
model2.fit(x,y.values.ravel())
pickle.dump(model2, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
