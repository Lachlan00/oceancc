"""
Ocean classification algorithm
------------------------------
Experimental program to classify two different water masses 
based on temperature and salinity profiles.
"""

# import modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm

# set randomness seed
np.random.seed(1010)

# create dummy data points for training algorithm
# 1 = classA (EAC), 0 = classB (TS)
# Class A
var1 = np.random.normal(20,1.5,500)
var2 = np.random.normal(38,1.5,500)
classA = {'var1': var1, 'var2': var2, 'class': 1}
classA = pd.DataFrame(data=classA)
# Class B
var1 = np.random.normal(14,1.5,500)
var2 = np.random.normal(32,1.5,500)
classB = {'var1': var1, 'var2': var2, 'class': 0}
classB = pd.DataFrame(data=classB)
# Merge data frames
train_data = pd.concat([classA, classB])

# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2']], np.ravel(train_data[['class']]))

# set new randomness seed
np.random.seed(4242)

# Create data to test classifier
# ClassA
var1 = np.random.normal(20,2,500)
var2 = np.random.normal(38,2,500)
classA_test = {'var1': var1, 'var2': var2, 'class': 1}
classA_test = pd.DataFrame(data=classA_test)
# ClassB
var1 = np.random.normal(14,2,500)
var2 = np.random.normal(32,2,500)
classB_test = {'var1': var1, 'var2': var2, 'class': 0}
classB_test = pd.DataFrame(data=classB_test)
# Merge and shuffle dataframes
test_dat = pd.concat([classA_test, classB_test])
test_dat = test_dat.iloc[np.random.permutation(np.arange(len(test_dat)))]
test_dat = test_dat.reset_index(drop=True)

# test on dummy data
probs = lr_model.predict_proba(test_dat[['var1','var2']])
classB_prob, classA_prob = zip(*probs)
test_result = {'classA_prob': classA_prob, 'classB_prob': classB_prob}
test_result = pd.DataFrame(data=test_result)
test_df = pd.concat([test_dat, test_result], axis=1)

print(test_df)

# check on scatter
color = test_df['class']
color = color.replace([0, 1], ['blue', 'red'])
plt.scatter(test_df['var2'],test_df['var1'],c=test_df['classA_prob'], cmap='plasma')
plt.colorbar()
plt.xlabel('Salinity', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.show()



