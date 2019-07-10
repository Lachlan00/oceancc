# Scatter plot of machine learning TS profiles

import numpy as np
import pandas as pd
from netCDF4 import Dataset # reads netCDF file
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import cmocean # oceanogrpahy colorscales - https://matplotlib.org/cmocean/
import matplotlib.pyplot as plt

####################
# define functions #
####################

def grab_sst_time(time_idx):
    """
    gets datetime object for sst map projection
    """
    dtcon_days = time[time_idx]
    dtcon_start = datetime(1990,1,1) # This is the "days since" part
    dtcon_delta = timedelta(dtcon_days/24/60/60) # Create a time delta object from the number of days
    dtcon_offset = dtcon_start + dtcon_delta # Add the specified number of days to 1990
    frame_time = dtcon_offset
    return frame_time

# get probs
def get_prob(temp, salt, lr_model, DoY_val):
    # ravel to 1D array
    temp1d = temp.ravel()
    salt1d = salt.ravel()
    # make data frame and replace NaNs
    data = {'var1': temp1d, 'var2': salt1d}
    data = pd.DataFrame(data=data)
    data['DoY'] = DoY_val
    data = data.fillna(-9999)
    # scale data
    data['var1'] = scaler_temp.transform(data[['var1']])
    data['var2'] = scaler_salt.transform(data[['var2']])
    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2','DoY']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)
    # sub back in nans
    prob_EAC = [x if x != 0.0 else np.nan for x in prob_EAC]
    # make 1D array
    prob_EAC = np.asarray(prob_EAC)

    return prob_EAC

###############
# make models #
###############

# __Make_Model__

# Read in classification training data
csv_data = pd.read_csv('../train/CARS/CARS_output/CARS_data_20180408.csv', parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

# add "month of year" (MoY) to dataset 
csv_data['DoY'] = [int(x.day) for x in csv_data['datetime']]

# make training dataset model
# create data points for training algorithm
# 1 = classA (EAC), 0 = classB (TS)
var1 = list(csv_data['temp'])
var2 = list(csv_data['salt'])
DoY = list(csv_data['DoY'])
water_class = list(csv_data['class'])
# make data frame
train_data = {'var1': var1, 'var2': var2, 'DoY': DoY, 'class': water_class}
train_data = pd.DataFrame(data=train_data)
# replace current data strings with binary integers
train_data['class'] = train_data['class'].replace(to_replace='EAC', value=1)
train_data['class'] = train_data['class'].replace(to_replace='BS', value=0)

###################
# Standarise Data #
###################

# get fits to be used for later scaling
scaler_temp = preprocessing.StandardScaler().fit(train_data[['var1']])
scaler_salt = preprocessing.StandardScaler().fit(train_data[['var2']])
# scale training dataset
train_data['var1'] = scaler_temp.transform(train_data[['var1']])
train_data['var2'] = scaler_salt.transform(train_data[['var2']])

###################

# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2','DoY']], np.ravel(train_data[['class']]))

##################
# Grab Plot Data #
##################

# how many files to process?
start = 0
end = 12

# for 
prob_ls = []
time_ls = []
temp_ls = []
salt_ls = []

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

i = 101
# import file
nc_file = in_directory + '/' + file_ls[i]
fh = Dataset(nc_file, mode='r')
print('Grabbing data from: '+file_ls[i]+' | '+str(i+1).zfill(3)+' of '+ str(len(file_ls)))
fname = str(file_ls[i])[11:16]
# extract time
time = fh.variables['ocean_time'][:]

# iterate through file
j = 29
# grab data
temp = fh.variables['temp'][j,29,:,:]
salt = fh.variables['salt'][j,29,:,:]
# remove masks
temp = np.ma.filled(temp.astype(float), np.nan)
salt = np.ma.filled(salt.astype(float), np.nan)
# ravel
temp = temp.ravel()
salt = salt.ravel()
# remove NaNs
temp = temp[~np.isnan(temp)]
salt = salt[~np.isnan(salt)]
# calculate probabilities
frame_time = grab_sst_time(j)
DoY_val = int(frame_time.day)
prob = get_prob(temp, salt, lr_model, DoY_val)

print('Generating plot...')
# plotting function


plt.scatter(salt, temp, c=prob, alpha=0.7, cmap=cmocean.cm.balance)
plt.title('EAC Probability Profile\n'+str(frame_time))
plt.xlabel('Salinity (PSU)')
#plt.locator_params(axis='x', nbins=10)
plt.ylabel('Temperature (℃)')
plt.show()










