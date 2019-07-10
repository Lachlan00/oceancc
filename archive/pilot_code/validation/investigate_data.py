# Investigative plots

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # basemap tools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta # for working with datetimes
from netCDF4 import Dataset # reads netCDF file
from os import listdir
from os.path import isfile, join
import argparse
from scipy import stats
import cmocean
from random import randint


# __SETUP__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# how many files
start = randint(0, 276)
end = start + 1

# for 
prob_ls = np.empty(0)
time_ls = np.empty(0)
temp_ls = np.empty(0)
salt_ls = np.empty(0)

# range for floats
def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

# for getting time data
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
def get_prob(temp1d, salt1d, lr_model):
    # make data frame and replace NaNs
    data = {'var1': temp1d, 'var2': salt1d}
    data = pd.DataFrame(data=data)
    data['DoY'] = DoY_val
    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2','DoY']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)

    return prob_EAC

# __Make_Model__

# Read in classification training data
csv_data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

# add "Day of year" (DoY) to dataset 
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
# fit logistic regression to the training data
lr_model = LogisticRegression()
lr_model = lr_model.fit(train_data[['var1','var2','DoY']], np.ravel(train_data[['class']]))

# __Grab_Data__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# set output directory

# load one file to get lats and lons and data for initial plot
nc_file = in_directory + '/' + file_ls[1]
fh = Dataset(nc_file, mode='r')

# get all the probability arrays
# itterate through all files
idx = 0
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Grabbing data from: '+file_ls[i]+' | '+str(i+1).zfill(3)+' of '+ str(len(file_ls)))

    # extract time
    time = fh.variables['ocean_time'][:]

    # iterate through all time steps
    for j in range(0, len(time)-29):
        frame_time = grab_sst_time(int(j))
        DoY_val = int(frame_time.day)
        # get data
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
        # get prob
        prob = get_prob(temp, salt, lr_model)
        # add data to lists
        temp_ls = np.concatenate([temp_ls,temp])
        salt_ls = np.concatenate([salt_ls,salt])
        prob_ls = np.concatenate([prob_ls,prob])
        time_ls = np.concatenate([time_ls,np.repeat(frame_time,len(temp))])
        idx += 1

    # close file
    fh.close()

# make dataframe
df  = {'temp':temp_ls, 'salt': salt_ls, 'prob':prob_ls}
df  = pd.DataFrame(data=df)

fig1 = plt.figure()
plt.scatter(df.salt,df.temp,c=df.prob, cmap='bwr', marker='.', s=2)
plt.colorbar()
plt.xlabel('Salinity', fontsize=14)
plt.ylabel('Temperature', fontsize=14)

plt.show()










