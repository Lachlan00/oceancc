"""
Ocean classification algorithm
------------------------------
Experimental program to classify two different water masses 
based on temperature and salinity profiles. 
Added time dependance variable
"""

print('_____Extracting_EAC_Ratio_Stack_____')

# import modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap # basemap tools
import argparse
import datetime
from datetime import datetime, timedelta # for working with datetimes
from netCDF4 import Dataset # reads netCDF file
from os import listdir
from os.path import isfile, join
from itertools import compress # for filteriung using boolean masks
import sys

# debugging function
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

# __Setup__

parser = argparse.ArgumentParser(description=__doc__)
# add a positional writable file argument
parser.add_argument('input_csv_file', type=argparse.FileType('r'))
# parser.add_argument('output_csv_file', type=argparse.FileType('w')) 
args = parser.parse_args()

# set range for data to collect
# updated to 200km from 50km on 20th of March 2018
xmin = 149.111697
xmax = 151.342103
ymin = -37.151332
ymax = -35.352688

# set colour scale variables
prob_max = 1.
prob_min = 0.

# lists to collect ratio data
yEAC = []
yTSW = []
xtime = []

# How many NetCDF files to use (set the range)
start = 0
end = 277 # for 277 files (zero index but range() does not include last number)

# randmoness seed (for shuffling)
np.random.seed(420)

# add to list witout append
def add(lst, obj, index): return lst[:index] + [obj] + lst[index:]

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

def grab_prob(time_idx):
    # make frame__idx an integer to avoid slicing errors
    frame_idx = int(time_idx)
    # get 'frame_time'
    frame_time = grab_sst_time(frame_idx)

    # set month of year
    DoY_val = int(frame_time.day)

    # Get probs
    temp = fh.variables['temp'][frame_idx,29,:,:] 
    salt = fh.variables['salt'][frame_idx,29,:,:]
    # ravel to 1D array
    temp1d = temp.ravel()
    salt1d = salt.ravel()
    # make data frame and replace NaNs
    data = {'var1': temp1d, 'var2': salt1d}
    data = pd.DataFrame(data=data)
    data['DoY'] = DoY_val
    data = data.fillna(-9999)
    # calculate probabilities
    probs = lr_model.predict_proba(data[['var1','var2','DoY']])
    prob_TSW, prob_EAC = zip(*probs)
    # convert tuples to list
    prob_EAC = list(prob_EAC)
    # sub back in nans
    prob_EAC = [x if x != 0.0 else np.nan for x in prob_EAC]
    # make 1D array
    prob_EAC = np.asarray(prob_EAC)
    # make 2D array
    prob_EAC = np.reshape(prob_EAC, (-1, 165))
    prob_EAC = prob_EAC[eta_rho,xi_rho]

    # calulcate ratio metric
    count_EAC = np.count_nonzero(prob_EAC > 0.5)
    count_TSW = np.count_nonzero(prob_EAC < 0.5)

    # add to lists
    return count_EAC, count_TSW, frame_time

# __Make_Model__

print('\nBuilding predictive model from training data...')

# Read in classification training data
csv_data = pd.read_csv(args.input_csv_file, parse_dates = ['datetime'], 
                        infer_datetime_format = True) #Read as DateTime obsject

# add "month of year" (DoY) to dataset 
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

# __Main_Program__

# get list of files in data directory
in_directory = "/Users/lachlanphillips/PhD_Large_Data/ROMS/Montague_subset"
file_ls = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]
file_ls = list(filter(lambda x:'naroom_avg' in x, file_ls))
file_ls = sorted(file_ls)

# get lats and lons
nc_file = in_directory + '/' + file_ls[0]
fh = Dataset(nc_file, mode='r')
lats = fh.variables['lat_rho'][:] 
lons = fh.variables['lon_rho'][:]
# combine to list of tuples
point_tuple = zip(lats.ravel(), lons.ravel())
point_list = []

j = 0
# iterate over tuple points and keep every point that is in box
for i in point_tuple:
    if ymin <= i[0] <= ymax and xmin <= i[1] <=xmax:
        point_list.append(j)
    j = j + 1

# make point list into tuple list of array coordinates
eta_rho = []
xi_rho = []
for i in point_list:
    eta_rho.append(int(i/165))
    xi_rho.append(int(i%165))

# get data for each file (may take a while)
for i in range(start, end):
    # import file
    nc_file = in_directory + '/' + file_ls[i]
    fh = Dataset(nc_file, mode='r')
    print('Extracting data from file: '+file_ls[i]+' | '+str(i+1)+' of '+ str(len(file_ls)))
    # fname = str(file_ls[i])[11:16]

    # extract time
    time = fh.variables['ocean_time'][:]

    # get data
    for i in range(0, len(time)):
        yEAC_proto, yTSW_proto, xtime_proto = grab_prob(i)
        yEAC, yTSW, xtime = add(yEAC, yEAC_proto, i), add(yTSW, yTSW_proto, i), add(xtime, xtime_proto, i)
    # close file
    fh.close()

# make dataframe from ratio metric lists
df_ratio  = {'yEAC': yEAC, 'yTSW': yTSW, 'xtime': xtime}
df_ratio = pd.DataFrame(data=df_ratio)
total_vals = df_ratio['yEAC'][0] + df_ratio['yTSW'][0]
df_ratio['yEACr'] = [x/total_vals for x in df_ratio['yEAC']]
df_ratio['yTSWr'] = [x/total_vals for x in df_ratio['yTSW']]
df_ratio['xmonth'] = [x.month for x in df_ratio['xtime']]
# remove duplicates
df_ratio = df_ratio.drop_duplicates('xtime').reset_index(drop=True)
df_ratio = df_ratio.sort_values('xtime').reset_index(drop=True)
print(df_ratio)

# save to csv
print('_____Please_Wait_____')
print('Writing ocean data to file...')
dt = datetime.now()
dt = dt.strftime('%Y%m%d-%H%M')
output_fn = '../montague_ratio-stack/output/' + 'mont-stack_' + str(start).zfill(3) + '-' + str(end).zfill(3) + '_' + dt + '.csv' 
df_ratio.to_csv(output_fn, index=False)

# make stack plot of results
print('Displaying plot - close plot to continue...')
plt.close('all')
fig, ax = plt.subplots()
yTSWr, yEACr, xtime = list(df_ratio['yTSWr']), list(df_ratio['yEACr']), list(df_ratio['xtime'])
ax.stackplot(xtime, yTSWr, yEACr)
plt.show()

print('_____Program_End_____')

